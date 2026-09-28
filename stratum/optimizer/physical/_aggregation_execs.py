"""Physical implementations of ``AggregateOp`` (grouped and whole-object).

Same-shape backend-variant family: the concrete impls subclass the logical
``AggregateOp``.

The two backends split the work differently, which is the point of keeping the
logical op expression-based. Polars consumes an ``AggExpr`` directly -- one
expression list handed to ``group_by(...).agg(...)`` or ``select(...)``, so a
computed aggregation like ``SUM(a * b)`` stays a single kernel. Pandas has no
expression language, so the grouped path materialises each entry's row-wise child
into a working frame first and then reduces it per column.
"""
from __future__ import annotations

import pandas as pd

from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._column_expr import AllCols, Col, EvalContext
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


class AggregateExec(AggregateOp, PhysicalOp):
    """Physical base: shares the evaluation context and the output naming rule."""

    def _ctx(self, inputs: list, mode: str) -> EvalContext:
        return EvalContext(frame=inputs[0], inputs=inputs, mode=mode)

    def _entry_name(self, index: int) -> str:
        """Output name for entry ``index``.

        An explicit name wins. Otherwise the name comes from the entry's own
        expression, so ``groupby(k)["v"].sum()`` still produces a ``v`` and not an
        internal placeholder. Only a computed child with no name of its own falls
        back to a generated one.
        """
        name, agg = self.aggregations[index]
        if name is not None:
            return name
        return agg.child.name if isinstance(agg.child, Col) else f"_agg{index}"


@physical_impl(of=AggregateOp, backend="pandas")
class PandasAggregateOp(AggregateExec):
    def process(self, mode: str, inputs: list):
        ctx = self._ctx(inputs, mode)
        if not self.grouped:
            return self._reduce_whole(ctx)
        return self._reduce_grouped(ctx)

    def _reduce_whole(self, ctx: EvalContext):
        """A bare reduction: ``df.sum()`` / ``series.mean()``."""
        if self._is_wildcard_only():
            _, agg = self.aggregations[0]
            return agg.to_pandas(ctx)
        results = {self._entry_name(i): agg.to_pandas(ctx)
                   for i, (_, agg) in enumerate(self.aggregations)}
        if self.output_type is OutputType.SERIES:
            return pd.Series(results)
        return results

    def _reduce_grouped(self, ctx: EvalContext):
        keys = [expr.to_pandas(ctx) for expr in self.grouping]
        options = dict(self.options)
        options.pop("level", None)  # a level-based grouping is carried by `keys`
        if self._is_wildcard_only():
            # `.agg(func)` on the grouped frame keeps pandas' own column naming
            # and its exclusion of the grouping keys.
            _, agg = self.aggregations[0]
            grouped = ctx.frame.groupby(keys, **options)
            return getattr(grouped, agg.func)(**agg.params)
        # Materialise each entry's row-wise child, then reduce column by column.
        columns: dict = {}
        for index, (_, agg) in enumerate(self.aggregations):
            columns.setdefault(agg.child, f"_child{index}")
        work = pd.DataFrame({name: child.to_pandas(ctx)
                             for child, name in columns.items()})
        grouped = work.groupby(keys, **options)
        out = {}
        for index, (_, agg) in enumerate(self.aggregations):
            series = grouped[columns[agg.child]]
            out[self._entry_name(index)] = getattr(series, agg.func)(**agg.params)
        if self.output_type is OutputType.SERIES and len(out) == 1:
            # A single-column result keeps its name; `grouped[...]` carries the
            # internal working-frame name, so rename it back.
            name, series = next(iter(out.items()))
            return series.rename(name)
        return pd.DataFrame(out)

    def _is_wildcard_only(self) -> bool:
        """A single unnamed entry over every column, i.e. a plain ``.agg(func)``."""
        return (len(self.aggregations) == 1
                and self.aggregations[0][0] is None
                and isinstance(self.aggregations[0][1].child, AllCols))


@physical_impl(of=AggregateOp, backend="polars")
class PolarsAggregateOp(AggregateExec):
    def process(self, mode: str, inputs: list):
        ctx = self._ctx(inputs, mode)
        exprs = []
        for index, (name, agg) in enumerate(self.aggregations):
            expr = agg.to_polars(ctx)
            # A wildcard keeps one output per source column, so it must not be
            # collapsed under a single alias.
            if name is not None:
                expr = expr.alias(name)
            exprs.append(expr)
        if not self.grouped:
            return ctx.frame.select(exprs)
        keys = [expr.to_polars(ctx) for expr in self.grouping]
        # pandas sorts the group keys by default; polars neither sorts nor
        # preserves order unless asked, so both cases are made explicit.
        sort = self.options.get("sort", True)
        result = ctx.frame.group_by(keys, maintain_order=not sort).agg(exprs)
        if sort:
            result = result.sort(result.columns[:len(keys)])
        return result
