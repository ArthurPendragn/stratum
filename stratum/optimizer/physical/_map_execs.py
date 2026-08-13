"""Physical implementations of ``AssignMapOp`` (folded column-map).

Same-shape backend-variant family: the concrete impls subclass ``AssignMapOp``
(+ :class:`PhysicalOp`) and carry the backend-specific kernel. Each assignment
batch runs against the frame materialized by the preceding batch. The
backend-agnostic ``ColumnExpr`` folding and ``make_context`` stay on the logical
side.
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


@physical_impl(of=AssignMapOp, backend="pandas")
class PandasAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        for batch in self.batches:
            ctx = self.make_context(mode, [frame, *inputs[1:]])
            values = {
                name: expr.to_pandas(ctx)
                for name, expr in batch.items()
            }
            frame = frame.assign(**values)
        return frame


@physical_impl(of=AssignMapOp, backend="polars")
class PolarsAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        for batch in self.batches:
            ctx = self.make_context(mode, [frame, *inputs[1:]])
            columns = {}
            for name, expr in batch.items():
                result = expr.to_polars(ctx)
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    # An OperandLeaf can feed pandas data into a polars plan.
                    logger.warning(f"Converting pandas object to polars object for column {name}")

                    result = pl.from_pandas(result)
                elif isinstance(result, list):
                    # Polars treats a list passed through the keyword API as one
                    # list-valued scalar; assign semantics require a column.
                    result = pl.Series(result)
                columns[name] = result
            # The keyword API accepts expressions, series, arrays and scalars,
            # broadcasting the latter just like pandas.DataFrame.assign.
            frame = frame.with_columns(**columns)
        return frame
