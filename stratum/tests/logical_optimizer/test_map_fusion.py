import operator

import pandas as pd
import polars as pl
import pytest

import stratum as st
from stratum.optimizer._map_rewrites import (
    _build_fused_assign_map, _collect_assign_map_chain,
    _replace_assign_map_chain, fuse_assign_maps, match_assign_map_chain)
from stratum.optimizer._op_utils import validate_dag
from stratum.optimizer._optimize import OptConfig, optimize as build_plan
from stratum.optimizer.ir._column_expr import (
    BinOpExpr, Col, ColumnExpr, Const, DatetimeExpr, DtExpr, OperandLeaf,
    StrExpr)
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._ops import Op, OperandRef, OutputType
from stratum.optimizer.ir._projection_ops import ColumnProjectionOp
from stratum.optimizer.ir._selection_ops import SelectionOp
from stratum.optimizer.physical._impl_selection import (
    ImplementationSelector, bind_op)
from stratum.optimizer.physical._map_execs import (
    PandasAssignMapOp, PolarsAssignMapOp)
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.runtime._scheduler import SequentialScheduler
from stratum.tests.logical_optimizer.test_dataframe_ops import optimize


def _maps(ops):
    return [op for op in ops if isinstance(op, AssignMapOp)]


class _ExactBackendSelector(ImplementationSelector):
    """Select one requested frame backend without relying on global policy."""

    _BACKEND_AGNOSTIC = ("sklearn-skrub", "numpy")

    def __init__(self, backend):
        self.backend = backend

    def choose(self, op, candidates, ctx):
        for backend in (self.backend, *self._BACKEND_AGNOSTIC):
            matches = [
                candidate
                for candidate in candidates
                if candidate.backend_name == backend
            ]
            if len(matches) > 1:
                raise AssertionError(
                    f"Multiple supported {backend!r} implementations for {op!r}"
                )
            if matches:
                return matches[0]
        return None


def _evaluate(dag, config, backend):
    plan, split_pos, flagged_ops = build_plan(
        dag,
        config,
        env=dag.skb.get_data(),
        selector=_ExactBackendSelector(backend),
    )
    scheduler = SequentialScheduler(plan, split_pos, flagged_ops)
    return scheduler.evaluate(), plan


def _evaluate_fusion_modes(make_output, backend, unfused_map_count):
    """Evaluate a fresh copy of one pipeline with fusion disabled and enabled."""
    results = {}
    plans = {}
    physical_type = (
        PolarsAssignMapOp if backend == "polars" else PandasAssignMapOp
    )
    for fuse in (False, True):
        results[fuse], plans[fuse] = _evaluate(
            make_output(),
            OptConfig(dataframe_ops=True, fuse_assign_maps=fuse),
            backend,
        )

    assert len([
        op for op in plans[False] if isinstance(op, physical_type)
    ]) == unfused_map_count
    assert len([
        op for op in plans[True] if isinstance(op, physical_type)
    ]) == 1
    return results, plans


def _assert_frames_equal(left, right):
    """Compare results without erasing backend-specific schema or index state."""
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right)
    else:
        assert isinstance(left, pl.DataFrame)
        assert left.equals(right)


def _three_map_pipeline():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    source = st.as_data_op(frame)
    first = source.assign(x2=source["x"] * 2)
    second = first.assign(x4=first["x2"] * 2)
    return frame, second.assign(x8=second["x4"] * 2)


def _logical_assign(name, primary, *auxiliary):
    map_op = AssignMapOp(
        batches=({name: Col("x")},),
        inputs=[primary, *auxiliary],
    )
    for producer in map_op.inputs:
        producer.add_output(map_op)
    return map_op


def test_collect_assign_map_chain_finds_maximal_logical_chain():
    source = Op()
    first = _logical_assign("first", source)

    assert _collect_assign_map_chain(first) == (first,)

    second = _logical_assign("second", first)
    third = _logical_assign("third", second)
    downstream = Op(inputs=[third])
    third.add_output(downstream)

    assert _collect_assign_map_chain(first) == (first, second, third)


def test_collect_assign_map_chain_stops_at_branch_and_wrong_primary_input():
    source = Op()
    first = _logical_assign("first", source)
    second = _logical_assign("second", first)
    third = _logical_assign("third", second)
    branch = Op(inputs=[second])
    second.add_output(branch)

    assert _collect_assign_map_chain(first) == (first, second)

    other_source = Op()
    candidate = _logical_assign("candidate", other_source)
    third.outputs = [candidate]

    assert _collect_assign_map_chain(third) == (third,)


def test_collect_assign_map_chain_carries_canonical_source_across_links():
    source = Op()
    first = _logical_assign("first", source)
    second = _logical_assign("second", first)
    third = _logical_assign("third", second, source)

    assert _collect_assign_map_chain(first) == (first, second)
    assert _collect_assign_map_chain(third) == (third,)


def test_build_fused_assign_map_preserves_order_refs_inputs_and_metadata():
    source = Op()
    shared = Op()
    distinct = Op()
    first = AssignMapOp(
        batches=({
            "shared": OperandLeaf(OperandRef(1)),
            "stage": OperandLeaf(OperandRef(0)),
        },),
        inputs=[source, shared],
    )
    second = AssignMapOp(
        batches=({
            "shared_again": OperandLeaf(OperandRef(2)),
            "distinct": OperandLeaf(OperandRef(1)),
        },),
        inputs=[first, distinct, shared],
    )
    first.is_y = True
    second.is_X = True
    second.is_split_op = True

    fused = _build_fused_assign_map((first, second))

    assert fused.inputs == [source, shared, distinct]
    assert [list(batch) for batch in fused.batches] == [
        ["shared", "stage"],
        ["shared_again", "distinct"],
    ]
    assert [
        [ref.k for ref in expr.iter_operand_refs()]
        for batch in fused.batches
        for expr in batch.values()
    ] == [[1], [0], [1], [2]]
    assert fused.name == "assign: shared, stage, shared_again, distinct"
    assert fused.is_X is True
    assert fused.is_y is True
    assert fused.is_split_op is True


def test_replace_assign_map_chain_rewires_edges_and_preserves_siblings():
    source = Op()
    auxiliary = Op()
    first = AssignMapOp(
        batches=({"first": OperandLeaf(OperandRef(1))},),
        inputs=[source, auxiliary],
    )
    second = AssignMapOp(
        batches=({"second": OperandLeaf(OperandRef(1))},),
        inputs=[first, auxiliary],
    )
    unrelated = Op(inputs=[auxiliary])
    downstream = Op(inputs=[second])
    root = Op(inputs=[downstream, unrelated])
    source.outputs = [first]
    auxiliary.outputs = [first, second, unrelated]
    first.outputs = [second]
    second.outputs = [downstream]
    unrelated.outputs = [root]
    downstream.outputs = [root]
    chain = (first, second)
    fused = _build_fused_assign_map(chain)

    replaced_root = _replace_assign_map_chain(root, chain, fused)

    assert replaced_root is root
    assert source.outputs == [fused]
    assert auxiliary.outputs == [unrelated, fused]
    assert downstream.inputs == [fused]
    assert fused.outputs == [downstream]
    assert first.inputs == first.outputs == []
    assert second.inputs == second.outputs == []
    validate_dag(root)


def test_replace_assign_map_chain_replaces_tail_root():
    source = Op()
    first = _logical_assign("first", source)
    second = _logical_assign("second", first)
    chain = (first, second)
    fused = _build_fused_assign_map(chain)

    root = _replace_assign_map_chain(second, chain, fused)

    assert root is fused
    assert source.outputs == [fused]
    assert fused.outputs == []
    validate_dag(root)


def test_build_fused_assign_map_visits_each_expression_once():
    class CountingExpr(ColumnExpr):
        visits = 0

        def _key(self):
            return ()

        def remap_operand_refs(self, mapping):
            type(self).visits += 1
            return self

    source = Op()
    chain = []
    primary = source
    depth = 1_000
    for stage in range(depth):
        map_op = AssignMapOp(
            batches=({f"x{stage}": CountingExpr()},),
            inputs=[primary],
        )
        chain.append(map_op)
        primary = map_op

    fused = _build_fused_assign_map(tuple(chain))

    assert len(fused.batches) == depth
    assert CountingExpr.visits == depth


def test_fuse_two_maps_plan():
    frame = pd.DataFrame({"x": [1, 2]})
    source = st.as_data_op(frame)
    first = source.assign(x2=source["x"] * 2)
    output = first.assign(x4=first["x2"] * 2)

    maps = _maps(optimize(output, OptConfig(dataframe_ops=True)))

    assert len(maps) == 1
    assert maps[0].batches == (
        {"x2": BinOpExpr(operator.mul, Col("x"), Const(2))},
        {"x4": BinOpExpr(operator.mul, Col("x2"), Const(2))},
    )


def test_fuse_three_maps_reaches_fixed_point():
    _, output = _three_map_pipeline()

    maps = _maps(optimize(output, OptConfig(dataframe_ops=True)))
    assert len(maps) == 1
    assert maps[0].batches == (
        {"x2": BinOpExpr(operator.mul, Col("x"), Const(2))},
        {"x4": BinOpExpr(operator.mul, Col("x2"), Const(2))},
        {"x8": BinOpExpr(operator.mul, Col("x4"), Const(2))},
    )


def test_maximal_chain_fusion_allocates_one_replacement_and_detaches_chain():
    source = Op()
    source.output_type = OutputType.FRAME
    first = AssignMapOp(batches=({"x1": Col("x0")},), inputs=[source])
    second = AssignMapOp(
        batches=({"x2": BinOpExpr(operator.add, Col("x1"), Const(1))},),
        inputs=[first],
    )
    third = AssignMapOp(
        batches=({"x3": BinOpExpr(operator.add, Col("x2"), Const(1))},),
        inputs=[second],
    )
    source.outputs = [first]
    first.outputs = [second]
    second.outputs = [third]

    fused = fuse_assign_maps(third)

    assert fused is not first
    assert fused is not second
    assert fused is not third
    assert len(fused.batches) == 3
    assert fused.name == "assign: x1, x2, x3"
    assert source.outputs == [fused]
    assert first.inputs == first.outputs == []
    assert second.inputs == second.outputs == []
    assert third.inputs == third.outputs == []


def test_fused_map_rewires_downstream_consumer():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    source = st.as_data_op(frame)
    first = source.assign(x2=source["x"] * 2)
    second = first.assign(x4=first["x2"] * 2)
    projected = second[["x4"]]

    assert len(_maps(optimize(projected, OptConfig(dataframe_ops=True)))) == 1
    assert list(st._api.evaluate(projected)["x4"]) == [4, 8, 12]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_fused_chain_preserves_values(backend):
    _, output = _three_map_pipeline()

    result, _ = _evaluate(
        output,
        OptConfig(dataframe_ops=True),
        backend,
    )

    assert list(result["x2"]) == [2, 4, 6]
    assert list(result["x4"]) == [4, 8, 12]
    assert list(result["x8"]) == [8, 16, 24]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_overwrite_and_earlier_map_siblings_preserve_values(backend):
    frame = pd.DataFrame({"a": [1, 2, 3]})
    source = st.as_data_op(frame)
    first = source.assign(
        a=source["a"] * 10,
        doubled=source["a"] * 2,
    )
    output = first.assign(total=first["a"] + first["doubled"])

    result, _ = _evaluate(
        output,
        OptConfig(dataframe_ops=True),
        backend,
    )

    assert list(result["a"]) == [10, 20, 30]
    assert list(result["doubled"]) == [2, 4, 6]
    assert list(result["total"]) == [12, 24, 36]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_second_map_siblings_do_not_see_each_other(backend):
    frame = pd.DataFrame({"a": [1, 2, 3]})
    source = st.as_data_op(frame)
    first = source.assign(a=source["a"] * 10)
    output = first.assign(
        a=first["a"] + 1,
        snapshot=first["a"] * 2,
    )

    result, _ = _evaluate(
        output,
        OptConfig(dataframe_ops=True),
        backend,
    )

    assert list(result["a"]) == [11, 21, 31]
    assert list(result["snapshot"]) == [20, 40, 60]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_string_scalar_materialization_matches_unfused_execution(backend):
    results = {}
    plans = {}
    for fuse in (False, True):
        source = st.as_data_op(pd.DataFrame({"x": [1, 2]}))
        first = source.assign(s="a")
        output = first.assign(result=first["s"].str.upper())
        results[fuse], plans[fuse] = _evaluate(
            output,
            OptConfig(dataframe_ops=True, fuse_assign_maps=fuse),
            backend,
        )

    assert list(results[True]["result"]) == list(results[False]["result"])
    assert list(results[True]["result"]) == ["A", "A"]

    physical_type = (
        PolarsAssignMapOp if backend == "polars" else PandasAssignMapOp
    )
    assert len([op for op in plans[False] if isinstance(op, physical_type)]) == 2
    assert len([op for op in plans[True] if isinstance(op, physical_type)]) == 1


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_none_scalar_materialization_matches_unfused_execution(backend):
    results = {}
    plans = {}
    for fuse in (False, True):
        source = st.as_data_op(pd.DataFrame({"x": [1, 2]}))
        first = source.assign(a=None)
        output = first.assign(result=operator.eq(first["a"], None))
        results[fuse], plans[fuse] = _evaluate(
            output,
            OptConfig(dataframe_ops=True, fuse_assign_maps=fuse),
            backend,
        )

    assert list(results[True]["result"]) == list(results[False]["result"])
    expected = [None, None] if backend == "polars" else [False, False]
    assert list(results[True]["result"]) == expected

    physical_type = (
        PolarsAssignMapOp if backend == "polars" else PandasAssignMapOp
    )
    assert len([op for op in plans[False] if isinstance(op, physical_type)]) == 2
    assert len([op for op in plans[True] if isinstance(op, physical_type)]) == 1


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_numeric_overwrite_dependency_chain_matches_unfused_execution(backend):
    def make_output():
        source = st.as_data_op(pd.DataFrame({"a": [1, 2, 3]}))
        first = source.assign(
            a=source["a"] * 10,
            doubled=source["a"] * 2,
        )
        second = first.assign(
            a=first["a"] + 1,
            subtotal=first["a"] + first["doubled"],
        )
        return second.assign(total=second["a"] + second["subtotal"])

    results, _ = _evaluate_fusion_modes(make_output, backend, 3)

    _assert_frames_equal(results[True], results[False])
    assert list(results[True]["a"]) == [11, 21, 31]
    assert list(results[True]["subtotal"]) == [12, 24, 36]
    assert list(results[True]["total"]) == [23, 45, 67]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_empty_frame_matches_unfused_execution(backend):
    def make_output():
        frame = pd.DataFrame({"a": pd.Series(dtype="float64")})
        source = st.as_data_op(frame)
        first = source.assign(shifted=source["a"] + 1)
        return first.assign(scaled=first["shifted"] * 2)

    results, _ = _evaluate_fusion_modes(make_output, backend, 2)

    _assert_frames_equal(results[True], results[False])
    assert results[True].shape == (0, 3)
    assert list(results[True].columns) == ["a", "shifted", "scaled"]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_nan_and_null_values_match_unfused_execution(backend):
    def make_output():
        frame = pd.DataFrame({
            "a": [1.0, float("nan"), 3.0],
            "s": ["a", None, "c"],
        })
        source = st.as_data_op(frame)
        first = source.assign(
            shifted=source["a"] + 1,
            cleaned=source["s"].str.upper(),
        )
        return first.assign(
            scaled=first["shifted"] * 2,
            length=first["cleaned"].str.len(),
        )

    results, _ = _evaluate_fusion_modes(make_output, backend, 2)

    _assert_frames_equal(results[True], results[False])
    assert pd.isna(list(results[True]["scaled"])[1])
    assert pd.isna(list(results[True]["cleaned"])[1])
    assert pd.isna(list(results[True]["length"])[1])


def test_non_default_index_and_external_series_alignment_matches_unfused():
    def make_output():
        frame = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
        external = st.as_data_op(pd.Series(
            [300, 100, 200],
            index=[30, 10, 20],
            name="external",
        ))
        source = st.as_data_op(frame)
        first = source.assign(doubled=source["x"] * 2)
        second = first.assign(external=external)
        return second.assign(
            total=second["doubled"] + second["external"])

    results, _ = _evaluate_fusion_modes(make_output, "pandas", 3)

    _assert_frames_equal(results[True], results[False])
    assert list(results[True].index) == [10, 20, 30]
    assert list(results[True]["external"]) == [100, 200, 300]
    assert list(results[True]["total"]) == [102, 204, 306]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_nested_expressions_with_other_rewrites_match_unfused(backend):
    def make_output():
        frame = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [4, 3, 2, 1],
        })
        source = st.as_data_op(frame)
        first = source.assign(
            base=(source["a"] + source["b"]) * 2 - source["a"] ** 2,
        )
        second = first.assign(
            score=(first["base"] + first["a"] * 3) ** 2 // 2,
        )
        selected = second[second["score"] > 20]
        return selected[["base", "score"]]

    results, plans = _evaluate_fusion_modes(make_output, backend, 2)

    _assert_frames_equal(results[True], results[False])
    assert any(isinstance(op, SelectionOp) for op in plans[True])
    assert any(isinstance(op, ColumnProjectionOp) for op in plans[True])
    assert list(results[True].columns) == ["base", "score"]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_repeated_expression_chain_stays_linear(backend):
    depth = 16
    source = Op()
    source.output_type = OutputType.FRAME
    output = source
    for stage in range(1, depth + 1):
        previous = f"x{stage - 1}"
        next_map = AssignMapOp(
            batches=({
                f"x{stage}": BinOpExpr(
                    operator.add,
                    Col(previous),
                    Col(previous),
                )
            },),
            inputs=[output],
        )
        output.outputs = [next_map]
        output = next_map

    fused = fuse_assign_maps(output)
    bind_op(
        fused,
        PlanContext.from_flags(),
        selector=_ExactBackendSelector(backend),
    )
    frame = pd.DataFrame({"x0": [1, 2]})
    value = pl.from_pandas(frame) if backend == "polars" else frame
    result = fused.process("fit_transform", [value])

    assert len(fused.batches) == depth
    for stage, batch in enumerate(fused.batches, start=1):
        previous = f"x{stage - 1}"
        assert batch == {
            f"x{stage}": BinOpExpr(
                operator.add,
                Col(previous),
                Col(previous),
            )
        }
    assert list(result[f"x{depth}"]) == [2**depth, 2 ** (depth + 1)]


def test_overwrite_across_maps_preserves_materialization_boundary():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    source = st.as_data_op(frame)
    first = source.assign(a=source["a"] * 10)
    output = first.assign(b=first["a"] * 2)

    maps = _maps(optimize(output, OptConfig(dataframe_ops=True)))

    assert len(maps) == 1
    assert maps[0].batches == (
        {"a": BinOpExpr(operator.mul, Col("a"), Const(10))},
        {"b": BinOpExpr(operator.mul, Col("a"), Const(2))},
    )


def test_multi_consumer_map_is_not_fused():
    frame = pd.DataFrame({"x": [1, 2]})
    source = st.as_data_op(frame)
    first = source.assign(y=source["x"] + 1)
    second = first.assign(y=first["y"] * 2)
    output = first.skb.concat([second], axis=0)

    maps = _maps(optimize(output, OptConfig(dataframe_ops=True)))

    assert len(maps) == 2


def test_maps_across_selection_are_not_fused():
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})
    source = st.as_data_op(frame)
    first = source.assign(y=source["x"] * 2)
    filtered = first.head(2)
    output = filtered.assign(z=filtered["y"] * 2)

    ops = optimize(output, OptConfig(dataframe_ops=True))

    assert len(_maps(ops)) == 2
    assert len([op for op in ops if isinstance(op, SelectionOp)]) == 1


def test_flag_disables_fusion():
    _, output = _three_map_pipeline()

    maps = _maps(optimize(
        output,
        OptConfig(dataframe_ops=True, fuse_assign_maps=False),
    ))

    assert len(maps) == 3


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_operand_leaf_inputs_are_merged_and_remapped(backend):
    frame = pd.DataFrame({"x": [1, 2, 3]})
    source = st.as_data_op(frame)
    first_factor = st.as_data_op(2)
    second_factor = st.as_data_op(3)
    first = source.assign(x2=source["x"] * first_factor)
    output = first.assign(x6=first["x2"] * second_factor)

    result, plan = _evaluate(
        output,
        OptConfig(dataframe_ops=True),
        backend,
    )
    maps = _maps(plan)

    assert len(maps) == 1
    assert len(maps[0].inputs) == 3
    assert [ref.k for ref in maps[0].batches[0]["x2"].iter_operand_refs()] == [1]
    assert [ref.k for ref in maps[0].batches[1]["x6"].iter_operand_refs()] == [2]
    assert list(result["x6"]) == [6, 12, 18]


def test_shared_operand_leaf_is_deduplicated_by_identity():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    source = st.as_data_op(frame)
    factor = st.as_data_op(2)
    first = source.assign(x2=source["x"] * factor)
    output = first.assign(x4=first["x2"] * factor)

    maps = _maps(optimize(output, OptConfig(dataframe_ops=True)))

    assert len(maps) == 1
    assert len(maps[0].inputs) == 2
    assert [ref.k for ref in maps[0].batches[1]["x4"].iter_operand_refs()] == [1]


def test_original_source_frame_operand_prevents_rebinding_during_fusion():
    results = {}
    plans = {}
    for fuse in (False, True):
        source = st.as_data_op(pd.DataFrame({"x": [1, 2]}))
        first = source.assign(y=source["x"] * 10)
        output = first.assign(original=source)
        results[fuse], plans[fuse] = _evaluate(
            output,
            OptConfig(dataframe_ops=True, fuse_assign_maps=fuse),
            "pandas",
        )

    pd.testing.assert_frame_equal(results[True], results[False])
    assert list(results[True]["original"]) == [1, 2]
    assert len(_maps(plans[True])) == 2


def test_later_original_source_operand_splits_maximal_chain_without_rebinding():
    results = {}
    plans = {}
    for fuse in (False, True):
        source = st.as_data_op(pd.DataFrame({"x": [1, 2]}))
        first = source.assign(x=source["x"] * 10)
        second = first.assign(y=first["x"] + 1)
        output = second.assign(original=source)
        results[fuse], plans[fuse] = _evaluate(
            output,
            OptConfig(dataframe_ops=True, fuse_assign_maps=fuse),
            "pandas",
        )

    pd.testing.assert_frame_equal(results[True], results[False])
    assert list(results[True]["x"]) == [10, 20]
    assert list(results[True]["y"]) == [11, 21]
    assert list(results[True]["original"]) == [1, 2]
    assert len(_maps(plans[True])) == 2
    assert [len(map_op.batches) for map_op in _maps(plans[True])] == [2, 1]


def test_second_map_frame_operand_leaf_keeps_stage_relative_binding():
    source = Op()
    source.output_type = OutputType.FRAME
    first = AssignMapOp(batches=({"x": Col("x")},), inputs=[source])
    second = AssignMapOp(
        batches=({"whole_frame": OperandLeaf(OperandRef(0))},),
        inputs=[first],
    )
    source.outputs = [first]
    first.outputs = [second]

    assert match_assign_map_chain(first) == (first, second)
    fused = fuse_assign_maps(second)
    assert fused.batches == (
        {"x": Col("x")},
        {"whole_frame": OperandLeaf(OperandRef(0))},
    )


def test_physical_assign_maps_are_not_matched():
    source = Op()
    physical = PandasAssignMapOp(batches=({"x": Col("x")},), inputs=[source])

    assert _collect_assign_map_chain(physical) == ()
    assert match_assign_map_chain(physical) is None


def test_fused_map_copies_consumer_metadata_flags():
    source = Op()
    source.output_type = OutputType.FRAME
    first = AssignMapOp(batches=({"x": Col("x")},), inputs=[source])
    second = AssignMapOp(
        batches=({"y": BinOpExpr(operator.mul, Col("x"), Const(2))},),
        inputs=[first],
    )
    second.is_X = True
    second.is_split_op = True
    source.outputs = [first]
    first.outputs = [second]

    fused = fuse_assign_maps(second)

    assert isinstance(fused, AssignMapOp)
    assert fused is not second
    assert fused.is_X is True
    assert fused.is_y is False
    assert fused.is_split_op is True


def test_fused_map_preserves_producer_metadata_flags():
    source = Op()
    source.output_type = OutputType.FRAME
    first = AssignMapOp(batches=({"x": Col("x")},), inputs=[source])
    second = AssignMapOp(
        batches=({"y": BinOpExpr(operator.mul, Col("x"), Const(2))},),
        inputs=[first],
    )
    first.is_X = True
    first.is_y = True
    source.outputs = [first]
    first.outputs = [second]

    fused = fuse_assign_maps(second)

    assert fused.is_X is True
    assert fused.is_y is True
    assert fused.is_split_op is False


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_fused_datetime_and_string_chain_preserves_values(backend):
    frame = pd.DataFrame({
        "ts": ["2021-03-05", "2023-11-20"],
        "s": ["  ab  ", " c "],
    })
    source = st.as_data_op(frame)
    first = source.assign(
        parsed=source["ts"].skb.apply_func(pd.to_datetime),
        cleaned=source["s"].str.strip(),
    )
    output = first.assign(
        year=first["parsed"].dt.year,
        up=first["cleaned"].str.upper(),
    )

    result, plan = _evaluate(
        output,
        OptConfig(dataframe_ops=True),
        backend,
    )
    maps = _maps(plan)

    assert len(maps) == 1
    assert maps[0].batches == (
        {
            "parsed": DatetimeExpr(Col("ts")),
            "cleaned": StrExpr(Col("s"), "strip"),
        },
        {
            "year": DtExpr(Col("parsed"), "year"),
            "up": StrExpr(Col("cleaned"), "upper"),
        },
    )
    assert list(result["year"]) == [2021, 2023]
    assert list(result["up"]) == ["AB", "C"]
