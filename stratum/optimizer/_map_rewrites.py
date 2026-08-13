from __future__ import annotations

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.ir._column_expr import ColumnExpr
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._ops import Op


def _next_assign_map(
    current: AssignMapOp,
    canonical_source: Op,
) -> AssignMapOp | None:
    """Return the next safe logical map in a chain, if one exists."""
    if len(current.outputs) != 1:
        return None

    candidate = current.outputs[0]
    if (
        type(candidate) is not AssignMapOp
        or not candidate.inputs
        or candidate.inputs[0] is not current
    ):
        return None

    # Input zero remains stage-relative after fusion. An auxiliary edge from
    # the chain's original source cannot share fused input zero without changing
    # that auxiliary reference to mean the staged frame instead.
    if any(
        producer is canonical_source
        for producer in candidate.inputs[1:]
    ):
        return None

    return candidate


def _collect_assign_map_chain(head: Op) -> tuple[AssignMapOp, ...]:
    """Collect the maximal safe logical assign-map chain starting at ``head``."""
    if type(head) is not AssignMapOp or not head.inputs:
        return ()

    canonical_source = head.inputs[0]
    chain = [head]
    current = head

    while (
        candidate := _next_assign_map(current, canonical_source)
    ) is not None:
        chain.append(candidate)
        current = candidate

    return tuple(chain)


def match_assign_map_chain(op: Op) -> tuple[AssignMapOp, AssignMapOp] | None:
    """Match the first two nodes of a safe logical assign-map chain."""
    if type(op) is not AssignMapOp or not op.inputs:
        return None
    candidate = _next_assign_map(op, op.inputs[0])
    if candidate is None:
        return None
    return op, candidate


def _build_fused_assign_map(
    chain: tuple[AssignMapOp, ...],
) -> AssignMapOp:
    """Build a fused map for ``chain`` without modifying the graph."""
    source = chain[0].inputs[0]
    inputs = [source]
    input_indices = {id(source): 0}
    batches: list[dict[str, ColumnExpr]] = []

    def add_input(producer: Op) -> int:
        index = input_indices.get(id(producer))
        if index is None:
            index = len(inputs)
            inputs.append(producer)
            input_indices[id(producer)] = index
        return index

    for map_op in chain:
        refs = {0: 0}
        for old_index, producer in enumerate(map_op.inputs[1:], start=1):
            refs[old_index] = add_input(producer)

        for batch in map_op.batches:
            batches.append({
                name: expr.remap_operand_refs(refs)
                for name, expr in batch.items()
            })

    fused = AssignMapOp(batches=tuple(batches), inputs=inputs)
    fused.is_X = any(op.is_X for op in chain)
    fused.is_y = any(op.is_y for op in chain)
    fused.is_split_op = any(op.is_split_op for op in chain)
    return fused


def _replace_assign_map_chain(
    root: Op,
    chain: tuple[AssignMapOp, ...],
    fused: AssignMapOp,
) -> Op:
    """Replace ``chain`` with ``fused`` and detach every old chain node."""
    chain_set = set(chain)
    tail = chain[-1]

    # One auxiliary producer can feed several maps in the chain. Remove every
    # old chain edge and install its single de-duplicated edge to the fused map.
    for producer in fused.inputs:
        producer.outputs = [
            output
            for output in producer.outputs
            if output not in chain_set
        ]
        producer.add_output(fused)

    # Earlier chain members have no external consumers by construction, so all
    # downstream edges that need moving belong to the tail.
    for downstream in list(tail.outputs):
        downstream.replace_input(tail, fused)
        fused.add_output(downstream)

    for map_op in chain:
        map_op.inputs = []
        map_op.outputs = []

    return fused if root is tail else root


def fuse_assign_maps(root: Op) -> Op:
    """Fuse every maximal safe chain of logical :class:`AssignMapOp` nodes."""
    consumed: set[AssignMapOp] = set()

    for candidate in list(topological_iterator(root)):
        if candidate in consumed:
            continue

        chain = _collect_assign_map_chain(candidate)
        if len(chain) < 2:
            continue

        fused = _build_fused_assign_map(chain)
        root = _replace_assign_map_chain(root, chain, fused)
        consumed.update(chain)

    return root
