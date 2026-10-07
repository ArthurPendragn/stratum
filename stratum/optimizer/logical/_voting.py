"""Expand a ``VotingClassifier`` into member predictors plus a vote.

``VotingClassifier`` is one estimator object. Left that way, the members are
invisible to common-subexpression elimination, to implementation selection, and
to the scheduler. This rewrite replaces that one predictor with one predictor
per live member and a :class:`VotingOp` that combines their outputs.

The combination matches scikit-learn where that call succeeds. Members are
fitted on encoded labels: ``VotingClassifier.fit`` runs ``LabelEncoder`` and
trains every member on ``0 .. k-1``, and the expansion does the same. An
:class:`EncodedLabelsOp` replaces ``y`` for the members and records
``classes_`` in ``numpy.unique`` order (the order ``LabelEncoder`` uses).
The vote reads that ``classes_`` and maps the winning index back through it.
A setting keyed by the label value, such as a ``class_weight`` dict, therefore
sees the encoded classes.

Hard voting is a weighted majority over those indices (a tie is the class
that sorts first). Soft voting is a weighted average of the member
probability matrices, stacked by position as scikit-learn 1.8 does: a
member's ``classes_`` is not consulted, and members whose widths differ are an
error.

The training target is checked with ``type_of_target`` on the fitting pass,
before any member fits, as ``VotingClassifier.fit`` does. A continuous or
unknown target raises ``ValueError``. A multilabel or multi-output target
raises ``NotImplementedError``; flattening it would vote over one label per
cell instead of one label per row.

A pass for a method the vote does not serve returns labels, as an unexpanded
voter does. That covers ``predict_proba`` on a hard vote and
``predict_log_proba`` on either: scikit-learn 1.8's ``VotingClassifier``
implements neither.

The rewrite is logical, and it runs before physical lowering. Lowering
replaces one node by one node and then copies the old inputs onto the
replacement, which would hang the vote directly off ``X`` and orphan the
members. Exact type only: a subclass of ``VotingClassifier`` stays a
passthrough, the same rule the predictor family table uses.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._ops import FITTING_MODE, Op, PredictorOp

import logging
logger = logging.getLogger(__name__)


# The methods ``VotingClassifier`` defines for each voting mode. Any other
# pass falls back to labels, as it does for an unexpanded voter.
_SOFT_RESPONSES = frozenset({"predict", "predict_proba"})
_HARD_RESPONSES = frozenset({"predict"})


def _to_numpy(values) -> np.ndarray:
    """Array of a column, whatever frame library produced it."""
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        return to_numpy()
    return np.asarray(values)


def _as_labels(values) -> np.ndarray:
    """Member predictions as a flat array."""
    return np.ravel(_to_numpy(values))


def _as_single_target(values) -> np.ndarray:
    """One label per row, after the target check ``VotingClassifier.fit`` runs.

    The check sees the target before it is flattened. A single column (shape
    ``(n, 1)``) is one target; a wider one is multi-output, and flattening it
    first would hide that and vote over one label per cell.
    """
    matrix = _to_numpy(values)
    y_type = type_of_target(matrix, input_name="y")
    if y_type in ("unknown", "continuous"):
        raise ValueError(
            f"Unknown label type: {y_type}. Maybe you are trying to fit a "
            "classifier, which expects discrete classes on a "
            "regression target with continuous values."
        )
    if y_type not in ("binary", "multiclass"):
        raise NotImplementedError(
            "VotingClassifier only supports binary or multiclass "
            "classification. Multilabel and multi-output classification are not "
            "supported."
        )
    return np.ravel(matrix)


def average_probas(member_probas, weights) -> np.ndarray:
    """Weighted mean of member probability matrices, column by column.

    ``VotingClassifier.predict_proba`` stacks the matrices as they are, so
    members of different widths raise rather than being aligned.
    """
    stacked = np.stack([np.asarray(proba, dtype=float) for proba in member_probas], axis=0)
    return np.average(stacked, axis=0, weights=weights)


def _unknown_label(label) -> ValueError:
    return ValueError(
        f"member predict returned {label!r}, which is not an index into the training labels"
    )


def _class_indices(labels, n_classes: int) -> np.ndarray:
    """Encoded class index of each predicted label.

    Members are fitted on ``0 .. k-1``, so a prediction is already a column
    of the training ``classes_``. A value outside that range was not in the
    training set.
    """
    labels = _as_labels(labels)
    if labels.size == 0:
        return np.empty(0, dtype=np.intp)
    if not np.issubdtype(labels.dtype, np.integer):
        raise _unknown_label(labels.flat[0])
    encoded = labels.astype(np.intp, copy=False)
    out_of_range = (encoded < 0) | (encoded >= n_classes)
    if out_of_range.any():
        raise _unknown_label(encoded[out_of_range][0])
    return encoded


def majority_labels(member_labels, weights, classes) -> np.ndarray:
    """Weighted majority, returned as the original training labels.

    ``member_labels`` are encoded indices, one class per row per member.
    Ties resolve to the smallest index, the class that sorts first. No rows
    in yields no labels out.
    """
    classes = np.asarray(classes)
    n_classes = len(classes)
    columns = [_class_indices(labels, n_classes) for labels in member_labels]
    encoded = np.column_stack(columns) if columns else np.empty((0, 0), dtype=np.intp)
    n_samples, n_members = encoded.shape
    counts = np.zeros((n_samples, n_classes), dtype=float)
    if n_samples and n_members:
        sample_idx = np.arange(n_samples)
        if weights is None:
            weight_vector = np.ones(n_members, dtype=float)
        else:
            weight_vector = np.asarray(weights, dtype=float)
        # Each row takes one class from this member, so each (row, class)
        # pair is written once.
        for member in range(n_members):
            counts[sample_idx, encoded[:, member]] += weight_vector[member]
    return classes[np.argmax(counts, axis=1)]


class EncodedLabelsOp(Op):
    """Training labels as ``0 .. k-1``, in ``numpy.unique`` order.

    ``VotingClassifier`` fits every member on ``LabelEncoder`` output and maps
    the vote back through ``classes_``. Members of an expanded voter read this
    op instead of the original ``y``, so a ``class_weight`` dict and any other
    label-keyed setting sees the same integers scikit-learn would pass. The
    fitting pass keeps ``classes_`` for the vote, which does not scan ``y``
    again.

    A concrete ``y`` is stored on the op. A graph-fed ``y`` is the sole input.
    Only the fitting pass encodes. Members do not read ``y`` on a response
    pass, and this op returns an empty array so the node still has a value
    to pin. A target ``VotingClassifier.fit`` refuses is rejected here, before
    the members fit.
    """

    logical_family = "EncodeLabels"
    fields = ["y"]

    def __init__(self, y=None, inputs=None):
        super().__init__(name="LabelEncoder", inputs=list(inputs) if inputs else [])
        self.y = y
        self.classes_ = None

    def process(self, mode: str, inputs: list):
        if mode != FITTING_MODE:
            return np.empty(0, dtype=np.intp)
        values = self.y if self.y is not None else inputs[0]
        self.classes_, encoded = np.unique(_as_single_target(values), return_inverse=True)
        return encoded


class VotingOp(Op):
    """Combine member predictor outputs into one hard or soft vote.

    Inputs are the member predictors, in estimator order, and then the
    :class:`EncodedLabelsOp` they were fitted against. ``weights`` is ``None``
    for a uniform vote, or one weight per live member. Dropped members are
    already gone, and so are their weights.

    The fitting pass copies ``classes_`` from that encoder, in ``numpy.unique``
    order, and emits the vote of the members' training outputs. Those outputs
    are encoded: probability columns line up with ``classes_`` by position, and
    a hard prediction is an index into it. A response pass only combines. A
    pass for a method the vote does not serve (``predict_proba`` on a hard
    vote, ``predict_log_proba``, ``decision_function``) returns labels, as an
    unexpanded voter does. ``transform`` is not implemented: the predictor
    path never calls it.

    ``n_jobs`` on the original voter is not preserved. Members are separate
    nodes, so the scheduler decides when they run.
    """

    logical_family = "Vote"
    fields = ["voting", "weights"]
    #: Fit-pass planning must run this op. ``classes_`` is copied from the
    #: encoder here, and a response pass does not see the training labels again.
    records_training_labels = True

    def __init__(self, voting, weights=None, inputs=None):
        super().__init__(name=str(voting), inputs=list(inputs) if inputs else [])
        if voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got {voting!r}")
        self.voting = voting
        self.weights = None if weights is None else tuple(weights)
        self.classes_ = None
        self.supported_modes = _SOFT_RESPONSES if voting == "soft" else _HARD_RESPONSES

    @property
    def n_members(self) -> int:
        # The last input is the encoder, not a member.
        return len(self.inputs) - 1

    def consumes_inputs_positionally(self) -> bool:
        # Each member is one column of the vote, so two slots that happen to
        # be the same predictor must stay distinct. The encoder occupies the
        # last slot and must not be collapsed into a member either.
        return True

    def _member_values(self, inputs) -> list:
        # The last value is the encoder's output. ``classes_`` is read from
        # the encoder op; this array is not a member prediction.
        return list(inputs[:-1])

    def _probabilities(self, inputs) -> np.ndarray:
        return average_probas(self._member_values(inputs), self.weights)

    def _labels(self, inputs) -> np.ndarray:
        if self.voting == "soft":
            averaged = self._probabilities(inputs)
            return self.classes_[np.argmax(averaged, axis=1)]
        return majority_labels(self._member_values(inputs), self.weights, self.classes_)

    def process(self, mode: str, inputs: list):
        if mode == FITTING_MODE:
            # The encoder ran first: it is an input, and it already checked the
            # target and stored ``classes_``. Copying it here is what a later
            # response pass maps indices through.
            classes = self.inputs[-1].classes_
            if classes is None:
                raise RuntimeError(
                    "VotingOp was asked to fit before its label encoder ran"
                )
            self.classes_ = classes
            # The fitting pass emits the same kind of value a predictor does
            # after fit: labels, not the probabilities the members produced.
            return self._labels(inputs)
        if self.classes_ is None:
            raise RuntimeError(
                "VotingOp was asked for a response before it saw training labels"
            )
        if mode == "predict_proba" and mode in self.supported_modes:
            return self._probabilities(inputs)
        # ``predict``, and any pass whose method this vote does not serve,
        # returns labels, as a predictor without that method does.
        return self._labels(inputs)


def _blocks_expansion(member) -> bool:
    """A member this rewrite cannot prove equivalent to a plain predictor."""
    if isinstance(member, (Pipeline, VotingClassifier, VotingRegressor)):
        return True
    return not isinstance(member, BaseEstimator)


def _live_members(estimator):
    """``(members, weights)`` with ``"drop"`` slots removed, or ``None`` to skip.

    ``weights`` is ``None`` when the voter asked for a uniform vote. A vector
    of the wrong length, or a weight ``float`` rejects, is left for
    scikit-learn. The rewrite does not fire.
    """
    estimators = list(estimator.estimators)
    raw_weights = estimator.weights
    if raw_weights is not None:
        raw_weights = list(raw_weights)
        if len(raw_weights) != len(estimators):
            return None
    members = []
    weights = []
    for index, pair in enumerate(estimators):
        if not isinstance(pair, tuple) or len(pair) != 2:
            return None
        _name, member = pair
        if member == "drop":
            continue
        if _blocks_expansion(member):
            return None
        members.append(member)
        if raw_weights is not None:
            weights.append(raw_weights[index])
    if not members:
        return None
    if raw_weights is None:
        return members, None
    # ``None`` and ``"x"`` are legal constructor values. Casting them here
    # would fail compilation; the unexpanded voter lets scikit-learn decide.
    try:
        return members, tuple(float(w) for w in weights)
    except (TypeError, ValueError):
        return None


def _fit_group_is_sample_weight_only(fit_group) -> bool:
    """True when the fit group is absent or names only ``sample_weight``."""
    if fit_group is None:
        return True
    if not isinstance(fit_group, dict):
        return False
    return set(fit_group).issubset({"sample_weight"})


def match_voting_classifier(op: Op) -> tuple[VotingOp, list[PredictorOp]] | None:
    """The vote that replaces ``op``, or ``None`` when ``op`` must stay as it is.

    The returned nodes are not wired into the graph. :func:`expand_voting_classifiers`
    does that.
    """
    if not isinstance(op, PredictorOp):
        return None
    estimator = op.original_estimator
    if type(estimator) is not VotingClassifier:
        return None
    # A graph-fed parameter (a member, a weight vector, ``voting`` itself) is
    # not a value this rewrite can copy onto a child op.
    if op.param_refs:
        return None
    # Anything but the fit group belongs to the voter call. Dropping it would
    # change results, so the voter stays a passthrough.
    if any(key != "fit" for key in op.kwargs):
        return None
    # ``VotingClassifier.fit`` accepts ``sample_weight`` and rejects every
    # other keyword when metadata routing is off. A group that names anything
    # else, or whose keys are not known yet, stays on the voter so that
    # rejection still happens there.
    fit_group = op.kwargs.get("fit")
    if not _fit_group_is_sample_weight_only(fit_group):
        return None
    if estimator.voting not in ("hard", "soft"):
        return None
    live = _live_members(estimator)
    if live is None:
        return None
    members, weights = live
    override = "predict_proba" if estimator.voting == "soft" else "predict"
    # ``y`` is still the original target. ``_splice`` points each member at an
    # encoded copy so ``fit`` sees ``0 .. k-1``, and gives the vote that encoder.
    children = [
        PredictorOp(
            estimator=member,
            y=op.y,
            cols=op.cols,
            exclude_cols=op.exclude_cols,
            no_wrap=op.no_wrap,
            allow_reject=op.allow_reject,
            unsupervised=op.unsupervised,
            kwargs={} if fit_group is None else {"fit": fit_group},
            response_override=override,
        )
        for member in members
    ]
    voter = VotingOp(voting=estimator.voting, weights=weights)
    return voter, children


def _encoded_y(old: PredictorOp) -> EncodedLabelsOp:
    """The node members fit against: original ``y``, encoded as ``0 .. k-1``."""
    if isinstance(old.y, OperandRef):
        encoder = EncodedLabelsOp()
        y_op = old.inputs[old.y.k]
        encoder.add_input(y_op)
        y_op.add_output(encoder)
        return encoder
    return EncodedLabelsOp(y=old.y)


def _point_member_at_encoded_y(old: PredictorOp, child: PredictorOp, encoder: EncodedLabelsOp) -> None:
    """Give ``child`` ``old``'s inputs, with the ``y`` slot replaced by ``encoder``."""
    if isinstance(old.y, OperandRef):
        child.inputs = list(old.inputs)
        child.inputs[old.y.k] = encoder
    else:
        child.inputs = [*old.inputs, encoder]
        child.y = OperandRef(len(old.inputs))
    for source in child.inputs:
        source.add_output(child)


def _splice(old: PredictorOp, voter: VotingOp, children: list[PredictorOp], root: Op) -> Op:
    """Replace ``old`` with ``voter``. Members and the vote read the encoder.

    The encoder is the vote's last input for a concrete ``y`` and a graph-fed
    one. The vote copies ``classes_`` from it, so the raw target is not also
    an input. Its buffer can be freed once the encoder has run, instead of
    staying live until the vote.
    """
    encoder = _encoded_y(old)
    for child in children:
        _point_member_at_encoded_y(old, child, encoder)
        child.add_output(voter)
    voter.inputs = [*children, encoder]
    encoder.add_output(voter)
    for consumer in list(old.outputs):
        consumer.replace_input(old, voter)
        voter.add_output(consumer)
    for source in old.inputs:
        source.outputs = [out for out in source.outputs if out is not old]
    old.inputs = []
    old.outputs = []
    return voter if root is old else root


def expand_voting_classifiers(root: Op) -> tuple[Op, bool]:
    """Replace every plain ``VotingClassifier`` predictor in ``root`` with a vote.

    The bool is whether any voter was expanded. A later CSE is only useful in
    that case: identical members are separate nodes now, and can be shared.
    """
    expanded = False
    for op in list(topological_iterator(root)):
        matched = match_voting_classifier(op)
        if matched is None:
            continue
        voter, children = matched
        logger.debug(
            "Expanding VotingClassifier(%s) into %d members",
            voter.voting, len(children),
        )
        root = _splice(op, voter, children, root)
        expanded = True
    return root, expanded
