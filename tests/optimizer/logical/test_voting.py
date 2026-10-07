"""VotingClassifier expands into member predictors plus a vote.

The expanded plan has to match ``VotingClassifier`` fit and predicted on the
same rows. A pass for a method the voter does not define returns labels, as it
does for an unexpanded voter. Configurations the rewrite cannot prove
equivalent stay on the passthrough path.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import stratum as st
from stratum.adapters.tree_classifier import supports_rust_histogram_random_forest_classifier
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, SearchConfig, optimize
from stratum.optimizer.physical import FlagBasedSelector
from stratum.optimizer.logical._base import All, OperandRef
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._ops import Op, PredictorOp
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.optimizer.logical._voting import (
    EncodedLabelsOp,
    VotingOp,
    expand_voting_classifiers,
    match_voting_classifier,
)
from stratum.optimizer.physical._predictor_execs import (
    DecisionTreeOp,
    PassthroughPredictor,
    SklearnRandomForestClassifier,
    StratumHistogramRandomForestClassifier,
)
from stratum.runtime._scheduler import SequentialScheduler


@pytest.fixture(autouse=True)
def _no_preview():
    with st.config_context(eager_data_ops=False):
        yield


def _frame(labels, n=120, seed=0):
    x, encoded = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_redundant=0,
        random_state=seed,
    )
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    df["y"] = np.asarray(labels)[encoded]
    return df


def _voter(voting, weights=None, *, drop=True, n_jobs=2):
    members = [
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("lr", LogisticRegression(max_iter=400)),
        ("rf", RandomForestClassifier(n_estimators=8, random_state=0)),
    ]
    if drop:
        members.append(("gone", "drop"))
        if weights is not None:
            weights = list(weights) + [5]
    return VotingClassifier(
        estimators=members, voting=voting, weights=weights, n_jobs=n_jobs,
    )


def _plan(df, estimator, **apply_kwargs):
    data = st.as_data_op(df)
    y = data["y"].skb.mark_as_y()
    x = data.drop(columns=["y"]).skb.mark_as_X()
    return x.skb.apply(estimator, y=y, **apply_kwargs)


def _response(plan, mode, seed=0, test_size=0.25, *, test_index=None, config=None):
    """Run the plan the way ``evaluate`` does, in ``mode`` instead of ``predict``.

    ``test_index`` replaces the holdout split. The remaining rows are the
    training fold, so an empty index fits on every row and predicts on none.
    ``config`` overrides the optimizer config for one run, which is how a
    polars frame keeps a polars backend through the vote.
    """
    linearized, split_pos, flagged = optimize(plan, config=config, env=get_data(plan))
    sched = SequentialScheduler(linearized, split_pos, flagged)
    split_op = sched.compute_xy()
    x_data = sched.pool.pin(split_op.inputs[0])
    if test_index is None:
        train_index, test_index = train_test_split(
            range(len(x_data)), test_size=test_size, random_state=seed,
        )
    else:
        test_index = list(test_index)
        held_out = set(test_index)
        train_index = [i for i in range(len(x_data)) if i not in held_out]
    split_op.indices = train_index
    sched.compute(sched.pos_split_op)
    split_op.indices = test_index
    rows = sched.compute(sched.pos_split_op, mode=mode)
    sched._finish()
    return rows[0]["vals"]


def _split(df, seed=0, test_size=0.25):
    train_index, test_index = train_test_split(
        range(len(df)), test_size=test_size, random_state=seed,
    )
    x = df.drop(columns=["y"])
    y = df["y"]
    return (
        x.iloc[list(train_index)],
        x.iloc[list(test_index)],
        y.iloc[list(train_index)],
        y.iloc[list(test_index)],
    )


def _compiled(plan):
    linearized, _, _ = optimize(plan, env=get_data(plan))
    return linearized


def _votes(plan):
    return [op for op in _compiled(plan) if isinstance(op, VotingOp)]


class _LabelSource(Op):
    """A graph node that yields one concrete target column."""

    fields = ["y"]

    def __init__(self, y):
        super().__init__(name="y")
        self.y = y

    def process(self, mode, inputs):
        return self.y


class _ConstantClass(ClassifierMixin, BaseEstimator):
    """Predict one class, by its index in ``numpy.unique(y)`` order."""

    def __init__(self, class_index=0):
        self.class_index = class_index

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[self.class_index])

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        proba[:, self.class_index] = 1.0
        return proba


class _DropLastClass(ClassifierMixin, BaseEstimator):
    """Fit without the class that sorts last, so ``predict_proba`` is narrower."""

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        x = np.asarray(X)
        classes = np.unique(y)
        mask = y != classes[-1]
        self.inner_ = LogisticRegression(max_iter=400).fit(x[mask], y[mask])
        self.classes_ = np.asarray(self.inner_.classes_)
        return self

    def predict(self, X):
        return self.inner_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.inner_.predict_proba(np.asarray(X))


def test_response_override_changes_the_post_fit_call_only():
    y = np.array([0, 1, 0, 1])
    x = np.ones((4, 1))
    overridden = PredictorOp(
        estimator=_ConstantClass(class_index=0), y=y, cols=All(), no_wrap=True,
        response_override="predict_proba",
    )
    plain = overridden.clone()
    plain.response_override = None

    assert overridden.effective_mode("fit_transform") == "fit_transform"
    assert overridden.effective_mode("predict") == "predict_proba"
    assert plain.effective_mode("predict") == "predict"
    assert overridden._response_after_fit() == "predict_proba"
    assert plain._response_after_fit() == "predict"

    proba = overridden.process("fit_transform", [x])
    labels = plain.process("fit_transform", [x])
    assert proba.shape == (4, 2)
    assert labels.shape == (4,)
    cloned = overridden.clone()
    assert cloned.response_override == "predict_proba"
    assert cloned is not overridden


@pytest.mark.parametrize("voting", ["hard", "soft"])
@pytest.mark.parametrize("labels", [
    np.arange(3),
    np.array(["b", "a", "c"]),
    np.array([-1, 2, 10]),
])
def test_expanded_vote_matches_sklearn(voting, labels):
    df = _frame(labels)
    weights = [2, 1, 1]
    estimator = _voter(voting, weights)
    plan = _plan(df, estimator)
    x_train, x_test, y_train, _y_test = _split(df)
    reference = clone(estimator).fit(x_train, y_train)

    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict")),
        np.asarray(reference.predict(x_test)),
    )
    votes = _votes(plan)
    assert len(votes) == 1
    assert votes[0].voting == voting
    assert votes[0].weights == (2.0, 1.0, 1.0)
    assert votes[0].n_members == 3
    encoder = votes[0].inputs[-1]
    assert isinstance(encoder, EncodedLabelsOp)
    assert all(member.inputs[member.y.k] is encoder for member in votes[0].inputs[:-1])
    # The raw target feeds the encoder only. The vote must not keep it alive.
    y_op = encoder.inputs[0]
    assert y_op not in votes[0].inputs
    assert y_op in encoder.remove_after
    assert y_op not in votes[0].remove_after

    # ``VotingClassifier`` has no ``predict_log_proba``, and a hard voter has no
    # ``predict_proba``. An unexpanded voter returns labels on those passes.
    expected_labels = np.asarray(reference.predict(x_test))
    assert not hasattr(reference, "predict_log_proba")
    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict_log_proba")), expected_labels,
    )
    if voting == "hard":
        assert not hasattr(reference, "predict_proba")
        np.testing.assert_array_equal(
            np.asarray(_response(plan, "predict_proba")), expected_labels,
        )
        return

    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict_proba")),
        np.asarray(reference.predict_proba(x_test)),
    )


@pytest.mark.parametrize("graph_fed", [False, True])
def test_vote_reads_classes_from_the_encoder(graph_fed):
    """Concrete and graph-fed targets both reach the vote through the encoder.

    The encoder is the vote's last input. A tie between the members resolves
    to the class that sorts first; treating the encoded labels as another
    member would hand the first row to ``"b"``.
    """
    labels = np.array(["b", "a", "c", "a"])
    estimator = VotingClassifier(
        estimators=[
            ("low", _ConstantClass(class_index=0)),
            ("high", _ConstantClass(class_index=1)),
        ],
        voting="hard",
    )
    if graph_fed:
        y_op = _LabelSource(labels)
        old = PredictorOp(estimator=estimator, y=OperandRef(0))
        old.add_input(y_op)
        y_op.add_output(old)
    else:
        y_op = None
        old = PredictorOp(estimator=estimator, y=labels)

    vote, expanded = expand_voting_classifiers(old)
    assert expanded
    assert isinstance(vote, VotingOp)
    encoder = vote.inputs[-1]
    assert isinstance(encoder, EncodedLabelsOp)
    assert vote.n_members == 2
    assert all(member.inputs[member.y.k] is encoder for member in vote.inputs[:-1])
    if graph_fed:
        assert encoder.inputs == [y_op]
        assert y_op not in vote.inputs
        assert y_op.outputs == [encoder]
        encoded = encoder.process("fit_transform", [labels])
    else:
        assert encoder.inputs == []
        assert encoder.y is labels
        encoded = encoder.process("fit_transform", [])

    np.testing.assert_array_equal(encoder.classes_, np.array(["a", "b", "c"]))
    np.testing.assert_array_equal(encoded, np.array([1, 0, 2, 0]))
    labels_out = vote.process("fit_transform", [
        np.zeros(4, dtype=int), np.ones(4, dtype=int), encoded,
    ])
    assert vote.classes_ is encoder.classes_
    np.testing.assert_array_equal(labels_out, np.array(["a", "a", "a", "a"]))
    # A response pass uses the copied class order and ignores the encoder output.
    predicted = vote.process("predict", [
        np.full(4, 2, dtype=int), np.full(4, 2, dtype=int), np.empty(0, dtype=np.intp),
    ])
    np.testing.assert_array_equal(predicted, np.array(["c", "c", "c", "c"]))


@pytest.mark.parametrize("labels,class_weight,accepts", [
    (np.array(["a", "b", "c"]), {"a": 50, "b": 1, "c": 1}, False),
    (np.array([-1, 2, 10]), {0: 50, 1: 1, 2: 1}, True),
])
def test_members_are_fit_on_encoded_labels(labels, class_weight, accepts):
    """Members see ``0 .. k-1``, as they do under ``VotingClassifier.fit``.

    A ``class_weight`` dict is keyed by whatever ``y`` the member's ``fit``
    receives. Original labels and encoded integers each succeed in one
    direction and fail in the other.
    """
    df = _frame(labels)
    estimator = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=400, class_weight=class_weight)),
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ],
        voting="soft",
    )
    plan = _plan(df, estimator)
    x_train, x_test, y_train, _y_test = _split(df)
    if not accepts:
        with pytest.raises(ValueError, match="class_weight"):
            clone(estimator).fit(x_train, y_train)
        with pytest.raises(RuntimeError, match="class_weight"):
            _response(plan, "predict")
        return

    reference = clone(estimator).fit(x_train, y_train)
    bare = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=400)),
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ],
        voting="soft",
    )
    unweighted = clone(bare).fit(x_train, y_train)
    actual = np.asarray(_response(plan, "predict_proba"))
    np.testing.assert_array_equal(actual, np.asarray(reference.predict_proba(x_test)))
    assert not np.allclose(actual, unweighted.predict_proba(x_test))


def test_multicolumn_target_is_rejected():
    """A two-column ``y`` is a multi-output target, which the voter refuses.

    Flattening it would emit one label per cell. ``VotingClassifier.fit``
    raises ``NotImplementedError`` before it fits the members.
    """
    df = _frame(np.arange(3), n=40)
    df["y2"] = 1 - df["y"]
    data = st.as_data_op(df)
    y = data[["y", "y2"]].skb.mark_as_y()
    x = data.drop(columns=["y", "y2"]).skb.mark_as_X()
    estimator = VotingClassifier(
        estimators=[
            ("a", DecisionTreeClassifier(max_depth=2, random_state=0)),
            ("b", DecisionTreeClassifier(max_depth=3, random_state=1)),
        ],
        voting="hard",
    )
    plan = x.skb.apply(estimator, y=y)
    with pytest.raises(RuntimeError, match="multi-output"):
        _response(plan, "predict")

    encoder = EncodedLabelsOp(y=np.zeros((4, 2), dtype=int))
    with pytest.raises(NotImplementedError, match="multi-output"):
        encoder.process("fit_transform", [])
    assert encoder.classes_ is None


class _RecordsFit(ClassifierMixin, BaseEstimator):
    """Count ``fit`` calls across clones, which share the class attribute."""

    fits = []

    def fit(self, X, y):
        type(self).fits.append(len(X))
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        proba[:, 0] = 1.0
        return proba


@pytest.mark.parametrize("labels,y_type", [
    (np.linspace(0.0, 1.0, 40), "continuous"),
    (np.array([0, 1, None, 2] * 10, dtype=object), "unknown"),
])
def test_non_classification_target_is_rejected_before_members_fit(labels, y_type):
    """A target ``type_of_target`` does not call binary or multiclass raises.

    ``VotingClassifier.fit`` raises before it fits anyone. The expanded plan
    has to raise the same error, from the encoder the members read, so no
    member fits either.
    """
    df = _frame(np.arange(3), n=40)
    df["y"] = pd.Series(labels, dtype=labels.dtype)
    estimator = VotingClassifier(
        estimators=[("a", _RecordsFit()), ("b", _RecordsFit())], voting="soft",
    )
    message = f"Unknown label type: {y_type}"
    with pytest.raises(ValueError, match=message):
        clone(estimator).fit(df.drop(columns=["y"]), df["y"])

    plan = _plan(df, estimator)
    assert len(_votes(plan)) == 1
    _RecordsFit.fits.clear()
    with pytest.raises(RuntimeError, match=message) as raised:
        _response(plan, "predict")
    assert "EncodeLabels" in str(raised.value)
    assert _RecordsFit.fits == []

    encoder = EncodedLabelsOp(y=labels)
    with pytest.raises(ValueError, match=message):
        encoder.process("fit_transform", [])
    assert encoder.classes_ is None


def test_polars_labels_match_sklearn():
    """A polars column is a legal ``y``. Encoding must not assume pandas."""
    labels = np.array(["b", "a", "c"])
    pdf = _frame(labels, n=80)
    frame = pl.from_pandas(pdf)
    data = st.as_data_op(frame)
    y = data["y"].skb.mark_as_y()
    # The polars drop op takes the column names positionally.
    x = data.drop("y").skb.mark_as_X()
    estimator = VotingClassifier(
        estimators=[
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("lr", LogisticRegression(max_iter=400)),
        ],
        voting="hard",
        weights=[2, 1],
    )
    plan = x.skb.apply(estimator, y=y)
    config = OptConfig(selector=FlagBasedSelector(backend="polars"))
    x_train, x_test, y_train, _y_test = _split(pdf)
    reference = clone(estimator).fit(x_train, y_train)
    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict", config=config)),
        np.asarray(reference.predict(x_test)),
    )

    column = pl.Series("y", ["b", "a", "b", "c"])
    encoder = EncodedLabelsOp(y=column)
    encoded = encoder.process("fit_transform", [])
    np.testing.assert_array_equal(encoded, np.array([1, 0, 1, 2]))
    np.testing.assert_array_equal(encoder.classes_, np.array(["a", "b", "c"]))
    wide = pl.DataFrame({"y": ["a", "b"], "z": ["b", "a"]})
    with pytest.raises(NotImplementedError, match="multi-output"):
        EncodedLabelsOp(y=wide).process("fit_transform", [])

    # The two members tie, so the class that sorts first wins. Counting the
    # encoder's output as a third member would break the tie on the first row.
    vote = VotingOp(voting="hard", inputs=[None, None, encoder])
    labels_out = vote.process("fit_transform", [
        np.zeros(4, dtype=int), np.ones(4, dtype=int), encoded,
    ])
    assert vote.classes_ is encoder.classes_
    np.testing.assert_array_equal(labels_out, np.array(["a", "a", "a", "a"]))


def test_hard_vote_probability_metric_scores_predictions():
    """A probability metric falls back to ``predict`` for a hard vote.

    The members can serve ``predict_proba``. The vote cannot, and it is the
    response the plan produces. Scoring must not ask the vote for
    probabilities.
    """
    df = _frame(np.array(["b", "a", "c"]), n=80)
    data = st.as_data_op(df)
    y = data["y"].skb.mark_as_y()
    x = data.drop(columns=["y"]).skb.mark_as_X()

    def _hard(seed):
        return VotingClassifier(
            estimators=[
                ("tree", DecisionTreeClassifier(max_depth=3, random_state=seed)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            voting="hard",
        )

    plan = x.skb.apply(st.choose_from({"a": _hard(0), "b": _hard(1)}, name="vote"), y=y)
    metric = resolve_scoring(
        make_scorer(accuracy_score, response_method=("predict_proba", "predict")),
    )
    compiled, split_pos, flagged = optimize(plan, env=get_data(plan), search=SearchConfig(metric))
    votes = [op for op in compiled if isinstance(op, VotingOp)]
    assert len(votes) == 2
    assert all("predict_proba" not in vote.supported_modes for vote in votes)
    members = [member for vote in votes for member in vote.inputs[:vote.n_members]]
    assert any("predict_proba" in member.supported_modes for member in members)
    sink = compiled[-1]
    assert isinstance(sink, ScoreCandidatesOp)
    assert sink.response_mode == "predict"

    sched = SequentialScheduler(compiled, split_pos, flagged)
    sched.grid_search(KFold(n_splits=2, shuffle=True, random_state=0))
    assert sched.results_.height == 2
    assert np.isfinite(sched.results_["scores"].to_numpy()).all()

    soft = x.skb.apply(
        VotingClassifier(
            estimators=[
                ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            voting="soft",
        ),
        y=y,
    )
    soft_compiled, _, _ = optimize(soft, env=get_data(soft), search=SearchConfig(metric))
    assert soft_compiled[-1].response_mode == "predict_proba"


def test_log_probability_metric_scores_predictions_for_a_soft_vote():
    """A soft vote does not advertise ``predict_log_proba``.

    ``VotingClassifier`` does not define it, so a metric that prefers it
    reads predictions from the unexpanded voter. The expanded vote has to
    pick the same response.
    """
    df = _frame(np.arange(3), n=80)
    members = [
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("lr", LogisticRegression(max_iter=200)),
    ]
    metric = resolve_scoring(
        make_scorer(accuracy_score, response_method=("predict_log_proba", "predict")),
    )
    expanded = _plan(df, VotingClassifier(estimators=members, voting="soft"))
    unexpanded = _plan(df, _NamedVote(estimators=members, voting="soft"))
    modes = []
    for plan in (expanded, unexpanded):
        compiled, _, _ = optimize(plan, env=get_data(plan), search=SearchConfig(metric))
        assert any(isinstance(op, VotingOp) for op in compiled) == (plan is expanded)
        modes.append(compiled[-1].response_mode)
    assert modes == ["predict", "predict"]


def test_tie_resolves_to_the_class_that_sorts_first():
    df = _frame(np.array(["b", "a", "c"]))
    estimators = [
        ("low", _ConstantClass(class_index=0)),
        ("high", _ConstantClass(class_index=2)),
    ]
    x_train, x_test, y_train, _y_test = _split(df)
    for voting in ("hard", "soft"):
        estimator = VotingClassifier(estimators=estimators, voting=voting, weights=[1, 1])
        plan = _plan(df, estimator)
        expected = clone(estimator).fit(x_train, y_train).predict(x_test)
        actual = np.asarray(_response(plan, "predict"))
        np.testing.assert_array_equal(actual, np.asarray(expected))
        assert set(np.unique(actual)) == {"a"}


class _ReversedClasses(ClassifierMixin, BaseEstimator):
    """``predict_proba`` columns follow the inner model; ``classes_`` is reversed.

    The matrix is still one column per training class. ``VotingClassifier``
    averages those columns by position and does not consult ``classes_``.
    """

    def fit(self, X, y):
        self.inner_ = LogisticRegression(max_iter=400).fit(np.asarray(X), np.asarray(y).ravel())
        self.classes_ = np.asarray(self.inner_.classes_)[::-1]
        return self

    def predict(self, X):
        return self.inner_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.inner_.predict_proba(np.asarray(X))


def test_same_width_classes_permutation_stays_positional():
    """A same-width ``classes_`` permutation is averaged by column position.

    The widths match, so the vote keeps the positional average, which is
    what ``VotingClassifier.predict_proba`` returns.
    """
    df = _frame(np.arange(3))
    estimator = VotingClassifier(
        estimators=[
            ("full", LogisticRegression(max_iter=400)),
            ("reversed", _ReversedClasses()),
        ],
        voting="soft",
        weights=[1, 3],
    )
    plan = _plan(df, estimator)
    x_train, x_test, y_train, _y_test = _split(df)
    classes = np.unique(np.asarray(y_train))
    reference = clone(estimator).fit(x_train, y_train)

    full = LogisticRegression(max_iter=400).fit(x_train, y_train)
    flipped = _ReversedClasses().fit(x_train, y_train)
    assert len(flipped.classes_) == len(classes)
    assert not np.array_equal(flipped.classes_, classes)

    full_proba = np.asarray(full.predict_proba(x_test))
    flipped_proba = np.asarray(flipped.predict_proba(x_test))
    positional = np.average(np.stack([full_proba, flipped_proba]), axis=0, weights=[1, 3])

    actual = np.asarray(_response(plan, "predict_proba"))
    np.testing.assert_array_equal(actual, np.asarray(reference.predict_proba(x_test)))
    np.testing.assert_array_equal(actual, positional)
    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict")),
        np.asarray(reference.predict(x_test)),
    )


def test_members_of_different_widths_raise_like_the_voter():
    """A member that never saw one class is not padded to the training labels.

    scikit-learn 1.8 stacks member probability matrices by position, so a
    narrower member next to a full one raises. The expanded vote raises too,
    and so does the unexpanded voter.
    """
    df = _frame(np.array(["b", "a", "c"]))
    members = [
        ("full", LogisticRegression(max_iter=400)),
        ("partial", _DropLastClass()),
    ]
    estimator = VotingClassifier(estimators=members, voting="soft", weights=[1, 3])
    x_train, x_test, y_train, _y_test = _split(df)
    with pytest.raises(ValueError):
        clone(estimator).fit(x_train, y_train).predict_proba(x_test)

    plan = _plan(df, estimator)
    assert len(_votes(plan)) == 1
    unexpanded = _plan(df, _NamedVote(estimators=members, voting="soft", weights=[1, 3]))
    assert not _votes(unexpanded)
    for compiled in (plan, unexpanded):
        for mode in ("predict_proba", "predict"):
            with pytest.raises(RuntimeError, match="shape"):
                _response(compiled, mode)


def test_members_that_are_all_narrower_match_sklearn():
    """Equal widths are averaged by position, even below the training class count.

    Every member drops the same class, so ``VotingClassifier.predict_proba``
    returns one column fewer than there are training labels, and ``predict``
    maps the winning column back through ``classes_``.
    """
    df = _frame(np.array(["b", "a", "c"]))
    estimator = VotingClassifier(
        estimators=[("p1", _DropLastClass()), ("p2", _DropLastClass())],
        voting="soft",
        weights=[1, 3],
    )
    plan = _plan(df, estimator)
    x_train, x_test, y_train, _y_test = _split(df)
    reference = clone(estimator).fit(x_train, y_train)
    expected = np.asarray(reference.predict_proba(x_test))
    assert expected.shape == (len(x_test), len(np.unique(y_train)) - 1)

    np.testing.assert_array_equal(np.asarray(_response(plan, "predict_proba")), expected)
    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict")),
        np.asarray(reference.predict(x_test)),
    )


def test_sample_weight_is_forwarded_to_every_member():
    df = _frame(np.arange(3))
    # Large enough that ignoring the weights changes the probabilities.
    df["w"] = np.where(df["x0"] > 0, 50.0, 0.01)
    estimator = _voter("soft", [2, 1, 1], drop=False, n_jobs=1)
    data = st.as_data_op(df)
    y = data["y"].skb.mark_as_y()
    marked = data.drop(columns=["y"]).skb.mark_as_X()
    features = marked.drop(columns=["w"])
    plan = features.skb.apply(
        estimator, y=y, fit_kwargs={"sample_weight": marked["w"]},
    )

    x_train, x_test, y_train, _y_test = _split(df)
    features_train = x_train.drop(columns=["w"])
    features_test = x_test.drop(columns=["w"])
    weighted = clone(estimator).fit(
        features_train, y_train, sample_weight=np.asarray(x_train["w"]),
    )
    unweighted = clone(estimator).fit(features_train, y_train)
    actual = np.asarray(_response(plan, "predict_proba"))
    np.testing.assert_array_equal(actual, weighted.predict_proba(features_test))
    assert not np.allclose(actual, unweighted.predict_proba(features_test))


def test_identical_members_of_two_candidates_are_one_node():
    df = _frame(np.arange(3))
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    shared = ("tree", tree)
    first = VotingClassifier(
        estimators=[shared, ("lr", LogisticRegression(C=1.0, max_iter=200))],
        voting="soft",
    )
    second = VotingClassifier(
        estimators=[shared, ("lr", LogisticRegression(C=0.1, max_iter=200))],
        voting="soft",
    )
    data = st.as_data_op(df)
    y = data["y"].skb.mark_as_y()
    x = data.drop(columns=["y"]).skb.mark_as_X()
    plan = x.skb.apply(st.choose_from({"a": first, "b": second}, name="vote"), y=y)

    compiled = _compiled(plan)
    trees = [op for op in compiled if isinstance(op, DecisionTreeOp)]
    votes = [op for op in compiled if isinstance(op, VotingOp)]
    assert len(votes) == 2
    assert len(trees) == 1
    assert all(trees[0] in vote.inputs for vote in votes)


def test_forest_member_uses_the_histogram_kernel_when_it_is_supported():
    df = _frame(np.arange(3))
    forest = RandomForestClassifier(n_estimators=8, random_state=0)
    estimator = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=200)),
            ("rf", forest),
        ],
        voting="soft",
    )
    plan = _plan(df, estimator)
    supported, _reason = supports_rust_histogram_random_forest_classifier(forest)

    with st.config(implementation_selector="greedy"):
        vote = _votes(plan)[0]
    forests = [
        op for op in vote.inputs[:vote.n_members]
        if isinstance(op, (StratumHistogramRandomForestClassifier, SklearnRandomForestClassifier))
    ]
    assert len(forests) == 1
    if supported:
        assert type(forests[0]) is StratumHistogramRandomForestClassifier
    else:
        assert type(forests[0]) is SklearnRandomForestClassifier

    # Equality stays on the reference selector. The histogram kernel is not
    # required to be bit-exact, and that gap must not loosen the vote.
    # ``_response`` compiles again, so the selector is pinned here: a
    # process-wide greedy selector would compare the histogram kernel to
    # scikit-learn with ``assert_array_equal``.
    x_train, x_test, y_train, _y_test = _split(df)
    reference = clone(estimator).fit(x_train, y_train)
    with st.config(implementation_selector="default"):
        np.testing.assert_array_equal(
            np.asarray(_response(plan, "predict")),
            np.asarray(reference.predict(x_test)),
        )


@pytest.mark.parametrize("estimator", [
    VotingRegressor(estimators=[("a", Ridge()), ("b", Ridge(alpha=2.0))]),
    VotingClassifier(
        estimators=[
            ("pipe", make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))),
            ("lr", LogisticRegression(max_iter=200)),
        ],
        voting="soft",
    ),
])
def test_skipped_estimators_stay_passthrough_and_match(estimator):
    if isinstance(estimator, VotingRegressor):
        x, y = make_regression(n_samples=80, n_features=4, random_state=0)
        df = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
        df["y"] = y
    else:
        df = _frame(np.arange(3), n=80)
    plan = _plan(df, estimator)
    compiled = _compiled(plan)
    assert not any(isinstance(op, VotingOp) for op in compiled)
    passthroughs = [op for op in compiled if type(op) is PassthroughPredictor]
    assert len(passthroughs) == 1
    assert type(passthroughs[0].original_estimator) is type(estimator)

    x_train, x_test, y_train, _y_test = _split(df)
    reference = clone(estimator).fit(x_train, y_train)
    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict")),
        np.asarray(reference.predict(x_test)),
    )


def test_non_numeric_weights_stay_passthrough_and_match():
    """A weight ``float`` rejects stays on the voter.

    ``weights=[None, 1]`` fits and predicts under a hard ``VotingClassifier``.
    Casting it during compilation would raise ``TypeError`` instead.
    """
    df = _frame(np.arange(3), n=80)
    estimator = VotingClassifier(
        estimators=[
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("lr", LogisticRegression(max_iter=400)),
        ],
        voting="hard",
        weights=[None, 1],
    )
    plan = _plan(df, estimator)
    compiled = _compiled(plan)
    assert not any(isinstance(op, VotingOp) for op in compiled)
    passthroughs = [op for op in compiled if type(op) is PassthroughPredictor]
    assert len(passthroughs) == 1
    assert type(passthroughs[0].original_estimator) is VotingClassifier

    x_train, x_test, y_train, _y_test = _split(df)
    reference = clone(estimator).fit(x_train, y_train)
    np.testing.assert_array_equal(
        np.asarray(_response(plan, "predict")),
        np.asarray(reference.predict(x_test)),
    )


def test_subclass_bad_weights_and_unknown_voting_are_not_rewritten():
    df = _frame(np.arange(3), n=40)
    data = st.as_data_op(df)
    y = data["y"].skb.mark_as_y()
    x = data.drop(columns=["y"]).skb.mark_as_X()

    subclass = _NamedVote(estimators=[("lr", LogisticRegression())], voting="soft")
    mismatched = VotingClassifier(
        estimators=[("lr", LogisticRegression())], voting="soft", weights=[1, 2],
    )
    unknown = VotingClassifier(
        estimators=[("lr", LogisticRegression())], voting="ranked",
    )
    for estimator in (subclass, mismatched, unknown):
        assert match_voting_classifier(PredictorOp(estimator=estimator, y=y)) is None
        compiled = _compiled(x.skb.apply(estimator, y=y))
        assert not any(isinstance(op, VotingOp) for op in compiled)

    with pytest.raises(Exception):
        _response(_plan(df, mismatched), "predict")


def test_predict_kwargs_and_graph_fed_params_skip_the_rewrite():
    voter = _voter("soft", [1, 1, 1], drop=False)
    with_predict_kwargs = PredictorOp(estimator=voter, kwargs={"predict": {"foo": 1}})
    assert match_voting_classifier(with_predict_kwargs) is None

    graph_fed = PredictorOp(
        estimator=voter, param_refs={"weights": OperandRef(1)},
    )
    assert match_voting_classifier(graph_fed) is None

    extra_fit_key = PredictorOp(
        estimator=voter, kwargs={"fit": {"sample_weight": np.ones(4), "bonus": 1}},
    )
    assert match_voting_classifier(extra_fit_key) is None
    # The whole group is a graph value, so its keys are not known here.
    graph_fed_group = PredictorOp(estimator=voter, kwargs={"fit": OperandRef(1)})
    assert match_voting_classifier(graph_fed_group) is None
    sample_weight_only = PredictorOp(
        estimator=voter, kwargs={"fit": {"sample_weight": OperandRef(1)}},
    )
    assert match_voting_classifier(sample_weight_only) is not None


class _AcceptsBonus(ClassifierMixin, BaseEstimator):
    """A member ``fit`` that would accept a keyword the voter itself rejects."""

    def fit(self, X, y, bonus=0):
        self.bonus_ = bonus
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        if len(X):
            proba[:, 0] = 1.0
        return proba


def test_extra_fit_key_is_rejected_by_the_voter():
    """A fit keyword other than ``sample_weight`` stays on the passthrough.

    With metadata routing off, ``VotingClassifier.fit`` rejects that keyword.
    Expanding would splat it into the member, which accepts it.
    """
    df = _frame(np.arange(3), n=40)
    estimator = VotingClassifier(
        estimators=[
            ("extra", _AcceptsBonus()),
            ("lr", LogisticRegression(max_iter=200)),
        ],
        voting="soft",
    )
    data = st.as_data_op(df)
    y = data["y"].skb.mark_as_y()
    x = data.drop(columns=["y"]).skb.mark_as_X()
    plan = x.skb.apply(estimator, y=y, fit_kwargs={"bonus": 1})
    compiled = _compiled(plan)
    assert not any(isinstance(op, VotingOp) for op in compiled)
    with pytest.raises(RuntimeError):
        _response(plan, "predict")


class _NamedVote(VotingClassifier):
    """Exact-type match fails, so this voter stays a passthrough."""


def test_skipped_voter_returns_labels_when_the_method_is_missing():
    """An unexpanded voter predicts when the requested method does not exist.

    The subclass has neither ``predict_proba`` on a hard vote nor
    ``predict_log_proba`` on a soft one, so both passes return labels. The
    expanded vote does the same; ``test_expanded_vote_matches_sklearn``
    checks that side.
    """
    df = _frame(np.arange(3), n=80)
    members = [
        ("lr", LogisticRegression(max_iter=200)),
        ("tree", DecisionTreeClassifier(max_depth=2, random_state=0)),
    ]
    x_train, x_test, y_train, _y_test = _split(df)

    hard = _NamedVote(estimators=members, voting="hard")
    hard_plan = _plan(df, hard)
    assert not any(isinstance(op, VotingOp) for op in _compiled(hard_plan))
    hard_reference = clone(hard).fit(x_train, y_train)
    hard_labels = np.asarray(_response(hard_plan, "predict_proba"))
    np.testing.assert_array_equal(hard_labels, np.asarray(hard_reference.predict(x_test)))
    assert hard_labels.shape == (len(x_test),)

    soft = _NamedVote(estimators=members, voting="soft")
    soft_plan = _plan(df, soft)
    assert not any(isinstance(op, VotingOp) for op in _compiled(soft_plan))
    soft_reference = clone(soft).fit(x_train, y_train)
    soft_labels = np.asarray(_response(soft_plan, "predict_log_proba"))
    np.testing.assert_array_equal(soft_labels, np.asarray(soft_reference.predict(x_test)))
    assert soft_labels.shape == (len(x_test),)


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_zero_row_vote(voting):
    """A vote of empty member outputs has a row axis of length zero.

    sklearn's hard vote rejects that input inside ``apply_along_axis``. The
    expanded vote still returns one label per input row, which is none.
    A soft vote matches ``VotingClassifier`` on the empty frame.
    """
    df = _frame(np.arange(3), n=40)
    estimator = VotingClassifier(
        estimators=[
            ("low", _ConstantClass(class_index=0)),
            ("high", _ConstantClass(class_index=2)),
        ],
        voting=voting,
        weights=[1, 1],
    )
    plan = _plan(df, estimator)
    labels = np.asarray(_response(plan, "predict", test_index=[]))
    assert labels.shape == (0,)
    if voting == "hard":
        assert labels.dtype == np.unique(np.asarray(df["y"])).dtype
        return

    x = df.drop(columns=["y"])
    reference = clone(estimator).fit(x, df["y"])
    empty = x.iloc[:0]
    np.testing.assert_array_equal(labels, np.asarray(reference.predict(empty)))
    proba = np.asarray(_response(plan, "predict_proba", test_index=[]))
    np.testing.assert_array_equal(proba, np.asarray(reference.predict_proba(empty)))
    assert proba.shape == (0, 3)
