import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
import pytest
import sklearn
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from stratum import _rust_backend as rb
from stratum.adapters.tree_classifier import (
    RustDecisionTreeClassifier,
    supports_rust_tree_classifier,
)


pytestmark = pytest.mark.skipif(
    not rb.HAVE_RUST, reason="Rust backend not built"
)


TREE_CLASSIFIER_CONTRACT = {
    "sklearn_version": "1.8.0",
    "implementation": "dt_exact_finite",
    "supported": {
        "criterion": ["gini"],
        "splitter": ["best"],
        "input": "dense numeric float32-normalizable finite matrix",
        "target": "single-output classification",
        "growth": "depth-first",
        "max_leaf_nodes": [None],
        "sample_weight": False,
        "class_weight": False,
        "missing_values": False,
    },
    "deferred_tasks": {
        "missing_values": 6,
        "best_first_growth": 7,
        "forest_bootstrap": 8,
    },
}


# Pin the supported sklearn contract so the Rust implementation stays aligned.
def test_semantics_contract_pins_sklearn_version_and_finite_scope():
    assert sklearn.__version__ == TREE_CLASSIFIER_CONTRACT["sklearn_version"]
    assert TREE_CLASSIFIER_CONTRACT["supported"]["missing_values"] is False
    assert TREE_CLASSIFIER_CONTRACT["deferred_tasks"]["missing_values"] == 6


# Validate direct native handles, prediction output, and reference lifetime.
def test_hand_built_model_prediction_normalization_lifetime_and_concurrency():
    handle = rb.tree_model_from_arrays(
        np.array([1, -1, -1], dtype=np.int64),
        np.array([2, -1, -1], dtype=np.int64),
        np.array([0, -2, -2], dtype=np.int64),
        np.array([0.5, -2.0, -2.0], dtype=np.float64),
        np.array([True, False, False], dtype=np.bool_),
        np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.7, 0.2], [0.8, 0.1, 0.1]],
            dtype=np.float64,
        ),
        1,
    )
    alias = handle
    del handle
    gc.collect()
    base = np.array([[99.0, 0.0], [99.0, 1.0], [99.0, np.nan]], dtype=np.float32)
    X = np.ascontiguousarray(base[:, 1:])

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _: rb.tree_predict(alias, X), range(8)))

    for probabilities, leaves in outputs:
        np.testing.assert_array_equal(leaves, [1, 2, 1])
        np.testing.assert_allclose(
            probabilities,
            [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.1, 0.7, 0.2]],
        )


# Compare the Rust tree adapter against sklearn across several configurations.
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"max_depth": 3, "min_samples_leaf": 2},
        {"min_samples_split": 0.15, "min_samples_leaf": 0.05},
        {"max_features": "sqrt"},
        {"max_features": "log2"},
        {"max_features": 0.7},
        {"min_impurity_decrease": 0.01},
    ],
)
def test_exact_tree_seeded_predictive_parity(params):
    X, y = make_classification(
        n_samples=160,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=13,
    )
    y = np.asarray(["zebra", "ant", "mouse"])[y]
    reference = DecisionTreeClassifier(random_state=7, **params).fit(X, y)
    rust = RustDecisionTreeClassifier(random_state=7, **params).fit(X, y)

    np.testing.assert_array_equal(rust.predict(X), reference.predict(X))
    np.testing.assert_allclose(rust.predict_proba(X), reference.predict_proba(X))
    np.testing.assert_allclose(rust.feature_importances_, reference.feature_importances_)
    assert rust.get_depth() == reference.get_depth()
    assert rust.get_n_leaves() == reference.get_n_leaves()
    assert rust.n_outputs_ == reference.n_outputs_ == 1
    assert rust.n_classes_ == reference.n_classes_
    np.testing.assert_array_equal(rust.classes_, reference.classes_)


# Check structural parity on a simple tie-free fixture.
def test_structural_parity_for_tie_free_fixture():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1, 2, 2])
    reference = DecisionTreeClassifier(random_state=0).fit(X, y)
    rust = RustDecisionTreeClassifier(random_state=0).fit(X, y)
    arrays = rust._inspect_model_arrays()

    np.testing.assert_array_equal(arrays["children_left"], reference.tree_.children_left)
    np.testing.assert_array_equal(arrays["children_right"], reference.tree_.children_right)
    np.testing.assert_array_equal(arrays["feature"], reference.tree_.feature)
    np.testing.assert_allclose(arrays["threshold"], reference.tree_.threshold)
    np.testing.assert_allclose(arrays["impurity"], reference.tree_.impurity)
    np.testing.assert_array_equal(arrays["n_node_samples"], reference.tree_.n_node_samples)
    np.testing.assert_allclose(
        arrays["weighted_n_node_samples"], reference.tree_.weighted_n_node_samples
    )
    np.testing.assert_allclose(arrays["value"], reference.tree_.value[:, 0, :])
    np.testing.assert_array_equal(rust.apply(X), reference.apply(X))


# Exercise rounding and the smallest non-trivial leaf boundary case.
def test_float32_tolerance_and_first_equal_gain_threshold_match_sklearn():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 1, 1, 0])
    reference = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    rust = RustDecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    assert rust._inspect_model_arrays()["threshold"][0] == reference.tree_.threshold[0] == 0.5

    near_constant = np.array([[0.0], [np.float32(1e-7)]], dtype=np.float64)
    rust.fit(near_constant, [0, 1])
    assert rust.get_n_leaves() == 1


# Verify feature-name handling and dataframe-backed inputs.
def test_dataframe_labels_feature_names_and_polars_input():
    X = pd.DataFrame({"left": [0.0, 1.0, 2.0, 3.0], "right": [1.0] * 4})
    y = np.array(["b", "b", "a", "a"])
    model = RustDecisionTreeClassifier(random_state=0).fit(X, y)
    np.testing.assert_array_equal(model.feature_names_in_, ["left", "right"])
    np.testing.assert_array_equal(model.predict(X), y)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(X[["right", "left"]])

    polars_model = RustDecisionTreeClassifier(random_state=0).fit(pl.from_pandas(X), y)
    np.testing.assert_array_equal(polars_model.predict(pl.from_pandas(X)), y)


# Confirm unsupported settings are rejected and input validation stays strict.
def test_validation_and_unsupported_configuration_boundaries():
    for estimator, reason in [
        (DecisionTreeClassifier(criterion="entropy"), "criterion"),
        (DecisionTreeClassifier(splitter="random"), "splitter"),
        (DecisionTreeClassifier(max_leaf_nodes=3), "max_leaf_nodes"),
        (DecisionTreeClassifier(class_weight="balanced"), "class_weight"),
    ]:
        supported, message = supports_rust_tree_classifier(estimator)
        assert not supported
        assert reason in message

    model = RustDecisionTreeClassifier(random_state=0)
    with pytest.raises(ValueError, match="Input X contains NaN"):
        model.fit([[0.0], [np.nan]], [0, 1])
    with pytest.raises(ValueError, match="0 sample"):
        model.fit(np.empty((0, 1)), np.empty(0))
    with pytest.raises(ValueError, match="sample_weight"):
        model.fit([[0.0], [1.0]], [0, 1], sample_weight=[1.0, 1.0])

    fitted = model.fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="Input contains NaN"):
        fitted.predict([[np.nan]])


# Check single-class handling and log-probability behavior on zeros.
def test_single_class_and_zero_probability_log_proba():
    X = np.arange(6, dtype=np.float32).reshape(-1, 1)
    model = RustDecisionTreeClassifier(random_state=0).fit(X, ["only"] * len(X))
    np.testing.assert_array_equal(model.predict(X), ["only"] * len(X))
    np.testing.assert_array_equal(model.predict_proba(X), np.ones((len(X), 1)))

    binary = RustDecisionTreeClassifier(random_state=0).fit([[0.0], [1.0]], [0, 1])
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        log_proba = binary.predict_log_proba([[0.0], [1.0]])
    assert np.isneginf(log_proba[[0, 1], [1, 0]]).all()
