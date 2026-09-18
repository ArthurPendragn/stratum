"""Standalone Rust adapters for exact tree and serial forest classification.

Both estimators share the compact native tree model, exact split finder, NaN
routing, and prediction traversal. Forest bootstrap samples stay in native
code as multiplicity weights rather than duplicated rows.
"""

from __future__ import annotations

import numbers

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_is_fitted,
    column_or_1d,
    validate_data,
)

from .. import _rust_backend as rb


# Keep the supported configuration narrow so the Rust path stays parity-safe.
def supports_rust_tree_classifier(estimator) -> tuple[bool, str]:
    """Return whether an estimator belongs to the initial exact-tree subset."""
    if not isinstance(estimator, DecisionTreeClassifier):
        return False, "estimator is not a sklearn DecisionTreeClassifier"
    if not rb.HAVE_RUST or rb.tree_fit_exact is None:
        return False, "Rust decision-tree runtime is not available"
    checks = (
        (estimator.criterion == "gini", "criterion must be 'gini'"),
        (estimator.splitter == "best", "splitter must be 'best'"),
        (estimator.min_weight_fraction_leaf == 0.0, "min_weight_fraction_leaf must be 0"),
        (estimator.class_weight is None, "class_weight is not supported"),
        (estimator.ccp_alpha == 0.0, "ccp_alpha must be 0"),
        (estimator.monotonic_cst is None, "monotonic_cst is not supported"),
        (
            estimator.random_state is None
            or isinstance(estimator.random_state, numbers.Integral),
            "random_state must be None or an integer",
        ),
    )
    for supported, reason in checks:
        if not supported:
            return False, reason
    return True, ""


class RustDecisionTreeClassifier(DecisionTreeClassifier):
    """Sklearn-style adapter backed by the Rust exact tree."""

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._validate_params()
        supported, reason = supports_rust_tree_classifier(self)
        if not supported:
            raise ValueError(f"unsupported Rust decision-tree configuration: {reason}")
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by the Rust decision tree")
        if not check_input:
            raise ValueError("check_input=False is not supported during fit")

        # Validate and normalize the input exactly once before crossing into Rust.
        X, y = validate_data(
            self,
            X,
            y,
            validate_separately=(
                dict(dtype=np.float32, accept_sparse=False, ensure_all_finite="allow-nan"),
                dict(ensure_2d=False, dtype=None),
            ),
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust decision tree")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1
        n_samples, self.n_features_in_ = X.shape

        # Derive sklearn-style integer hyperparameters from the public estimator API.
        min_samples_leaf = (
            int(self.min_samples_leaf)
            if isinstance(self.min_samples_leaf, numbers.Integral)
            else int(np.ceil(self.min_samples_leaf * n_samples))
        )
        min_samples_split = (
            int(self.min_samples_split)
            if isinstance(self.min_samples_split, numbers.Integral)
            else max(2, int(np.ceil(self.min_samples_split * n_samples)))
        )
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else int(self.max_depth)
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif self.max_features == "sqrt":
            self.max_features_ = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            self.max_features_ = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, numbers.Integral):
            self.max_features_ = int(self.max_features)
        else:
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))

        # Convert inputs to contiguous native arrays before calling the Rust runtime.
        random_state = check_random_state(self.random_state)
        tree_seed = int(random_state.randint(0, np.iinfo(np.int32).max))
        X = np.ascontiguousarray(X, dtype=np.float32)
        encoded = np.ascontiguousarray(encoded, dtype=np.int64)
        self._tree_model_handle_ = rb.tree_fit_exact(
            X,
            encoded,
            int(self.n_classes_),
            max_depth,
            min_samples_split,
            min_samples_leaf,
            self.max_features_,
            float(self.min_impurity_decrease),
            tree_seed,
            self.max_leaf_nodes,
        )
        return self

    # Shared prediction guard: allow NaNs but reject infinities and sparse input.
    def _validated_prediction_data(self, X, check_input):
        check_is_fitted(self, "_tree_model_handle_")
        X = self._validate_X_predict(X, check_input)
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust decision tree")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        return np.ascontiguousarray(X, dtype=np.float32)

    # Native probability prediction and sklearn-style array conversion.
    def predict_proba(self, X, check_input=True):
        X = self._validated_prediction_data(X, check_input)
        probabilities, _ = rb.tree_predict(self._tree_model_handle_, X)
        return np.asarray(probabilities, dtype=np.float64)

    # Class labels are reconstructed from the native probability matrix.
    def predict(self, X, check_input=True):
        probabilities = self.predict_proba(X, check_input=check_input)
        return self.classes_.take(np.argmax(probabilities, axis=1), axis=0)

    def predict_log_proba(self, X):
        with np.errstate(divide="warn"):
            return np.log(self.predict_proba(X))

    # Expose leaf indices for parity checks and downstream diagnostics.
    def apply(self, X, check_input=True):
        X = self._validated_prediction_data(X, check_input)
        _, leaves = rb.tree_predict(self._tree_model_handle_, X)
        return np.asarray(leaves, dtype=np.intp)

    # Mirror sklearn's tree introspection helpers.
    def get_depth(self):
        return int(self._inspect_model_arrays()["max_depth"])

    def get_n_leaves(self):
        return int(self._inspect_model_arrays()["n_leaves"])

    @property
    def feature_importances_(self):
        return np.asarray(
            self._inspect_model_arrays()["feature_importances"], dtype=np.float64
        )

    # Pull the serialized native tree arrays for inspection and testing.
    def _inspect_model_arrays(self):
        check_is_fitted(self, "_tree_model_handle_")
        return rb.tree_model_arrays(self._tree_model_handle_)


def supports_rust_random_forest_classifier(estimator) -> tuple[bool, str]:
    """Return whether an estimator belongs to the standalone serial subset."""
    if not isinstance(estimator, RandomForestClassifier):
        return False, "estimator is not a sklearn RandomForestClassifier"
    if not rb.HAVE_RUST or rb.forest_fit_exact is None:
        return False, "Rust random-forest runtime is not available"
    checks = (
        (estimator.criterion == "gini", "criterion must be 'gini'"),
        (estimator.n_jobs in (None, 1), "n_jobs must be None or 1 for the serial forest"),
        (
            estimator.bootstrap or estimator.max_samples is None,
            "max_samples requires bootstrap=True",
        ),
        (not estimator.oob_score, "oob_score is not supported"),
        (not estimator.warm_start, "warm_start is not supported"),
        (estimator.verbose == 0, "verbose must be 0"),
        (estimator.min_weight_fraction_leaf == 0.0, "min_weight_fraction_leaf must be 0"),
        (estimator.class_weight is None, "class_weight is not supported"),
        (estimator.ccp_alpha == 0.0, "ccp_alpha must be 0"),
        (estimator.monotonic_cst is None, "monotonic_cst is not supported"),
        (
            estimator.random_state is None
            or isinstance(estimator.random_state, numbers.Integral),
            "random_state must be None or an integer",
        ),
    )
    for supported, reason in checks:
        if not supported:
            return False, reason
    return True, ""


class RustRandomForestClassifier(RandomForestClassifier):
    """Sklearn-style adapter backed by a serial Rust exact random forest."""

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        supported, reason = supports_rust_random_forest_classifier(self)
        if not supported:
            raise ValueError(f"unsupported Rust random-forest configuration: {reason}")
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by the Rust random forest")

        X, y = validate_data(
            self,
            X,
            y,
            validate_separately=(
                dict(dtype=np.float32, accept_sparse=False, ensure_all_finite="allow-nan"),
                dict(ensure_2d=False, dtype=None),
            ),
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust random forest")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1
        n_samples, self.n_features_in_ = X.shape

        min_samples_leaf = (
            int(self.min_samples_leaf)
            if isinstance(self.min_samples_leaf, numbers.Integral)
            else int(np.ceil(self.min_samples_leaf * n_samples))
        )
        min_samples_split = (
            int(self.min_samples_split)
            if isinstance(self.min_samples_split, numbers.Integral)
            else max(2, int(np.ceil(self.min_samples_split * n_samples)))
        )
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else int(self.max_depth)
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif self.max_features == "sqrt":
            self.max_features_ = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            self.max_features_ = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, numbers.Integral):
            self.max_features_ = int(self.max_features)
        else:
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))

        if not self.bootstrap:
            n_bootstrap = n_samples
        elif self.max_samples is None:
            n_bootstrap = n_samples
        elif isinstance(self.max_samples, numbers.Integral):
            n_bootstrap = int(self.max_samples)
            if n_bootstrap > n_samples:
                raise ValueError(
                    f"`max_samples` must be <= n_samples={n_samples} but got "
                    f"value {n_bootstrap}"
                )
        else:
            n_bootstrap = max(round(n_samples * self.max_samples), 1)

        random_state = check_random_state(self.random_state)
        tree_seeds = np.ascontiguousarray(
            random_state.randint(
                0, np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64
            ),
            dtype=np.int64,
        )
        X = np.ascontiguousarray(X, dtype=np.float32)
        encoded = np.ascontiguousarray(encoded, dtype=np.int64)
        self._forest_model_handle_ = rb.forest_fit_exact(
            X,
            encoded,
            tree_seeds,
            int(self.n_classes_),
            max_depth,
            min_samples_split,
            min_samples_leaf,
            self.max_features_,
            float(self.min_impurity_decrease),
            self.max_leaf_nodes,
            bool(self.bootstrap),
            n_bootstrap,
        )
        return self

    def _validated_prediction_data(self, X):
        check_is_fitted(self, "_forest_model_handle_")
        X = validate_data(
            self,
            X,
            reset=False,
            dtype=np.float32,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust random forest")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        return np.ascontiguousarray(X, dtype=np.float32)

    def predict_proba(self, X):
        X = self._validated_prediction_data(X)
        return np.asarray(
            rb.forest_predict(self._forest_model_handle_, X), dtype=np.float64
        )

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_.take(np.argmax(probabilities, axis=1), axis=0)

    def predict_log_proba(self, X):
        with np.errstate(divide="warn"):
            return np.log(self.predict_proba(X))

    @property
    def feature_importances_(self):
        return np.asarray(
            self._inspect_model()["feature_importances"], dtype=np.float64
        )

    def _inspect_model(self):
        check_is_fitted(self, "_forest_model_handle_")
        return rb.forest_model_info(self._forest_model_handle_)
