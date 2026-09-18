"""Internal finite-value Rust adapter for exact decision-tree classification.
The Rust implementation is only intended to handle inputs that are fully
numeric and have no missing or infinite values."""

from __future__ import annotations

import numbers

import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    assert_all_finite,
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
        (estimator.max_leaf_nodes is None, "max_leaf_nodes is not supported yet"),
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
    """Sklearn-style adapter backed by the finite-value Rust exact tree.

    This class is internal and intentionally does not expose ``tree_``. Missing
    value training, sparse matrices, sample weights, and best-first growth are
    deferred to later milestones.
    """

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
                dict(dtype=np.float32, accept_sparse=False, ensure_all_finite=True),
                dict(ensure_2d=False, dtype=None),
            ),
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust decision tree")
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
        )
        return self

    # Shared prediction guard: validate input and keep only finite dense arrays.
    def _validated_prediction_data(self, X, check_input):
        check_is_fitted(self, "_tree_model_handle_")
        X = self._validate_X_predict(X, check_input)
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust decision tree")
        assert_all_finite(X)
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
