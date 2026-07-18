from ._registry import (
    CURRENT_BACKENDS,
    CURRENT_LOGICAL_OPERATOR_TYPES,
    BackendSpec,
    OperatorFamily,
    PhysicalImpl,
    PhysicalRegistry,
    RustPhysicalImpl,
    build_default_physical_registry,
    get_default_physical_registry,
    numpy_impl,
    pandas_impl,
    physical_impl,
    polars_impl,
    rust_impl,
    sklearn_skrub_impl,
)
from ._physical_ops import PhysicalOp, RustPhysicalOp
from ._plan_context import PlanContext
from ._lowering import lower_to_physical, lowering_rule
from ._impl_selection import (
    FlagBasedSelector,
    ImplementationSelector,
    select_implementations,
)

__all__ = [
    "CURRENT_BACKENDS",
    "CURRENT_LOGICAL_OPERATOR_TYPES",
    "BackendSpec",
    "FlagBasedSelector",
    "ImplementationSelector",
    "OperatorFamily",
    "PhysicalImpl",
    "PhysicalOp",
    "PhysicalRegistry",
    "PlanContext",
    "RustPhysicalImpl",
    "RustPhysicalOp",
    "build_default_physical_registry",
    "get_default_physical_registry",
    "lower_to_physical",
    "lowering_rule",
    "numpy_impl",
    "pandas_impl",
    "physical_impl",
    "polars_impl",
    "rust_impl",
    "select_implementations",
    "sklearn_skrub_impl",
]
