from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._numeric_ops import NumericOp
from stratum.optimizer.ir._ops import PredictorOp, Op, TransformerOp
from stratum.optimizer.physical import (
    CURRENT_BACKENDS,
    CURRENT_LOGICAL_OPERATOR_TYPES,
    OperatorFamily,
    PhysicalImpl,
    PhysicalRegistry,
    RustPhysicalImpl,
    build_default_physical_registry,
)
from stratum.optimizer.physical._transform_execs import (RustOneHotEncoder,
                                                        RustStringEncoder,
                                                        SkrubStringEncoder,
                                                        StringEncoderOp)


def test_default_registry_has_logical_surface_and_adapter_candidates():
    registry = build_default_physical_registry()

    assert not registry.empty()
    assert registry.op_types()
    assert ConcatOp in CURRENT_LOGICAL_OPERATOR_TYPES
    assert NumericOp in CURRENT_LOGICAL_OPERATOR_TYPES
    assert "rust" in {backend.name for backend in CURRENT_BACKENDS}
    # StringEncoder migrated to its own physical op, so only OneHotEncoder's rust
    # kernel is still keyed on the logical TransformerOp.
    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")
    sklearn_candidates = registry.candidates_for(TransformerOp, backend_name="sklearn-skrub")
    assert len(rust_candidates) == 1
    assert all(candidate.backend_name == "rust" for candidate in rust_candidates)
    assert len(sklearn_candidates) == 1
    assert len(registry.candidates_for(PredictorOp, backend_name="sklearn-skrub")) == 1
    # The migrated StringEncoder physical op carries both a skrub and a rust impl.
    assert len(registry.candidates_for(StringEncoderOp, backend_name="rust")) == 1
    assert len(registry.candidates_for(StringEncoderOp, backend_name="sklearn-skrub")) == 1


def test_rust_kernels_are_class_based_impls():
    # After unification every Rust kernel is a class-based @rust_impl: OneHotEncoder
    # is still keyed on the logical TransformerOp, StringEncoder on its own op.
    registry = build_default_physical_registry()

    ohe_rust = registry.candidates_for(TransformerOp, backend_name="rust")
    se_rust = registry.candidates_for(StringEncoderOp, backend_name="rust")

    assert len(ohe_rust) == 1 and ohe_rust[0].impl_class is RustOneHotEncoder
    assert len(se_rust) == 1 and se_rust[0].impl_class is RustStringEncoder


def test_rust_impl_is_its_own_dataclass_with_capability_hints():
    # Rust has a dedicated PhysicalImpl subclass carrying scheduling capabilities,
    # read off the op class (RustPhysicalOp). Other backends stay on the base
    # PhysicalImpl, which has no such fields -- the schema is not shared.
    registry = build_default_physical_registry()

    (rust,) = registry.candidates_for(StringEncoderOp, backend_name="rust")
    (skrub,) = registry.candidates_for(StringEncoderOp, backend_name="sklearn-skrub")

    assert isinstance(rust, RustPhysicalImpl)
    assert rust.impl_class is RustStringEncoder
    assert rust.releases_gil and rust.data_parallel
    # Hints are sourced from the op class, so the entry and the operator agree.
    assert RustStringEncoder.releases_gil and RustStringEncoder.data_parallel

    assert type(skrub) is PhysicalImpl
    assert skrub.impl_class is SkrubStringEncoder
    assert not hasattr(skrub, "releases_gil")


def test_registry_registers_and_queries_impls_by_logical_type():
    registry = PhysicalRegistry()

    class DummyOp(Op):
        pass

    pandas_impl = PhysicalImpl(
        op_type=DummyOp,
        backend_name="pandas",
        input_format="frame",
        output_format="frame",
        supports=lambda op: isinstance(op, DummyOp),
        cost=lambda op, stats: 1.0,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("concat", mode, len(inputs)),
    )
    rust_impl = PhysicalImpl(
        op_type=DummyOp,
        backend_name="rust",
        input_format="frame",
        output_format="frame",
        supports=lambda op: isinstance(op, DummyOp),
        cost=lambda op, stats: 0.5,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("rust-concat", mode, len(inputs)),
    )

    registry.register(pandas_impl)
    registry.register(rust_impl)

    assert registry.candidates_for(DummyOp) == (pandas_impl, rust_impl)
    assert registry.candidates_for_op(DummyOp()) == (pandas_impl, rust_impl)
    assert registry.candidates_for(DummyOp, backend_name="rust") == (rust_impl,)
    assert registry.backends_for(DummyOp) == ("pandas", "rust")
    assert registry.candidates_by_backend("pandas") == (pandas_impl,)
    assert registry.candidates_by_backend("rust") == (rust_impl,)


def test_register_family_tracks_known_logical_types():
    registry = PhysicalRegistry()

    family = OperatorFamily(
        name="custom",
        op_types=(ConcatOp,),
        default_backends=("pandas",),
    )
    registry.register_family(family)

    assert registry.families() == (family,)
    assert ConcatOp in registry.op_types()
