"""End-to-end showcase of Stratum's logical abstractions and the physical
compiling pass, exercised through one realistic supervised pipeline.

A single e-commerce pipeline -- read a CSV, engineer features, encode the
categoricals, fit a model -- chains everything the recent optimizer work
introduced, over both frame backends:

* a source **read** ``pd.read_csv`` lowered to a source op;
* a **mask selection** ``df[predicate]`` that folds a boolean expression tree
  into one :class:`SelectionOp` (``stratum.optimizer.ir._selection_ops``);
* a folded **column map** ``df.assign(...)`` -- arithmetic, a ``.dt`` accessor
  over a parsed date -- collapsing into one :class:`AssignMapOp`
  (``stratum.optimizer.ir._map_ops``);
* a literal **column projection** ``df[[...]]`` -> :class:`ColumnProjectionOp`
  and two selector-driven ``skb.select(...)`` splits -> :class:`ColumnSelectorOp`
  (``stratum.optimizer.ir._projection_ops``);
* a **transformer** (:class:`TransformerOp`) with *multiple* registered physical
  implementations -- the sklearn/skrub ``StringEncoder`` and its ``rust`` kernel
  -- and an **estimator** (:class:`EstimatorOp`) that fits the model.

The tests assert three things:

1. each backend-agnostic *logical* abstraction is recognised;
2. the *compiling* pass binds the frame abstractions to concrete
   :class:`PhysicalOp` implementations, and binds the transformer to the
   backend implementation chosen by the plan context (skrub vs. Rust);
3. the compiled plan trains and scores end-to-end via ``make_grid_search`` on
   both backends.
"""
import pytest
import pandas as pd
import numpy as np
from skrub import StringEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score

import stratum as st
from stratum.adapters.string_encoder import (RustyStringEncoder,
                                             supports_rust_string_encoder)
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.ir._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._projection_ops import (
    ColumnProjectionOp, ColumnSelectorOp)
from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._ops import PredictorOp, TransformerOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.tests._helpers import csv_file
from stratum.tests.logical_optimizer.test_dataframe_ops import force_polars


def make_orders(n=60):
    """A seeded, messy order book with a regression target (``satisfaction``).

    Cancelled and zero-quantity rows are noise the filter must drop; the string
    ``category``/``country`` columns feed the encoder."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "order_id": range(n),
        "order_date": pd.to_datetime(
            rng.choice(pd.date_range("2021-01-01", "2021-12-31"), size=n)
        ).astype(str),
        "quantity": rng.randint(0, 30, size=n),
        "unit_price": rng.uniform(1, 100, size=n).round(2),
        "category": rng.choice(["books", "toys", "home", "garden"], size=n),
        "country": rng.choice(["DE", "FR", "US"], size=n),
        "status": rng.choice(["completed", "cancelled"], size=n, p=[0.8, 0.2]),
        "satisfaction": rng.uniform(1, 5, size=n).round(1),
    })


def build_pipeline(file_path, model=None):
    """read -> filter -> map -> projection -> select -> encode -> concat -> fit."""
    model = model if model is not None else Ridge(random_state=0)

    # (0) READ: the pipeline starts from a CSV on disk, lowered to a source read.
    src = st.as_data_op(file_path).skb.apply_func(pd.read_csv)

    # (1) FILTER: keep completed orders that actually shipped units. The whole
    # boolean tree folds into a single SelectionOp(MASK) predicate.
    active = src[(src["status"] == "completed") & (src["quantity"] > 0)]
    y = active["satisfaction"].skb.mark_as_y()

    # (2) MAP: derive numeric features. Arithmetic and a `.dt` accessor over a
    # parsed date are natively lazy, so they fold into one AssignMapOp.
    order_date = active["order_date"].skb.apply_func(pd.to_datetime)
    featured = active.assign(
        revenue=active["quantity"] * active["unit_price"],
        net_revenue=active["quantity"] * active["unit_price"] * (1 - 0.19),
        order_month=order_date.dt.month,
        is_bulk=active["quantity"] >= 10,
    )

    # (3) PROJECTION: keep the modelling columns by literal name (drops the id,
    # raw date, status and target) -> ColumnProjectionOp.
    X = featured[["quantity", "unit_price", "revenue", "net_revenue",
                  "order_month", "category", "country"]].skb.mark_as_X()

    # (4) SELECTORS: split numeric features from the string columns, each via a
    # ColumnSelectorOp.
    X_num = X.skb.select(st.selectors.numeric())
    X_cat = X.skb.select(~st.selectors.numeric())

    # (5) TRANSFORMER: encode the string columns. StringEncoder has both a
    # sklearn/skrub and a Rust physical implementation; the plan context picks.
    X_cat_enc = X_cat.skb.apply(StringEncoder())

    # (6) CONCAT the numeric + encoded blocks, then (7) fit the ESTIMATOR.
    X_vec = X_num.skb.concat([X_cat_enc], axis=1)
    return X_vec.skb.apply(model, y=y)


def _optimize(dag):
    return optimize_(dag, OptConfig(dataframe_ops=True))[0]


@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    with force_polars(request.param):
        yield request.param


def test_frame_ops_pipeline_plan(polars):
    """Every logical abstraction is recognised, and the compiling pass binds the
    frame abstractions to concrete physical implementations."""
    with csv_file(make_orders()) as path:
        ops = _optimize(build_pipeline(path))

    # The frame abstractions each appear and are compiled to a PhysicalOp -- no
    # abstract frame op survives to execution.
    for cls in (SelectionOp, AssignMapOp, ColumnProjectionOp, ColumnSelectorOp):
        matches = [o for o in ops if isinstance(o, cls)]
        assert matches, f"no {cls.__name__} in plan"
        assert all(isinstance(o, PhysicalOp) for o in matches), \
            f"{cls.__name__} not compiled to a physical op"

    # The filter folded into a mask predicate (not a method selection).
    sel = next(o for o in ops if isinstance(o, SelectionOp))
    assert sel.kind is SelectionKind.MASK

    # The learning ops (transformer, estimator) and the concat are present.
    assert any(isinstance(o, TransformerOp) for o in ops)
    assert any(isinstance(o, PredictorOp) for o in ops)
    assert any(isinstance(o, ConcatOp) for o in ops)


def test_transformer_binds_selected_backend(polars):
    """The *same* logical TransformerOp binds to different physical
    implementations depending on the plan context: the sklearn/skrub encoder by
    default, the Rust kernel when ``rust_backend`` is on."""
    supported, reason = supports_rust_string_encoder(StringEncoder())
    if not supported:
        pytest.skip(f"Rust StringEncoder unavailable: {reason}")

    def encoder_estimator(rust):
        with csv_file(make_orders()) as path, st.config(rust_backend=rust):
            ops = _optimize(build_pipeline(path))
        transformers = [o for o in ops if isinstance(o, TransformerOp)]
        assert len(transformers) == 1
        return transformers[0].estimator

    # Default: the backend-agnostic sklearn/skrub implementation.
    default_est = encoder_estimator(rust=False)
    assert not isinstance(default_est, RustyStringEncoder)

    # rust_backend on: the Rust kernel is bound in at plan time.
    rust_est = encoder_estimator(rust=True)
    assert isinstance(rust_est, RustyStringEncoder)
    assert rust_est._stratum_force_rust


def test_selection_binds_query_impl_under_flag():
    """The *same* logical MASK ``SelectionOp`` binds to different pandas impls
    depending on the plan context: boolean-mask indexing by default, the
    ``DataFrame.query()`` fast path when ``pandas_query`` is on. The choice is a
    plan-time bind, so no ``pandas_query`` branch survives into execution.

    Pandas-only: the flag has no effect on the polars backend, so this test does
    not use the ``polars`` fixture."""
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    def selection_impl(pandas_query):
        with csv_file(make_orders()) as path, st.config(pandas_query=pandas_query):
            ops = _optimize(build_pipeline(path))
        sels = [o for o in ops if isinstance(o, SelectionOp)]
        assert len(sels) == 1
        # The predicate (`status == "completed" & quantity > 0`) is fully
        # query-expressible, so the flag alone decides which impl binds.
        assert sels[0].kind is SelectionKind.MASK
        return sels[0]

    # Default: boolean-mask indexing.
    assert isinstance(selection_impl(pandas_query=False), PandasIndexSelectionOp)
    # pandas_query on: the query() fast path is bound in at plan time.
    assert isinstance(selection_impl(pandas_query=True), PandasQuerySelectionOp)


def test_query_selection_trains_end_to_end():
    """With ``pandas_query`` on, the query fast path runs through the scheduler
    and the pipeline still trains and scores end-to-end (pandas backend)."""
    scorer = make_scorer(r2_score)
    with csv_file(make_orders()) as path:
        preds = build_pipeline(path)
        with st.config(scheduler=True, rust_backend=False, pandas_query=True, explain=("logical", "physical_impl")):
            search = preds.skb.make_grid_search(fitted=True, cv=2, scoring=scorer)
            assert search.results_ is not None
            assert len(search.results_) > 0


def test_frame_ops_pipeline_grid_search(polars):
    """The compiled plan trains and scores end-to-end through Stratum's
    scheduler, driven by ``make_grid_search`` -- the same entry point the other
    application tests use. The pipeline carries a single candidate (no
    ``choose_from``), which grid search handles as a one-pipeline search."""
    scorer = make_scorer(r2_score)
    with csv_file(make_orders()) as path:
        preds = build_pipeline(path)
        with st.config(scheduler=True, rust_backend=False, debug_graph=False):
            search = preds.skb.make_grid_search(fitted=True, cv=2, scoring=scorer)
            assert search.results_ is not None
            assert len(search.results_) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
