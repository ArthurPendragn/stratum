"""Show map fusion on a feature-engineering ``assign`` chain.

Three staged assigns (impute + parse time, calendar fields, trip distances)
collapse into one ``AssignMapOp`` containing three ordered batches. Each batch
materializes its columns before the next batch reads them.

Calendar ``.dt`` fields are left without ``fillna``/``astype`` so they fold
into column expressions; distance features use arithmetic that stays in the
map grammar (``** 0.5`` instead of ``apply_func(np.sqrt)``, etc.).

Run from the repo root:

    python examples/map_fusion_demo.py
"""
import numpy as np
import pandas as pd

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize


def sample_trips() -> pd.DataFrame:
    return pd.DataFrame({
        "unit_count": [1.0, np.nan, 3.0],
        "start_time": [
            "2012-01-15 08:30:00",
            "2012-06-20 14:00:00",
            "2013-03-01 09:15:00",
        ],
        "origin_x": [0.0, 1.0, 2.0],
        "origin_y": [0.0, 1.0, 2.0],
        "dest_x": [3.0, 4.0, 5.0],
        "dest_y": [4.0, 5.0, 6.0],
    })


def build_pipeline():
    df = st.as_data_op(sample_trips())

    df = df.assign(
        unit_count=df["unit_count"].fillna(1),
        start_time=df["start_time"].skb.apply_func(
            pd.to_datetime, format="%Y-%m-%d %H:%M:%S", errors="coerce",
        ),
    )

    df = df.assign(
        hour=df["start_time"].dt.hour,
        dayofweek=df["start_time"].dt.dayofweek,
        month=df["start_time"].dt.month,
        year=df["start_time"].dt.year,
    )

    dx = df["dest_x"] - df["origin_x"]
    dy = df["dest_y"] - df["origin_y"]

    return df.assign(
        euclidean_dist=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan_dist=(dx * dx) ** 0.5 + (dy * dy) ** 0.5,
        bearing=dy / (dx + 1e-9),
    )


def main():
    st.set_config(make_map_op=True, explain="logical", scheduler=True)
    root = build_pipeline()

    print("\n--- fusion OFF (three AssignMapOps) ---")
    optimize(root, OptConfig(fuse_assign_maps=False, algebraic_rewrites=False))

    print("\n--- fusion ON (one staged AssignMapOp; three ordered batches) ---")
    optimize(root, OptConfig(fuse_assign_maps=True, algebraic_rewrites=False))

    print("\n--- evaluated result ---")
    print(st._api.evaluate(root))


if __name__ == "__main__":
    main()
