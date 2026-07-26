"""Macrobenchmark TfidfVectorizer through Stratum.

Both legs use ``stratum.TfidfVectorizer`` (the patched adapter):

* ``version=rust``     → ``rust_backend=True``  (Rust kernel)
* ``version=sklearn``  → ``rust_backend=False`` (sklearn fallback)

Thread count is set via ``stratum.set_config(num_threads=...)`` and
``SKRUB_RUST_THREADS`` before imports in each worker subprocess.

Workers report wall time of ``fit_transform``, process peak RSS, kernel
RSS delta, and CSR nbytes. By default the rust leg also enables
``debug_timing`` for one extra fit_transform and harvests stage timings.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


LENGTH_VALS = [10_000, 100_000, 1_000_000]
N_JOBS_VALS = [1, 2, 4, 8, 24]
WORD_VALS = [1000, 10000]

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
CSV_PATH = RESULTS_DIR / "macrobenchmark.csv"

# Matches util::print_timing / adapters' rb.print_timing when following
# DEBUG_TIMING.md. Stage names may contain spaces.
_RUST_STAGE_RE = re.compile(
    r"^\[rust\]\s+(?P<stage>.+?):\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\s*$"
)
_PYTHON_STAGE_RE = re.compile(
    r"^\[python\]\s+(?P<stage>.+?):\s*(?P<sec>\d+(?:\.\d+)?)\s*s\s*$"
)
_STAGE_BEGIN = "STAGE_PROFILE_BEGIN"
_STAGE_END = "STAGE_PROFILE_END"

# Preferred column order for console / stacked stage plots.
_STAGE_ORDER = (
    "tv py_materialize",
    "tfidf_vectorizer_fit",
    "tv pass_a_stats",
    "tv select_vocab",
    "tv pass_b_emit",
    "tv map_chunks",
    "tv assemble_csr",
    "tfidf_vectorizer_transform",
)


def make_series(n_rows, seed, vocab_size, avg_words=8, words_len_range=(3, 10)) -> pd.Series:
    rng = np.random.default_rng(seed)

    def rand_word():
        size = rng.integers(words_len_range[0], words_len_range[1])
        return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=size))

    vocab = [rand_word() for _ in range(vocab_size)]
    n_per_row = np.maximum(1, rng.poisson(avg_words, size=n_rows))

    rows = []
    for k in n_per_row:
        idx = rng.integers(0, vocab_size, size=k)
        rows.append(" ".join(vocab[i] for i in idx))

    return pd.Series(rows, name="text")


def _read_proc_status_kb(keys: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            for key in keys:
                if line.startswith(f"{key}:"):
                    out[key] = int(line.split()[1])
    except OSError:
        pass
    return out


def _current_rss_mb() -> float:
    """Current RSS in MiB (Linux VmRSS when available)."""
    vals = _read_proc_status_kb({"VmRSS"})
    if "VmRSS" in vals:
        return vals["VmRSS"] / 1024.0
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _peak_rss_mb() -> float:
    """Process peak RSS in MiB (Linux VmHWM when available)."""
    vals = _read_proc_status_kb({"VmHWM"})
    if "VmHWM" in vals:
        return vals["VmHWM"] / 1024.0
    import resource

    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _csr_nbytes_mb(matrix) -> float:
    """Backing-store size of a scipy CSR (data + indices + indptr) in MiB."""
    return (
        int(matrix.data.nbytes)
        + int(matrix.indices.nbytes)
        + int(matrix.indptr.nbytes)
    ) / (1024.0 * 1024.0)


def _parse_stage_lines(lines: list[str]) -> dict[str, float]:
    """Parse timing lines, optionally clipped to a STAGE_PROFILE_* window."""
    begin = next((i for i, line in enumerate(lines) if _STAGE_BEGIN in line), None)
    end = next((i for i, line in enumerate(lines) if _STAGE_END in line), None)
    if begin is not None and end is not None and end > begin:
        lines = lines[begin + 1 : end]

    stages: dict[str, float] = {}
    for line in lines:
        rust = _RUST_STAGE_RE.match(line.strip())
        if rust:
            stages[rust.group("stage")] = float(rust.group("ms"))
            continue
        py = _PYTHON_STAGE_RE.match(line.strip())
        if py:
            stages[py.group("stage")] = float(py.group("sec")) * 1000.0
    return stages


def parse_stage_timings(stderr: str, stdout: str = "") -> dict[str, float]:
    """Parse ``[rust]`` / ``[python]`` debug_timing lines into stage → ms.

    Rust helpers write to stderr; Python helpers write to stdout. Markers are
    emitted on both streams, so each stream is clipped independently and the
    dicts are merged (later values win on duplicate stage names).
    """
    stages = _parse_stage_lines(stderr.splitlines())
    stages.update(_parse_stage_lines(stdout.splitlines()))
    return stages


def _format_stages(stages: dict[str, float]) -> str:
    if not stages:
        return ""
    ordered = [s for s in _STAGE_ORDER if s in stages]
    ordered += sorted(k for k in stages if k not in _STAGE_ORDER)
    return ", ".join(f"{k}={stages[k]:.1f}ms" for k in ordered)


def run_worker(
    n_jobs: int,
    n_unique_words: int,
    dataset_length: int,
    version: str,
    profile_stages: bool,
) -> None:
    # Must be set before importing NumPy / sklearn / Stratum so Rayon picks it up.
    os.environ["SKRUB_RUST_THREADS"] = str(n_jobs)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Keep timed runs quiet; enable only for the optional profile pass.
    os.environ["SKRUB_RUST_DEBUG_TIMING"] = "0"

    import stratum
    from stratum import _rust_backend as rust_backend

    if version == "rust":
        if not rust_backend.HAVE_RUST:
            raise RuntimeError(
                "The Rust extension is unavailable. Build it with "
                "`maturin develop --release` before benchmarking."
            )
        stratum.set_config(
            rust_backend=True,
            allow_patch=True,
            num_threads=n_jobs,
            debug_timing=False,
        )
    else:
        stratum.set_config(
            rust_backend=False,
            allow_patch=True,
            num_threads=n_jobs,
            debug_timing=False,
        )

    # Always go through Stratum's patched adapter; the rust_backend flag
    # selects Rust vs sklearn fallback inside the same class.
    TfidfVectorizer = stratum.TfidfVectorizer

    X = make_series(dataset_length, seed=42, vocab_size=n_unique_words)

    probe = TfidfVectorizer()
    if version == "rust" and not probe._rust_enabled():
        raise RuntimeError(
            "TfidfVectorizer Rust path is not enabled; check rust_backend/"
            "allow_patch and that the extension is built."
        )

    X_small = X.iloc[: min(1000, dataset_length)]
    _ = TfidfVectorizer().fit_transform(X_small)
    gc.collect()

    times = []
    rss_deltas = []
    csr_mbs = []
    for _ in range(3):
        # Fresh estimator each trial: TfidfVectorizer is stateful (fit builds
        # vocabulary / IDF), unlike HashingVectorizer's pure transform.
        enc = TfidfVectorizer()
        gc.collect()
        rss0 = _current_rss_mb()
        t0 = time.perf_counter()
        res = enc.fit_transform(X)
        t1 = time.perf_counter()
        rss1 = _current_rss_mb()
        times.append((t1 - t0) * 1000.0)
        rss_deltas.append(max(0.0, rss1 - rss0))
        csr_mbs.append(_csr_nbytes_mb(res))
        del res, enc
        gc.collect()

    # Pair memory with the fastest timed fit_transform (same index as time_ms).
    best_i = int(np.argmin(times))

    stages: dict[str, float] = {}
    if profile_stages and version == "rust":
        # Untimed pass with debug_timing so stage lines hit stderr/stdout
        # without biasing the wall-clock measurements above. Markers go to
        # both streams because Rust logs stderr and Python logs stdout.
        stratum.set_config(debug_timing=True)
        for stream in (sys.stderr, sys.stdout):
            print(_STAGE_BEGIN, file=stream, flush=True)
        # Capture this process's own stage lines by temporarily tee-ing is
        # awkward; the master scrapes worker stdout/stderr instead. Still
        # emit markers here so the scrape window is well-defined.
        _ = TfidfVectorizer().fit_transform(X)
        for stream in (sys.stderr, sys.stdout):
            print(_STAGE_END, file=stream, flush=True)
        stratum.set_config(debug_timing=False)
        gc.collect()

    payload = {
        "time_ms": times[best_i],
        "peak_rss_mb": round(_peak_rss_mb(), 3),
        "rss_delta_mb": round(rss_deltas[best_i], 3),
        "csr_nbytes_mb": round(csr_mbs[best_i], 3),
        "stages_ms": stages,
    }
    print(f"RESULT:{json.dumps(payload, separators=(',', ':'))}")


def _run_case(
    n_jobs: int,
    n_unique_words: int,
    dataset_length: int,
    version: str,
    profile_stages: bool,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--n-jobs",
        str(n_jobs),
        "--n-unique-words",
        str(n_unique_words),
        "--dataset-length",
        str(dataset_length),
        "--version",
        version,
    ]
    if profile_stages:
        cmd.append("--profile-stages")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            payload = json.loads(line.removeprefix("RESULT:"))
            break
    if payload is None:
        raise RuntimeError(f"Worker produced no RESULT line:\n{proc.stdout}")

    # Rust print_timing → stderr; Python rb.print_timing → stdout.
    stages = parse_stage_timings(proc.stderr, proc.stdout)
    if stages:
        payload["stages_ms"] = stages
    elif profile_stages and version == "rust":
        raise RuntimeError(
            "profile-stages requested but no stage timings were harvested.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return payload


def _row_has_memory(row: dict) -> bool:
    return (
        row.get("peak_rss_mb") is not None
        and row.get("rss_delta_mb") is not None
        and row.get("csr_nbytes_mb") is not None
        and str(row.get("peak_rss_mb")) != "nan"
    )


def _row_has_stages(row: dict, version: str) -> bool:
    if version != "rust":
        return True
    raw = row.get("stages_ms_json", "{}")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return False
    try:
        stages = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(stages)


def run_master(tag: str, profile_stages: bool, force: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    if CSV_PATH.exists() and not force:
        try:
            results = pd.read_csv(CSV_PATH).to_dict("records")
            print(f"Loaded {len(results)} existing benchmark results.")
        except Exception:
            pass
    elif force and CSV_PATH.exists():
        print(f"--force: ignoring existing {CSV_PATH}")

    def already_done(length: int, words: int, n_jobs: int, version: str) -> bool:
        for r in results:
            if not (
                r["dataset_length"] == length
                and r["n_unique_words"] == words
                and r["n_jobs"] == n_jobs
                and r["version"] == version
            ):
                continue
            if not _row_has_memory(r):
                continue
            if profile_stages and not _row_has_stages(r, version):
                continue
            return True
        return False

    def record(length: int, words: int, n_jobs: int, version: str, metrics: dict) -> None:
        # Drop incomplete prior rows for this key so re-runs replace them.
        nonlocal results
        results = [
            r
            for r in results
            if not (
                r["dataset_length"] == length
                and r["n_unique_words"] == words
                and r["n_jobs"] == n_jobs
                and r["version"] == version
            )
        ]
        stages = metrics.get("stages_ms") or {}
        row = {
            "n_jobs": n_jobs,
            "n_unique_words": words,
            "dataset_length": length,
            "version": version,
            "time_ms": metrics["time_ms"],
            "peak_rss_mb": metrics["peak_rss_mb"],
            "rss_delta_mb": metrics["rss_delta_mb"],
            "csr_nbytes_mb": metrics["csr_nbytes_mb"],
            "stages_ms_json": json.dumps(stages, sort_keys=True),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(CSV_PATH, index=False)
        stage_note = f"  stages=[{_format_stages(stages)}]" if stages else ""
        print(
            f"Time: {metrics['time_ms']:.3f} ms  "
            f"peak_rss={metrics['peak_rss_mb']:.1f} MiB  "
            f"rss_delta={metrics['rss_delta_mb']:.1f} MiB  "
            f"csr={metrics['csr_nbytes_mb']:.1f} MiB"
            f"{stage_note}"
        )

    # Both legs run through Stratum; only rust_backend / num_threads differ.
    for length in LENGTH_VALS:
        for words in WORD_VALS:
            for n_jobs in N_JOBS_VALS:
                for version in ("sklearn", "rust"):
                    if already_done(length, words, n_jobs, version):
                        print(
                            f"Skipping {version} for length={length}, words={words}, "
                            f"n_jobs={n_jobs} (already exists)"
                        )
                        continue
                    print(
                        f"Running {version} for length={length}, words={words}, "
                        f"n_jobs={n_jobs}..."
                    )
                    try:
                        metrics = _run_case(
                            n_jobs, words, length, version, profile_stages
                        )
                    except RuntimeError as exc:
                        print(exc)
                        continue
                    record(length, words, n_jobs, version, metrics)

    print("All benchmarks finished! Running plot generation...")
    generate_plots(tag)


def generate_plots(tag: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df_all = pd.read_csv(CSV_PATH)
    df = df_all[df_all["version"] == "rust"]
    # Baseline: stratum with rust_backend=False at matching n_jobs=1.
    df_sklearn = df_all[
        (df_all["version"] == "sklearn") & (df_all["n_jobs"] == 1)
    ]

    unit = "ms"
    n_jobs_vals = sorted(df["n_jobs"].unique())
    word_vals = sorted(df["n_unique_words"].unique())
    length_vals = [10_000, 100_000, 1_000_000]

    metrics = (
        ("time_ms", f"time ({unit})", "latency"),
        ("peak_rss_mb", "peak RSS (MiB)", "peak RSS"),
        ("rss_delta_mb", "kernel ΔRSS (MiB)", "kernel ΔRSS"),
        ("csr_nbytes_mb", "CSR nbytes (MiB)", "CSR nbytes"),
    )

    fig, axes = plt.subplots(
        len(length_vals),
        len(metrics),
        figsize=(4.2 * len(metrics), 3.4 * len(length_vals)),
        squeeze=False,
    )

    njobs_colors = {
        1: "#C4C4C4",
        2: "#525252",
        4: "#f47e7e",
        8: "#620000",
        24: "#c1272d",
    }
    sklearn_color = "#444444"

    n_bars = len(n_jobs_vals) + 1
    total_width = 0.7
    bar_width = total_width / n_bars
    offsets = np.linspace(
        -(total_width - bar_width) / 2, (total_width - bar_width) / 2, n_bars
    )

    for i, length in enumerate(length_vals):
        for col, (metric, ylabel, title_metric) in enumerate(metrics):
            if metric not in df_all.columns:
                axes[i][col].set_visible(False)
                continue
            ax = axes[i][col]
            sub = df[df["dataset_length"] == length]
            sub_sklearn = df_sklearn[df_sklearn["dataset_length"] == length]
            x = np.arange(len(word_vals))

            sklearn_vals = []
            for w in word_vals:
                row = sub_sklearn[sub_sklearn.n_unique_words == w]
                sklearn_vals.append(
                    row[metric].iloc[0] if not row.empty else float("nan")
                )
            ax.bar(x + offsets[0], sklearn_vals, bar_width, color=sklearn_color)

            for offset, n_jobs in zip(offsets[1:], n_jobs_vals):
                vals = []
                for w in word_vals:
                    row = sub[(sub.n_unique_words == w) & (sub.n_jobs == n_jobs)]
                    vals.append(row[metric].iloc[0] if not row.empty else float("nan"))
                ax.bar(
                    x + offset, vals, bar_width, color=njobs_colors.get(n_jobs, "#888")
                )

                if metric == "time_ms" and n_jobs in (1, 24):
                    origin_nudge = {1: -1, 24: 0}[n_jobs] * 0.015
                    for xi_sklearn, xi_rust, t, st in zip(
                        x + offsets[0], x + offset, vals, sklearn_vals
                    ):
                        if not (np.isfinite(t) and np.isfinite(st) and t > 0):
                            continue
                        origin_y = st * (1 + origin_nudge)
                        ax.annotate(
                            "",
                            xy=(xi_rust, t),
                            xytext=(xi_sklearn, origin_y),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="black",
                                lw=0.8,
                                connectionstyle="angle,angleA=0,angleB=90",
                            ),
                        )
                        ax.annotate(
                            f"{st / t:.1f}×",
                            xy=(xi_rust, (origin_y + t) / 2),
                            xytext=(4, 0),
                            textcoords="offset points",
                            ha="left",
                            va="center",
                            fontsize=7,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(word_vals)
            ax.set_title(f"n_rows = {length:,} ({title_metric})")
            ax.set_xlabel("n_unique_words")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", ls=":", alpha=0.5)

    legend_handles = [
        Patch(color=sklearn_color, label="stratum (rust_backend=False)")
    ]
    legend_handles += [
        Patch(color=njobs_colors.get(j, "#888"), label=str(j)) for j in n_jobs_vals
    ]

    fig.legend(
        handles=legend_handles,
        title="rust_backend=True (n_jobs)",
        ncol=len(n_jobs_vals) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / f"macrobenchmark_njobs_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")

    _generate_speedup_plot(df_all, tag)
    _generate_stage_plot(df_all, tag)


def _generate_speedup_plot(df_all: pd.DataFrame, tag: str) -> None:
    """Standalone latency plot: time (ms) with sklearn baseline + rust n_jobs.

    Same bars/arrows as the latency column of ``macrobenchmark_njobs_*.png``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rust = df_all[df_all["version"] == "rust"]
    sklearn1 = df_all[
        (df_all["version"] == "sklearn") & (df_all["n_jobs"] == 1)
    ]
    if rust.empty or sklearn1.empty:
        print("Missing rust/sklearn rows; skipping time plot.")
        return

    n_jobs_vals = sorted(rust["n_jobs"].unique())
    word_vals = sorted(rust["n_unique_words"].unique())
    length_vals = [
        length for length in LENGTH_VALS if length in set(rust["dataset_length"])
    ]
    if not length_vals:
        length_vals = sorted(rust["dataset_length"].unique())

    njobs_colors = {
        1: "#C4C4C4",
        2: "#525252",
        4: "#f47e7e",
        8: "#620000",
        24: "#c1272d",
    }
    sklearn_color = "#444444"

    fig, axes = plt.subplots(
        len(length_vals),
        1,
        figsize=(8.5, 3.4 * len(length_vals)),
        squeeze=False,
    )

    n_bars = len(n_jobs_vals) + 1
    total_width = 0.7
    bar_width = total_width / n_bars
    offsets = np.linspace(
        -(total_width - bar_width) / 2, (total_width - bar_width) / 2, n_bars
    )

    for i, length in enumerate(length_vals):
        ax = axes[i][0]
        sub = rust[rust["dataset_length"] == length]
        sub_sklearn = sklearn1[sklearn1["dataset_length"] == length]
        x = np.arange(len(word_vals))

        sklearn_vals = []
        for w in word_vals:
            row = sub_sklearn[sub_sklearn.n_unique_words == w]
            sklearn_vals.append(
                float(row["time_ms"].iloc[0]) if not row.empty else float("nan")
            )
        ax.bar(x + offsets[0], sklearn_vals, bar_width, color=sklearn_color)

        for offset, n_jobs in zip(offsets[1:], n_jobs_vals):
            vals = []
            for w in word_vals:
                row = sub[(sub.n_unique_words == w) & (sub.n_jobs == n_jobs)]
                vals.append(
                    float(row["time_ms"].iloc[0]) if not row.empty else float("nan")
                )
            ax.bar(
                x + offset, vals, bar_width, color=njobs_colors.get(n_jobs, "#888")
            )

            if n_jobs in (1, 24):
                origin_nudge = {1: -1, 24: 0}[n_jobs] * 0.015
                for xi_sklearn, xi_rust, t, st in zip(
                    x + offsets[0], x + offset, vals, sklearn_vals
                ):
                    if not (np.isfinite(t) and np.isfinite(st) and t > 0):
                        continue
                    origin_y = st * (1 + origin_nudge)
                    ax.annotate(
                        "",
                        xy=(xi_rust, t),
                        xytext=(xi_sklearn, origin_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            lw=0.8,
                            connectionstyle="angle,angleA=0,angleB=90",
                        ),
                    )
                    ax.annotate(
                        f"{st / t:.1f}×",
                        xy=(xi_rust, (origin_y + t) / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(word_vals)
        ax.set_xlabel("n_unique_words")
        ax.set_ylabel("time (ms)")
        ax.set_title(f"n_rows = {length:,} (latency)")
        ax.grid(True, axis="y", ls=":", alpha=0.5)

    legend_handles = [
        Patch(color=sklearn_color, label="stratum (rust_backend=False)")
    ]
    legend_handles += [
        Patch(color=njobs_colors.get(j, "#888"), label=str(j)) for j in n_jobs_vals
    ]
    fig.legend(
        handles=legend_handles,
        title="rust_backend=True (n_jobs)",
        ncol=len(n_jobs_vals) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / f"macrobenchmark_speedup_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved time plot to {out}")


def _generate_stage_plot(df_all: pd.DataFrame, tag: str) -> None:
    """Stacked bars of rust stage timings when ``stages_ms_json`` is populated."""
    import matplotlib.pyplot as plt

    if "stages_ms_json" not in df_all.columns:
        return

    rust = df_all[df_all["version"] == "rust"].copy()
    parsed = []
    for _, row in rust.iterrows():
        raw = row.get("stages_ms_json", "{}")
        try:
            stages = json.loads(raw) if isinstance(raw, str) else {}
        except (TypeError, json.JSONDecodeError):
            stages = {}
        if not stages:
            continue
        parsed.append((row, stages))
    if not parsed:
        print("No stage timings found; skipping stage plot.")
        return

    stage_names: list[str] = []
    for s in _STAGE_ORDER:
        if any(s in stages for _, stages in parsed):
            stage_names.append(s)
    extras = sorted(
        {
            name
            for _, stages in parsed
            for name in stages
            if name not in stage_names
        }
    )
    stage_names.extend(extras)

    length_vals = sorted({int(row["dataset_length"]) for row, _ in parsed})
    word_vals = sorted({int(row["n_unique_words"]) for row, _ in parsed})
    n_jobs_vals = sorted({int(row["n_jobs"]) for row, _ in parsed})

    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % 20) for i, name in enumerate(stage_names)}

    fig, axes = plt.subplots(
        len(length_vals),
        len(word_vals),
        figsize=(4.5 * len(word_vals), 3.2 * len(length_vals)),
        squeeze=False,
        sharey="row",
    )

    for i, length in enumerate(length_vals):
        for j, words in enumerate(word_vals):
            ax = axes[i][j]
            x = np.arange(len(n_jobs_vals))
            bottoms = np.zeros(len(n_jobs_vals))
            for stage in stage_names:
                vals = []
                for n_jobs in n_jobs_vals:
                    ms = 0.0
                    for row, stages in parsed:
                        if (
                            int(row["dataset_length"]) == length
                            and int(row["n_unique_words"]) == words
                            and int(row["n_jobs"]) == n_jobs
                        ):
                            ms = float(stages.get(stage, 0.0))
                            break
                    vals.append(ms)
                ax.bar(
                    x,
                    vals,
                    bottom=bottoms,
                    color=colors[stage],
                    label=stage if i == 0 and j == 0 else None,
                    width=0.65,
                )
                bottoms = bottoms + np.asarray(vals, dtype=float)

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in n_jobs_vals])
            ax.set_xlabel("n_jobs")
            if j == 0:
                ax.set_ylabel("stage time (ms)")
            ax.set_title(f"n_rows={length:,}, words={words}")
            ax.grid(True, axis="y", ls=":", alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="stage",
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = RESULTS_DIR / f"macrobenchmark_stages_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved stage plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-unique-words", type=int, default=1000)
    parser.add_argument("--dataset-length", type=int, default=10000)
    parser.add_argument("--version", type=str, choices=["rust", "sklearn"])
    parser.add_argument("--tag", type=str, default="tfidf")
    parser.add_argument(
        "--profile-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After timed rust runs, enable debug_timing for one pass and "
        "harvest stage lines (default: on). Use --no-profile-stages to skip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing CSV rows and re-run the full grid.",
    )
    args = parser.parse_args()

    if args.worker:
        run_worker(
            args.n_jobs,
            args.n_unique_words,
            args.dataset_length,
            args.version,
            args.profile_stages,
        )
    else:
        run_master(args.tag, args.profile_stages, args.force)
