"""Macrobenchmark HashingVectorizer through Stratum.

Both legs use ``stratum.HashingVectorizer`` (the patched adapter):

* ``version=rust``     → ``rust_backend=True``  (Rust kernel)
* ``version=sklearn``  → ``rust_backend=False`` (sklearn fallback)

Thread count is set via ``stratum.set_config(num_threads=...)`` and
``SKRUB_RUST_THREADS`` before imports in each worker subprocess.

Workers report wall time and peak RSS. Pass ``--profile-stages`` to also
enable ``debug_timing`` for one extra transform and harvest stage timings
from stderr (see ``DEBUG_TIMING.md``).
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


def _peak_rss_mb() -> float:
    """Process peak RSS in MiB (Linux VmHWM when available)."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource

    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_stage_timings(text: str) -> dict[str, float]:
    """Parse ``[rust]/[python]`` debug_timing lines into stage → ms."""
    stages: dict[str, float] = {}
    for line in text.splitlines():
        rust = _RUST_STAGE_RE.match(line.strip())
        if rust:
            stages[rust.group("stage")] = float(rust.group("ms"))
            continue
        py = _PYTHON_STAGE_RE.match(line.strip())
        if py:
            stages[py.group("stage")] = float(py.group("sec")) * 1000.0
    return stages


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
    HashingVectorizer = stratum.HashingVectorizer

    X = make_series(dataset_length, seed=42, vocab_size=n_unique_words)

    enc = HashingVectorizer()
    if version == "rust" and not enc._rust_enabled():
        raise RuntimeError(
            "HashingVectorizer Rust path is not enabled; check rust_backend/"
            "allow_patch and that the extension is built."
        )

    X_small = X.iloc[: min(1000, dataset_length)]
    _ = enc.transform(X_small)
    gc.collect()

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = enc.transform(X)
        _ = res.shape
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        del res
        gc.collect()

    if profile_stages and version == "rust":
        # Untimed pass with debug_timing so stage lines hit stderr without
        # biasing the wall-clock measurements above. The master scrapes them.
        stratum.set_config(debug_timing=True)
        print("STAGE_PROFILE_BEGIN", file=sys.stderr, flush=True)
        _ = enc.transform(X)
        print("STAGE_PROFILE_END", file=sys.stderr, flush=True)
        stratum.set_config(debug_timing=False)
        gc.collect()

    payload = {
        "time_ms": min(times),
        "peak_rss_mb": round(_peak_rss_mb(), 3),
        "stages_ms": {},
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
    stages = parse_stage_timings(proc.stderr + "\n" + proc.stdout)
    if stages:
        payload["stages_ms"] = stages
    return payload


def run_master(tag: str, profile_stages: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    if CSV_PATH.exists():
        try:
            results = pd.read_csv(CSV_PATH).to_dict("records")
            print(f"Loaded {len(results)} existing benchmark results.")
        except Exception:
            pass

    def already_done(length: int, words: int, n_jobs: int, version: str) -> bool:
        return any(
            r["dataset_length"] == length
            and r["n_unique_words"] == words
            and r["n_jobs"] == n_jobs
            and r["version"] == version
            for r in results
        )

    def record(length: int, words: int, n_jobs: int, version: str, metrics: dict) -> None:
        row = {
            "n_jobs": n_jobs,
            "n_unique_words": words,
            "dataset_length": length,
            "version": version,
            "time_ms": metrics["time_ms"],
            "peak_rss_mb": metrics["peak_rss_mb"],
            "stages_ms_json": json.dumps(metrics.get("stages_ms") or {}, sort_keys=True),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(CSV_PATH, index=False)
        stages = metrics.get("stages_ms") or {}
        stage_note = ""
        if stages:
            top = ", ".join(f"{k}={v:.1f}ms" for k, v in list(stages.items())[:4])
            stage_note = f"  stages=[{top}{'…' if len(stages) > 4 else ''}]"
        print(
            f"Time: {metrics['time_ms']:.3f} ms  "
            f"peak_rss={metrics['peak_rss_mb']:.1f} MiB{stage_note}"
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

    fig, axes = plt.subplots(
        len(length_vals),
        2,
        figsize=(14, 3.4 * len(length_vals)),
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
        for col, (metric, ylabel) in enumerate(
            (("time_ms", f"time ({unit})"), ("peak_rss_mb", "peak RSS (MiB)"))
        ):
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
            title_metric = "latency" if metric == "time_ms" else "peak RSS"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-unique-words", type=int, default=1000)
    parser.add_argument("--dataset-length", type=int, default=10000)
    parser.add_argument("--version", type=str, choices=["rust", "sklearn"])
    parser.add_argument("--tag", type=str, default="hashing")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="After timed runs, enable debug_timing for one pass and harvest "
        "stage lines from stderr (see DEBUG_TIMING.md).",
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
        run_master(args.tag, args.profile_stages)
