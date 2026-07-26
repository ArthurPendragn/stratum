import argparse
import gc
import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd

# Define parameters
LENGTH_VALS = [10_000, 100_000, 1_000_000]
N_JOBS_VALS = [1, 2, 4, 8, 24]
WORD_VALS = [1000, 10000]

def make_series(n_rows, seed, vocab_size, avg_words=8, words_len_range=(3, 10)) -> pd.Series:
    rng = np.random.default_rng(seed)
    
    # Create a random lowercase word from ascii characters
    def rand_word():
        size = rng.integers(words_len_range[0], words_len_range[1])
        return ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), size=size))

    # Build a vocabulary of unique words
    vocab = [rand_word() for _ in range(vocab_size)]

    # Randomly generate number of words (around avg_words) in each row
    n_per_row = np.maximum(1, rng.poisson(avg_words, size=n_rows))

    rows = []
    for k in n_per_row:
        idx = rng.integers(0, vocab_size, size=k)
        rows.append(' '.join(vocab[i] for i in idx))

    return pd.Series(rows, name="text")

def run_worker(n_jobs, n_unique_words, dataset_length, version):
    # Set thread environment variable BEFORE importing/initializing stratum/numpy/etc.
    os.environ["SKRUB_RUST_THREADS"] = str(n_jobs)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    import stratum
    from sklearn.feature_extraction.text import HashingVectorizer
    
    # Enable/disable rust backend
    if version == "rust":
        stratum.set_config(rust_backend=True, num_threads=n_jobs)
    else:
        stratum.set_config(rust_backend=False)
        
    # Generate data
    X = make_series(dataset_length, seed=42, vocab_size=n_unique_words)
    
    # Warmup
    enc = HashingVectorizer()
    X_small = X.iloc[:1000]
    _ = enc.transform(X_small)
    gc.collect()
    
    # Measure
    # We do 3 runs and take the minimum time to reduce noise
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = enc.transform(X)
        _shape = res.shape
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        gc.collect()
        
    min_time_ms = min(times)
    print(f"RESULT:{min_time_ms:.9f}")

def run_master(tag):
    csv_path = "benchmark/results/macrobenchmark.csv"
    os.makedirs("benchmark/results", exist_ok=True)
    
    # Initialize or load CSV
    results = []
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            results = df_old.to_dict('records')
            print(f"Loaded {len(results)} existing benchmark results.")
        except Exception:
            pass
            
    # We want to run all combinations:
    # (n_jobs, n_unique_words, dataset_length, version)
    
    # 1. Run Sklearn (once per word count and length)
    for length in LENGTH_VALS:
        for w in WORD_VALS:
            # Check if already done
            exists = any(r['dataset_length'] == length and r['n_unique_words'] == w and r['version'] == 'sklearn' for r in results)
            if exists:
                print(f"Skipping sklearn for length={length}, words={w} (already exists)")
                continue
                
            print(f"Running sklearn for length={length}, words={w}...")
            env = os.environ.copy()
            cmd = [
                sys.executable, __file__,
                "--worker",
                "--n-jobs", "1",
                "--n-unique-words", str(w),
                "--dataset-length", str(length),
                "--version", "sklearn"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                print(f"Error running worker: {proc.stderr}")
                continue
                
            # Find result
            time_ms = None
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT:"):
                    time_ms = float(line.split(":")[1])
                    
            if time_ms is not None:
                results.append({
                    "n_jobs": 1,
                    "n_unique_words": w,
                    "dataset_length": length,
                    "version": "sklearn",
                    "time_ms": time_ms
                })
                # Save immediately
                pd.DataFrame(results).to_csv(csv_path, index=False)
                print(f"Time: {time_ms:.3f} ms")
            else:
                print("Failed to get result from worker:", proc.stdout, proc.stderr)

    # 2. Run Rust for different n_jobs
    for length in LENGTH_VALS:
        for w in WORD_VALS:
            for n_jobs in N_JOBS_VALS:
                # Check if already done
                exists = any(r['dataset_length'] == length and r['n_unique_words'] == w and r['n_jobs'] == n_jobs and r['version'] == 'rust' for r in results)
                if exists:
                    print(f"Skipping rust for length={length}, words={w}, n_jobs={n_jobs} (already exists)")
                    continue
                    
                print(f"Running rust for length={length}, words={w}, n_jobs={n_jobs}...")
                env = os.environ.copy()
                cmd = [
                    sys.executable, __file__,
                    "--worker",
                    "--n-jobs", str(n_jobs),
                    "--n-unique-words", str(w),
                    "--dataset-length", str(length),
                    "--version", "rust"
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if proc.returncode != 0:
                    print(f"Error running worker: {proc.stderr}")
                    continue
                    
                # Find result
                time_ms = None
                for line in proc.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        time_ms = float(line.split(":")[1])
                        
                if time_ms is not None:
                    results.append({
                        "n_jobs": n_jobs,
                        "n_unique_words": w,
                        "dataset_length": length,
                        "version": "rust",
                        "time_ms": time_ms
                    })
                    # Save immediately
                    pd.DataFrame(results).to_csv(csv_path, index=False)
                    print(f"Time: {time_ms:.3f} ms")
                else:
                    print("Failed to get result from worker:", proc.stdout, proc.stderr)

    print("All benchmarks finished! Running plot generation...")
    # Now generate the plots using the user's script
    generate_plots(tag)

def generate_plots(tag):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    df_all = pd.read_csv("benchmark/results/macrobenchmark.csv")
    df = df_all[df_all["version"] == "rust"]
    df_sklearn = df_all[df_all["version"] == "sklearn"]

    UNIT = "ms"
    n_jobs_vals = sorted(df["n_jobs"].unique())
    word_vals = sorted(df["n_unique_words"].unique())
    length_vals = [10_000, 100_000, 1_000_000]

    GRID_COLS = 1
    GRID_ROWS = len(length_vals)

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLS,
        figsize=(8, 3.4 * GRID_ROWS),
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
        ax = axes[i][0]
        sub = df[df["dataset_length"] == length]
        sub_sklearn = df_sklearn[df_sklearn["dataset_length"] == length]
        x = np.arange(len(word_vals))

        sklearn_times = []
        for w in word_vals:
            row = sub_sklearn[sub_sklearn.n_unique_words == w]
            sklearn_times.append(row.time_ms.iloc[0] if not row.empty else float("nan"))
        ax.bar(x + offsets[0], sklearn_times, bar_width, color=sklearn_color)

        for offset, n_jobs in zip(offsets[1:], n_jobs_vals):
            times = []
            for w in word_vals:
                row = sub[(sub.n_unique_words == w) & (sub.n_jobs == n_jobs)]
                times.append(row.time_ms.iloc[0] if not row.empty else float("nan"))
            ax.bar(x + offset, times, bar_width, color=njobs_colors[n_jobs])

            if n_jobs in (1, 24):
                origin_nudge = {1: -1, 24: 0}[n_jobs] * 0.015
                for xi_sklearn, xi_rust, t, st in zip(
                    x + offsets[0], x + offset, times, sklearn_times
                ):
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
        ax.set_title(f"n_rows = {length:,}")
        ax.set_xlabel("n_unique_words")
        ax.set_ylabel(f"time ({UNIT})")
        ax.grid(True, axis="y", ls=":", alpha=0.5)

    legend_handles = [Patch(color=sklearn_color, label="sklearn")]
    legend_handles += [Patch(color=njobs_colors[j], label=str(j)) for j in n_jobs_vals]

    fig.legend(
        handles=legend_handles,
        title="rust (n_jobs)",
        ncol=len(n_jobs_vals) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        f"benchmark/results/macrobenchmark_njobs_{tag}.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved plot to benchmark/results/macrobenchmark_njobs_{tag}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-unique-words", type=int, default=1000)
    parser.add_argument("--dataset-length", type=int, default=10000)
    parser.add_argument("--version", type=str, choices=["rust", "sklearn"])
    parser.add_argument("--tag", type=str, default="hashing")
    args = parser.parse_args()
    
    if args.worker:
        run_worker(args.n_jobs, args.n_unique_words, args.dataset_length, args.version)
    else:
        run_master(args.tag)
