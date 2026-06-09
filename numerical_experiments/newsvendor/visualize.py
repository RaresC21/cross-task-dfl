"""
Visualize how model performance varies across the three scenario axes:
  - n_tasks:   number of training tasks
  - n_samples: training set size
  - x_dim:     feature dimensionality

Run: python visualize.py
Produces: figures/*.png  (one per plot)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")
FIG_DIR      = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS      = ["Flat", "Concat", "FiLM", "Hypernetwork", "Structured"]
COLORS      = dict(zip(MODELS, ["#444444", "#2196F3", "#FF5722", "#9C27B0", "#4CAF50"]))
LINESTYLES  = dict(zip(MODELS, ["-", "--", "-.", ":", "-"]))
MARKERS     = dict(zip(MODELS, ["o", "s", "^", "D", "*"]))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("n_tasks","n_samples","x_dim") else int(v))
                         for k, v in row.items()})
    return rows


# ---------------------------------------------------------------------------
# Helper: aggregate rows matching a filter, grouped by key
# ---------------------------------------------------------------------------

def aggregate(rows, filter_fn, group_key):
    """Return dict: group_value -> {model: [costs]}"""
    from collections import defaultdict
    groups = defaultdict(lambda: {m: [] for m in MODELS})
    for r in rows:
        if filter_fn(r):
            for m in MODELS:
                groups[r[group_key]][m].append(r[f"ood_{m}"])
    return {k: {m: np.mean(v) for m, v in mv.items()} for k, mv in sorted(groups.items())}


def plot_effect(agg, title, xlabel, save_name, log_x=False):
    xs = sorted(agg.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in MODELS:
        ys = [agg[x][m] for x in xs]
        ax.plot(xs, ys, label=m, color=COLORS[m],
                linestyle=LINESTYLES[m], marker=MARKERS[m], markersize=6)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("OOD newsvendor cost (lower = better)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 1 — effect of n_tasks  (fix n_samples=1000, x_dim=20)
# ---------------------------------------------------------------------------

def plot_ntasks(rows):
    agg = aggregate(rows,
                    lambda r: r["n_samples"] == 750 and r["x_dim"] == 20,
                    "n_tasks")
    plot_effect(agg,
                title="Effect of number of training tasks\n(n_samples=750, x_dim=20)",
                xlabel="n_tasks (training tasks)",
                save_name="effect_ntasks.png",
                log_x=True)


# ---------------------------------------------------------------------------
# Plot 2 — effect of n_samples  (fix n_tasks=10, x_dim=20)
# ---------------------------------------------------------------------------

def plot_nsamples(rows):
    agg = aggregate(rows,
                    lambda r: r["n_tasks"] == 10 and r["x_dim"] == 20,
                    "n_samples")
    plot_effect(agg,
                title="Effect of training set size\n(n_tasks=10, x_dim=20)",
                xlabel="n_samples (training set size)",
                save_name="effect_nsamples.png",
                log_x=False)


# ---------------------------------------------------------------------------
# Plot 3 — effect of x_dim  (fix n_tasks=10, n_samples=1000)
# ---------------------------------------------------------------------------

def plot_xdim(rows):
    agg = aggregate(rows,
                    lambda r: r["n_tasks"] == 10 and r["n_samples"] == 750,
                    "x_dim")
    plot_effect(agg,
                title="Effect of feature dimensionality\n(n_tasks=10, n_samples=750)",
                xlabel="x_dim (feature dimension)",
                save_name="effect_xdim.png",
                log_x=False)


# ---------------------------------------------------------------------------
# Plot 4 — heatmap: n_tasks × n_samples for a fixed model and x_dim
# Shows how the landscape looks for the best decomposed model (FiLM)
# ---------------------------------------------------------------------------

def plot_heatmap(rows, model="FiLM", x_dim_fixed=20):
    from collections import defaultdict
    data = defaultdict(dict)
    for r in rows:
        if r["x_dim"] == x_dim_fixed:
            data[r["n_tasks"]][r["n_samples"]] = r[f"ood_{model}"]

    n_tasks_vals  = sorted(data.keys())
    n_sample_vals = sorted({ns for v in data.values() for ns in v})
    grid = np.array([[data[nt].get(ns, np.nan) for ns in n_sample_vals]
                     for nt in n_tasks_vals])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(n_sample_vals)))
    ax.set_xticklabels(n_sample_vals, rotation=45)
    ax.set_yticks(range(len(n_tasks_vals)))
    ax.set_yticklabels(n_tasks_vals)
    ax.set_xlabel("n_samples")
    ax.set_ylabel("n_tasks")
    ax.set_title(f"{model} OOD cost — x_dim={x_dim_fixed}\n(darker = lower = better)")
    plt.colorbar(im, ax=ax, label="OOD cost")
    for i in range(len(n_tasks_vals)):
        for j in range(len(n_sample_vals)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > grid[~np.isnan(grid)].mean() else "black")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"heatmap_{model}_xdim{x_dim_fixed}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 5 — in-dist vs OOD gap per model
# bars showing average (ood - in) gap, grouped by n_tasks
# (x_dim=20, n_samples=750 slice)
# ---------------------------------------------------------------------------

def plot_gap(rows, x_dim_fixed=20, n_samples_fixed=750):
    from collections import defaultdict
    groups = defaultdict(lambda: {m: {"in": [], "ood": []} for m in MODELS})
    for r in rows:
        if r["x_dim"] == x_dim_fixed and r["n_samples"] == n_samples_fixed:
            for m in MODELS:
                groups[r["n_tasks"]][m]["in"].append(r[f"in_{m}"])
                groups[r["n_tasks"]][m]["ood"].append(r[f"ood_{m}"])

    n_tasks_vals = sorted(groups.keys())
    x = np.arange(len(n_tasks_vals))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(MODELS):
        gaps = [np.mean(groups[nt][m]["ood"]) - np.mean(groups[nt][m]["in"])
                for nt in n_tasks_vals]
        ax.bar(x + (i - 2) * width, gaps, width, label=m,
               color=COLORS[m], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(nt) for nt in n_tasks_vals])
    ax.set_xlabel("n_tasks")
    ax.set_ylabel("OOD cost − in-dist cost (generalization gap)")
    ax.set_title(f"Generalization gap by model and n_tasks\n(x_dim={x_dim_fixed}, n_samples={n_samples_fixed})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "generalization_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 6 — model ranking across all scenarios (rank 1 = best)
# ---------------------------------------------------------------------------

def plot_rankings(rows):
    from collections import defaultdict
    rank_counts = {m: defaultdict(int) for m in MODELS}
    for r in rows:
        ood_vals = {m: r[f"ood_{m}"] for m in MODELS}
        sorted_models = sorted(ood_vals, key=ood_vals.get)
        for rank, m in enumerate(sorted_models, start=1):
            rank_counts[m][rank] += 1

    total = len(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ranks = list(range(1, len(MODELS) + 1))
    x = np.arange(len(ranks))
    width = 0.15
    for i, m in enumerate(MODELS):
        fracs = [rank_counts[m][r] / total for r in ranks]
        ax.bar(x + (i - 2) * width, fracs, width, label=m,
               color=COLORS[m], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}" for r in ranks])
    ax.set_ylabel("Fraction of scenarios")
    ax.set_title("OOD rank distribution across all scenarios\n(rank 1 = lowest OOD cost)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "rank_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 7 — OOD cost vs n_tasks, n_samples fixed at 750
# One subplot per x_dim so curves don't collapse onto each other
# ---------------------------------------------------------------------------

def plot_ntasks_fixed_nsamples(rows, n_samples_fixed=750):
    from collections import defaultdict

    x_dims = sorted(set(r["x_dim"] for r in rows))
    n_tasks_vals = sorted(set(r["n_tasks"] for r in rows))

    fig, axes = plt.subplots(1, len(x_dims), figsize=(4 * len(x_dims), 4), sharey=False)
    if len(x_dims) == 1:
        axes = [axes]

    for ax, xd in zip(axes, x_dims):
        data = {nt: {m: None for m in MODELS} for nt in n_tasks_vals}
        for r in rows:
            if r["n_samples"] == n_samples_fixed and r["x_dim"] == xd:
                for m in MODELS:
                    data[r["n_tasks"]][m] = r[f"ood_{m}"]

        for m in MODELS:
            ys = [data[nt][m] for nt in n_tasks_vals if data[nt][m] is not None]
            xs = [nt for nt in n_tasks_vals if data[nt][m] is not None]
            if ys:
                ax.plot(xs, ys, label=m, color=COLORS[m],
                        linestyle=LINESTYLES[m], marker=MARKERS[m], markersize=6)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(n_tasks_vals)
        ax.set_xlabel("n_tasks", fontsize=11)
        ax.set_title(f"x_dim = {xd}", fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("OOD newsvendor cost", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(f"OOD cost vs number of training tasks  (n_samples={n_samples_fixed})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"ntasks_nsamples{n_samples_fixed}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load(RESULTS_FILE)
    print(f"Loaded {len(rows)} rows from {RESULTS_FILE}")

    print("\nGenerating plots...")
    plot_ntasks(rows)
    plot_nsamples(rows)
    plot_xdim(rows)
    plot_heatmap(rows, model="FiLM",    x_dim_fixed=20)
    plot_heatmap(rows, model="Flat",    x_dim_fixed=20)
    plot_heatmap(rows, model="Structured", x_dim_fixed=20)
    plot_gap(rows)
    plot_rankings(rows)
    plot_ntasks_fixed_nsamples(rows, n_samples_fixed=750)

    print(f"\nAll figures saved to {FIG_DIR}/")
