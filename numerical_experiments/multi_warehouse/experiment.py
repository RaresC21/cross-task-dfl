"""
Multi-warehouse allocation experiment — scenario grid.

For each (n_tasks, n_train) cell the script:
  1. Trains all methods on n_train sequences drawn from n_tasks tasks.
  2. Evaluates in-distribution  (new sequences, same task pool).
  3. Evaluates out-of-distribution (new sequences, NEW unseen tasks).
  4. Appends one row to results.csv  (safe to re-run; skips done rows).

Methods
-------
  Agnostic E2E  — e2e DFL, ignores task p
  Concat E2E    — e2e DFL, [x, p] -> y  (flat baseline)
  FiLM E2E      — e2e DFL, f(x;p) = t(r(x);p)  (paper)
  SAA           — predict mu(x), optimise y over demand scenarios  (slow)
  Oracle / Naive baselines

Usage
-----
  python experiment.py                        # full grid, no SAA
  python experiment.py --saa                  # include SAA (slow)
  python experiment.py --n_tasks 5 10 --n_train 200 500 --seeds 0 1 2
  python experiment.py --results results.csv
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from data import generate_data, evaluate_cost_np, task_dim
from models import TaskAgnosticModel, ConcatModel, FiLMModel
from train import (
    seed_everything,
    train_demand_model,
    train_e2e,
    saa_demand_model,
    oracle_cost,
    naive_cost,
)

# ---------------------------------------------------------------------------
# Default grid
# ---------------------------------------------------------------------------

DEFAULT_N_TASKS = [5, 20, 100]
DEFAULT_N_TRAIN = [100, 200, 500]
DEFAULT_SEEDS   = [0]

# Fixed problem dimensions
T     = 4
W     = 3
X_DIM = 8

# Training hyper-params
LR         = 1e-3
BATCH_SIZE = 32
PATIENCE   = 30

EPOCHS_AGNOSTIC = 500
EPOCHS_CONCAT   = 500
EPOCHS_FILM     = 1000

# Test set sizes
N_TEST_IN  = 200   # in-distribution test sequences
N_TEST_OOD = 200   # OOD test sequences (unseen tasks)
N_TASKS_OOD = 10   # how many new OOD tasks to draw

# SAA settings
SAA_SAMPLES = 20
SAA_EPOCHS  = 50

METHODS = ["Agnostic", "Concat", "FiLM"]   # SAA added if --saa flag set


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def csv_columns(methods):
    cols = ["n_tasks", "n_train", "seed"]
    for split in ("in", "ood"):
        for m in methods + ["Oracle", "Naive"]:
            cols.append(f"{split}_{m}")
    return cols


def load_done(path, methods):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((int(row["n_tasks"]), int(row["n_train"]), int(row["seed"])))
    return done


def append_row(path, row, methods):
    cols   = csv_columns(methods)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Single scenario run
# ---------------------------------------------------------------------------

def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32)


def eval_model(model, fwd_fn, X_te, P_te, data_test):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        y = fwd_fn(model, X_te.to(device), P_te.to(device))
    return evaluate_cost_np(y.cpu().numpy(), data_test)


def run_scenario(n_tasks, n_train, seed, run_saa=False, verbose=True):
    """
    Returns dict with keys in_{method} and ood_{method} for each method.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = task_dim(W)

    # --- generate data -------------------------------------------------------
    data_train = generate_data(
        n_tasks=n_tasks, n_sequences=n_train,
        T=T, W=W, x_dim=X_DIM, seed=seed,
    )
    data_in = generate_data(
        n_tasks=n_tasks, n_sequences=N_TEST_IN,
        T=T, W=W, x_dim=X_DIM, seed=seed + 1000,
        W_true=data_train["W_true"],
        tasks=data_train["tasks"],
    )
    # OOD: same W_true (demand structure), but entirely new tasks
    data_ood = generate_data(
        n_tasks=N_TASKS_OOD, n_sequences=N_TEST_OOD,
        T=T, W=W, x_dim=X_DIM, seed=seed + 2000,
        W_true=data_train["W_true"],   # same demand model
        tasks=None,                    # new task pool drawn fresh
    )

    X_tr = to_t(data_train["X"]);  D_tr = to_t(data_train["D"])
    C_tr = to_t(data_train["C_net"]); P_tr = to_t(data_train["task_vecs"])

    X_in  = to_t(data_in["X"]);   P_in  = to_t(data_in["task_vecs"])
    X_ood = to_t(data_ood["X"]);  P_ood = to_t(data_ood["task_vecs"])

    results = {}

    # --- Agnostic E2E --------------------------------------------------------
    if verbose: print("  [Agnostic E2E]")
    m_agn   = TaskAgnosticModel(X_DIM, T, W)
    fwd_agn = lambda m, x, p: m(x)
    train_e2e(m_agn, X_tr, D_tr, C_tr, P_tr, W,
              forward_fn=fwd_agn, max_epochs=EPOCHS_AGNOSTIC,
              patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE, verbose=False,
              device=device)
    results["in_Agnostic"]  = eval_model(m_agn, fwd_agn, X_in,  P_in,  data_in)
    results["ood_Agnostic"] = eval_model(m_agn, fwd_agn, X_ood, P_ood, data_ood)

    # --- Concat E2E ----------------------------------------------------------
    if verbose: print("  [Concat E2E]")
    m_cat   = ConcatModel(X_DIM, td, T, W)
    fwd_cat = lambda m, x, p: m(x, p)
    train_e2e(m_cat, X_tr, D_tr, C_tr, P_tr, W,
              forward_fn=fwd_cat, max_epochs=EPOCHS_CONCAT,
              patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE, verbose=False,
              device=device)
    results["in_Concat"]  = eval_model(m_cat, fwd_cat, X_in,  P_in,  data_in)
    results["ood_Concat"] = eval_model(m_cat, fwd_cat, X_ood, P_ood, data_ood)

    # --- FiLM E2E (paper) ----------------------------------------------------
    if verbose: print("  [FiLM E2E]")
    m_film   = FiLMModel(X_DIM, td, T, W)
    fwd_film = lambda m, x, p: m(x, p)
    train_e2e(m_film, X_tr, D_tr, C_tr, P_tr, W,
              forward_fn=fwd_film, max_epochs=EPOCHS_FILM,
              patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE, verbose=False,
              device=device)
    results["in_FiLM"]  = eval_model(m_film, fwd_film, X_in,  P_in,  data_in)
    results["ood_FiLM"] = eval_model(m_film, fwd_film, X_ood, P_ood, data_ood)

    # --- SAA (optional) ------------------------------------------------------
    if run_saa:
        if verbose: print("  [SAA]")
        m_pred = TaskAgnosticModel(X_DIM, T, W)
        train_demand_model(m_pred, X_tr, D_tr, max_epochs=500, verbose=False,
                           device=device)
        results["in_SAA"]  = saa_demand_model(
            m_pred, X_tr, D_tr, data_in,  W,
            n_saa_samples=SAA_SAMPLES, inner_epochs=SAA_EPOCHS, device=device)
        results["ood_SAA"] = saa_demand_model(
            m_pred, X_tr, D_tr, data_ood, W,
            n_saa_samples=SAA_SAMPLES, inner_epochs=SAA_EPOCHS, device=device)

    # --- baselines -----------------------------------------------------------
    results["in_Oracle"]  = oracle_cost(data_in)
    results["ood_Oracle"] = oracle_cost(data_ood)
    results["in_Naive"]   = naive_cost(data_in)
    results["ood_Naive"]  = naive_cost(data_ood)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks",  type=int, nargs="+", default=DEFAULT_N_TASKS)
    parser.add_argument("--n_train",  type=int, nargs="+", default=DEFAULT_N_TRAIN)
    parser.add_argument("--seeds",    type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--saa",      action="store_true",  help="include SAA (slow)")
    parser.add_argument("--results",  type=str, default="results.csv")
    args = parser.parse_args()

    import torch as _torch
    _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    print(f"Device: {_dev}" + (f" ({_torch.cuda.get_device_name(0)})" if _dev.type == "cuda" else ""))

    methods = METHODS + (["SAA"] if args.saa else [])
    results_path = os.path.join(os.path.dirname(__file__), args.results)
    done = load_done(results_path, methods)

    total   = len(args.n_tasks) * len(args.n_train) * len(args.seeds)
    current = 0

    for n_tasks in args.n_tasks:
        for n_train in args.n_train:
            for seed in args.seeds:
                current += 1
                key = (n_tasks, n_train, seed)

                if key in done:
                    print(f"[{current}/{total}] n_tasks={n_tasks} n_train={n_train} seed={seed}  SKIP (already done)")
                    continue

                print(f"\n[{current}/{total}] n_tasks={n_tasks}  n_train={n_train}  seed={seed}")
                t0 = time.time()

                res = run_scenario(n_tasks, n_train, seed, run_saa=args.saa)

                row = {"n_tasks": n_tasks, "n_train": n_train, "seed": seed}
                row.update(res)
                # fill missing SAA columns with empty if not run
                for split in ("in", "ood"):
                    for m in methods + ["Oracle", "Naive"]:
                        row.setdefault(f"{split}_{m}", "")
                append_row(results_path, row, methods)

                elapsed = time.time() - t0
                print(f"  done in {elapsed:.0f}s")
                print(f"  in-dist:  Oracle={res['in_Oracle']:.2f}  FiLM={res['in_FiLM']:.2f}  Concat={res['in_Concat']:.2f}  Agnostic={res['in_Agnostic']:.2f}")
                print(f"  OOD:      Oracle={res['ood_Oracle']:.2f}  FiLM={res['ood_FiLM']:.2f}  Concat={res['ood_Concat']:.2f}  Agnostic={res['ood_Agnostic']:.2f}")

    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
