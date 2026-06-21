"""
Multi-warehouse allocation experiment.

Compares four approaches on cross-task two-stage DFL:
  1. Task-Agnostic E2E   — e2e cost, ignores task p
  2. Concat E2E          — e2e cost, [x, p] -> y  (flat baseline)
  3. FiLM E2E            — e2e cost, f(x;p) = t(r(x);p)  (paper approach)
  4. SAA                 — predict mu(x) via MSE model, draw demand/cost
                           scenarios, optimise y per sequence via QP

Baselines: Oracle (true demand), Naive (mean demand).

Usage:
    python experiment.py                     # default config
    python experiment.py --n_tasks 10 --n_train 500
"""

import argparse
import time
import torch
import numpy as np

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
# Config
# ---------------------------------------------------------------------------

def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--T",        type=int, default=4)
    p.add_argument("--W",        type=int, default=3)
    p.add_argument("--x_dim",    type=int, default=8)
    p.add_argument("--n_tasks",  type=int, default=5)
    p.add_argument("--n_train",  type=int, default=200)
    p.add_argument("--n_test",   type=int, default=100)
    p.add_argument("--seed",     type=int, default=42)
    # training
    p.add_argument("--epochs_agnostic", type=int, default=300)
    p.add_argument("--epochs_concat",   type=int, default=300)
    p.add_argument("--epochs_film",     type=int, default=1000)
    p.add_argument("--patience",        type=int, default=30)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--batch_size",      type=int, default=32)
    # SAA
    p.add_argument("--saa_samples",     type=int, default=20)
    p.add_argument("--saa_epochs",      type=int, default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)


def eval_model(model, fwd_fn, X_te, P_te, data_test):
    model.eval()
    with torch.no_grad():
        y = fwd_fn(model, X_te, P_te)
    return evaluate_cost_np(y.numpy(), data_test)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg):
    seed_everything(cfg.seed)
    T, W, x_dim = cfg.T, cfg.W, cfg.x_dim
    td = task_dim(W)

    # --- data ----------------------------------------------------------------
    section("Generating data")
    data_train = generate_data(
        n_tasks=cfg.n_tasks, n_sequences=cfg.n_train,
        T=T, W=W, x_dim=x_dim, seed=cfg.seed,
    )
    data_test = generate_data(
        n_tasks=cfg.n_tasks, n_sequences=cfg.n_test,
        T=T, W=W, x_dim=x_dim, seed=cfg.seed + 100,
        W_true=data_train["W_true"],
        tasks=data_train["tasks"],
    )
    print(f"  Train: {cfg.n_train} sequences, {cfg.n_tasks} tasks")
    print(f"  Test : {cfg.n_test} sequences (same task pool)")

    X_tr = to_tensor(data_train["X"]);  D_tr = to_tensor(data_train["D"])
    C_tr = to_tensor(data_train["C_net"]); P_tr = to_tensor(data_train["task_vecs"])

    X_te = to_tensor(data_test["X"]);   P_te = to_tensor(data_test["task_vecs"])

    results = {}

    # --- 1. Task-Agnostic E2E ------------------------------------------------
    section("1 / 4  Task-Agnostic E2E  (ignores task p)")
    m_agn = TaskAgnosticModel(x_dim, T, W)
    fwd_agn = lambda m, x, p: m(x)
    t0 = time.time()
    ep = train_e2e(m_agn, X_tr, D_tr, C_tr, P_tr, W,
                   forward_fn=fwd_agn,
                   max_epochs=cfg.epochs_agnostic, patience=cfg.patience,
                   lr=cfg.lr, batch_size=cfg.batch_size, verbose=True)
    results["Agnostic E2E"] = eval_model(m_agn, fwd_agn, X_te, P_te, data_test)
    print(f"  -> converged in {ep} epochs  ({time.time()-t0:.0f}s)")

    # --- 2. Concat E2E -------------------------------------------------------
    section("2 / 4  Concat E2E  ([x, p] -> y)")
    m_cat = ConcatModel(x_dim, td, T, W)
    fwd_cat = lambda m, x, p: m(x, p)
    t0 = time.time()
    ep = train_e2e(m_cat, X_tr, D_tr, C_tr, P_tr, W,
                   forward_fn=fwd_cat,
                   max_epochs=cfg.epochs_concat, patience=cfg.patience,
                   lr=cfg.lr, batch_size=cfg.batch_size, verbose=True)
    results["Concat E2E"] = eval_model(m_cat, fwd_cat, X_te, P_te, data_test)
    print(f"  -> converged in {ep} epochs  ({time.time()-t0:.0f}s)")

    # --- 3. FiLM E2E  (paper) ------------------------------------------------
    section("3 / 4  FiLM E2E  (paper: f(x;p) = t(r(x);p))")
    m_film = FiLMModel(x_dim, td, T, W)
    fwd_film = lambda m, x, p: m(x, p)
    t0 = time.time()
    ep = train_e2e(m_film, X_tr, D_tr, C_tr, P_tr, W,
                   forward_fn=fwd_film,
                   max_epochs=cfg.epochs_film, patience=cfg.patience,
                   lr=cfg.lr, batch_size=cfg.batch_size, verbose=True)
    results["FiLM E2E"] = eval_model(m_film, fwd_film, X_te, P_te, data_test)
    print(f"  -> converged in {ep} epochs  ({time.time()-t0:.0f}s)")

    # --- 4. SAA --------------------------------------------------------------
    section("4 / 4  SAA  (predict mu(x), optimise y over scenarios)")
    m_pred = TaskAgnosticModel(x_dim, T, W)
    train_demand_model(m_pred, X_tr, D_tr, max_epochs=500, verbose=False)
    t0 = time.time()
    results["SAA"] = saa_demand_model(
        m_pred, X_tr, D_tr, data_test, W,
        n_saa_samples=cfg.saa_samples,
        inner_epochs=cfg.saa_epochs,
        verbose=True,
    )
    print(f"  -> done  ({time.time()-t0:.0f}s)")

    # --- baselines -----------------------------------------------------------
    results["Oracle"] = oracle_cost(data_test)
    results["Naive"]  = naive_cost(data_test)

    # --- print summary -------------------------------------------------------
    section("Results")
    order = ["Oracle", "FiLM E2E", "Concat E2E", "SAA", "Agnostic E2E", "Naive"]
    for name in order:
        print(f"  {name:<22s}: {results[name]:.4f}")


if __name__ == "__main__":
    run(get_cfg())
