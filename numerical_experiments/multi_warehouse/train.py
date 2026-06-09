"""
Train demand prediction model and evaluate two-stage allocation cost.

Training loss: MSE on demand predictions (fit demand first).
Evaluation:    realized cost with Stage-2 QP for optimal shipments.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import generate_data, evaluate_cost_np, unpack_task, task_dim


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_demand_model(
    model,
    X_train: torch.Tensor,   # (n, x_dim)
    D_train: torch.Tensor,   # (n, T, W)
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 2000,
    patience: int = 20,
    check_every: int = 10,
    val_fraction: float = 0.2,
    verbose: bool = True,
):
    """Train with MSE on demand predictions, patience-based early stopping."""
    import copy

    n_val = max(1, int(len(X_train) * val_fraction))
    X_tr, D_tr = X_train[:-n_val], D_train[:-n_val]
    X_val, D_val = X_train[-n_val:], D_train[-n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tr, D_tr), batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for x_b, d_b in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_b), d_b)
            loss.backward()
            opt.step()

        if (epoch + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(X_val), D_val).item()
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if verbose:
                print(f"  epoch {epoch+1:4d}  val_mse={val_loss:.4f}  no_improve={no_improve}/{patience}")
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return epoch + 1


@torch.no_grad()
def predict_orders(model, X: torch.Tensor) -> np.ndarray:
    """Return y = D_hat as numpy array (n, T, W)."""
    model.eval()
    return model(X).numpy()


def evaluate(model, X: torch.Tensor, data: dict, task: np.ndarray = None) -> float:
    """Predict orders from model, run Stage-2 QP, return mean cost."""
    y = predict_orders(model, X)
    return evaluate_cost_np(y, data, task)


def oracle_cost(data: dict, task: np.ndarray = None) -> float:
    """
    Oracle: knows true demand, orders exactly D[t,w] every period.
    task=None uses per-sequence tasks from data["task_vecs"].
    """
    y_oracle = data["D"].copy()
    return evaluate_cost_np(y_oracle, data, task)


def naive_cost(data: dict, task: np.ndarray = None) -> float:
    """Naive: order mean demand (scalar, ignores x and task)."""
    mean_d = data["D"].mean()
    y_naive = np.full_like(data["D"], mean_d)
    return evaluate_cost_np(y_naive, data, task)


# ---------------------------------------------------------------------------
# End-to-end (DFL) training — backprop through Stage-2 QP cost
# ---------------------------------------------------------------------------

def train_e2e(
    model,
    X_train: torch.Tensor,        # (n, x_dim)
    D_train: torch.Tensor,        # (n, T, W)
    C_net_train: torch.Tensor,    # (n, T, W, W)
    tasks_train: torch.Tensor,    # (n, task_dim)  — one task per sequence
    W: int,
    forward_fn=None,              # callable(model, x_b, p_b) → y; default ignores p
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 20,
    check_every: int = 10,
    val_fraction: float = 0.2,
    verbose: bool = True,
):
    """
    Train model by minimising the realised two-stage allocation cost.
    Gradients flow back through the Stage-2 QPFunction (qpth).

    forward_fn(model, x_b, p_b) → y_b  allows task-conditioned models.
    Default: model(x_b) ignoring p.
    """
    import copy
    from dfl import e2e_cost
    from torch.utils.data import DataLoader, TensorDataset

    if forward_fn is None:
        forward_fn = lambda m, x, p: m(x)

    n_val = max(1, int(len(X_train) * val_fraction))
    X_tr  = X_train[:-n_val];    X_val  = X_train[-n_val:]
    D_tr  = D_train[:-n_val];    D_val  = D_train[-n_val:]
    C_tr  = C_net_train[:-n_val]; C_val = C_net_train[-n_val:]
    P_tr  = tasks_train[:-n_val]; P_val = tasks_train[-n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(X_tr, D_tr, C_tr, P_tr), batch_size=batch_size, shuffle=True
    )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for x_b, d_b, c_b, p_b in loader:
            opt.zero_grad()
            y_b = forward_fn(model, x_b, p_b)
            loss = e2e_cost(y_b, d_b, c_b, p_b, W)
            loss.backward()
            opt.step()

        if (epoch + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                y_val = forward_fn(model, X_val, P_val)
            val_cost = e2e_cost(y_val, D_val, C_val, P_val, W).item()
            if val_cost < best_val - 1e-4:
                best_val = val_cost
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if verbose:
                print(f"  epoch {epoch+1:4d}  val_cost={val_cost:.4f}  no_improve={no_improve}/{patience}")
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return epoch + 1


# ---------------------------------------------------------------------------
# Quick end-to-end demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models import TaskAgnosticModel, ConcatModel, FiLMModel
    from data import task_dim

    seed_everything(42)

    T, W, x_dim = 4, 3, 8
    n_tasks = 5   # task pool size
    td = task_dim(W)

    # Train: 5 tasks, 200 sequences each assigned one task uniformly
    data_train = generate_data(n_tasks=n_tasks, n_sequences=200,
                               T=T, W=W, x_dim=x_dim, seed=42)
    # Test: same task pool and demand weights, new sequences
    data_test  = generate_data(n_tasks=n_tasks, n_sequences=100,
                               T=T, W=W, x_dim=x_dim, seed=77,
                               W_true=data_train["W_true"],
                               tasks=data_train["tasks"])

    def to_t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    X_tr  = to_t(data_train["X"])
    D_tr  = to_t(data_train["D"])
    C_tr  = to_t(data_train["C_net"])
    P_tr  = to_t(data_train["task_vecs"])   # (n_train, td)

    X_te  = to_t(data_test["X"])
    D_te  = to_t(data_test["D"])
    C_te  = to_t(data_test["C_net"])
    P_te  = to_t(data_test["task_vecs"])

    EPOCHS  = 300
    VERBOSE = True

    # Helper: evaluate a trained model on test data
    def eval_model(m, fwd):
        m.eval()
        with torch.no_grad():
            y = fwd(m, X_te, P_te)
        return evaluate_cost_np(y.numpy(), data_test)

    def evaluate_cost_np_local(y_np, data):
        from data import evaluate_cost_np as _ev
        return _ev(y_np, data)

    # ---- 1. Task-agnostic E2E  (ignores p) ----------------------------------
    print("\n=== Task-Agnostic E2E (ignores task) ===")
    m_agn = TaskAgnosticModel(x_dim, T, W)
    fwd_agn = lambda m, x, p: m(x)
    train_e2e(m_agn, X_tr, D_tr, C_tr, P_tr, W,
              forward_fn=fwd_agn, max_epochs=EPOCHS, verbose=VERBOSE)

    # ---- 2. Concat E2E  ([x, p] → y) ----------------------------------------
    print("\n=== Concat E2E (flat task conditioning) ===")
    m_cat = ConcatModel(x_dim, td, T, W)
    fwd_cat = lambda m, x, p: m(x, p)
    train_e2e(m_cat, X_tr, D_tr, C_tr, P_tr, W,
              forward_fn=fwd_cat, max_epochs=EPOCHS, verbose=VERBOSE)

    # ---- 3. FiLM E2E  (paper approach: f(x;p) = t(r(x);p)) -----------------
    print("\n=== FiLM E2E (paper: f(x;p) = t(r(x);p)) ===")
    m_film = FiLMModel(x_dim, td, T, W)
    fwd_film = lambda m, x, p: m(x, p)
    train_e2e(m_film, X_tr, D_tr, C_tr, P_tr, W,
              forward_fn=fwd_film, max_epochs=EPOCHS, verbose=VERBOSE)

    # ---- Results ------------------------------------------------------------
    from data import evaluate_cost_np as _ev_np

    def ev(m, fwd):
        m.eval()
        with torch.no_grad():
            y = fwd(m, X_te, P_te)
        return _ev_np(y.numpy(), data_test)

    print(f"\n{'='*50}")
    print(f"  Oracle (per-seq task-optimal) : {oracle_cost(data_test):.4f}")
    print(f"  Naive  (mean demand, ignore x): {naive_cost(data_test):.4f}")
    print(f"  Task-Agnostic E2E             : {ev(m_agn,  fwd_agn):.4f}")
    print(f"  Concat E2E                    : {ev(m_cat,  fwd_cat):.4f}")
    print(f"  FiLM E2E  (paper)             : {ev(m_film, fwd_film):.4f}")
