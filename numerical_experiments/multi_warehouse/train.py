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
# SAA (Sample Average Approximation) baseline
# ---------------------------------------------------------------------------

def saa_demand_model(
    demand_model,
    X_train: torch.Tensor,    # (n, x_dim)  — used to estimate residual σ
    D_train: torch.Tensor,    # (n, T, W)
    data_test: dict,
    W: int,
    n_saa_samples: int = 20,  # demand scenarios per sequence
    inner_epochs: int = 50,
    inner_lr: float = 1e-2,
    verbose: bool = False,
) -> float:
    """
    SAA baseline: two-stage predict-then-optimise.

    Step 1 (statistical):  pretrained demand_model predicts mu(x) per sequence.
                           Residual std sigma estimated from training data.
    Step 2 (SAA):          for each test sequence i with task p_i:
                             - draw S demand scenarios D_s ~ N(mu_i, sigma^2), clipped >= 0
                             - draw S network cost scenarios C_s from p_i's mu_ship
                             - optimise y_i = argmin_y (1/S) sum_s e2e_cost(y, D_s, C_s, p_i)
                               via gradient descent through the Stage-2 QPFunction
    Step 3 (evaluation):   apply y_i to the true realised (D_i, C_net_i).

    This uses x (through mu), uses the task (through QP costs), but is NOT
    trained end-to-end — the demand model and decision are decoupled.
    """
    from dfl import e2e_cost

    T    = data_test["T"]
    rng  = np.random.default_rng(0)

    # --- estimate residual noise from training data --------------------------
    demand_model.eval()
    with torch.no_grad():
        mu_train = demand_model(X_train).numpy()   # (n, T, W)
    residuals = D_train.numpy() - mu_train         # (n, T, W)
    sigma = residuals.std(axis=(0, 1))             # (W,) per-warehouse std

    # --- per-sequence SAA optimisation on test set ---------------------------
    n_test = data_test["D"].shape[0]
    X_te   = torch.tensor(data_test["X"],      dtype=torch.float32)
    P_te   = torch.tensor(data_test["task_vecs"], dtype=torch.float32)

    demand_model.eval()
    with torch.no_grad():
        mu_test = demand_model(X_te).numpy()       # (n_test, T, W)

    _, _, _, mu_ship_all = zip(*[
        __import__('data').unpack_task(data_test["task_vecs"][i], W)
        for i in range(n_test)
    ])

    y_decisions = np.zeros((n_test, T, W))

    for i in range(n_test):
        mu_i = mu_test[i]                          # (T, W)
        p_i  = P_te[i].unsqueeze(0)               # (1, task_dim)

        # Draw S demand scenarios
        noise    = rng.standard_normal((n_saa_samples, T, W)) * sigma
        D_saa    = np.maximum(0.0, mu_i[np.newaxis] + noise)   # (S, T, W)

        # Draw S network cost scenarios from task i's mu_ship
        from data import unpack_task, mu_ship_to_matrix
        _, _, _, mu_ship_i = unpack_task(data_test["task_vecs"][i], W)
        mu_mat_i = mu_ship_to_matrix(mu_ship_i, W)             # (W, W)
        log_noise = rng.normal(0, 0.2, (n_saa_samples, T, W, W))
        C_saa    = mu_mat_i[np.newaxis, np.newaxis] * np.exp(log_noise)
        for w in range(W):
            C_saa[:, :, w, w] = 0.0

        D_s = torch.tensor(D_saa, dtype=torch.float32)
        C_s = torch.tensor(C_saa, dtype=torch.float32)
        P_s = p_i.expand(n_saa_samples, -1)       # (S, task_dim)

        # Optimise y_i: shape (T, W), constant across scenarios
        y_i = torch.tensor(mu_i, dtype=torch.float32).requires_grad_(True)
        opt = torch.optim.Adam([y_i], lr=inner_lr)

        for ep in range(inner_epochs):
            opt.zero_grad()
            y_batch = torch.relu(y_i).unsqueeze(0).expand(n_saa_samples, -1, -1)
            loss = e2e_cost(y_batch, D_s, C_s, P_s, W)
            loss.backward()
            opt.step()

        y_decisions[i] = torch.relu(y_i).detach().numpy()
        if verbose and (i % 10 == 0):
            print(f"  seq {i}/{n_test}  saa_cost={loss.item():.4f}")

    return evaluate_cost_np(y_decisions, data_test)


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

