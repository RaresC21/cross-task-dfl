"""
Multi-period multi-warehouse allocation — data generation.

Two-stage structure
-------------------
Stage 1 (model):  observe x and task p → commit to orders y[w, t] for all
                  warehouses and periods before any demand is seen.

Stage 2 (QP):     each period, after demand D[t] and network costs C_net[t]
                  are revealed, solve a QP for the optimal shipments
                  s[t, w→w'] given the current inventory positions.

Inventory dynamics (per period t, per warehouse w)
---------------------------------------------------
    I_pre[w]   = I[w, t-1] + y[w, t]                 (receive order)
    s*[t]      = argmin QP(I_pre, D[t], C_net[t], h, b)
    net[w]     = Σ_{w'} s[w'→w] - Σ_{w'} s[w→w']    (net inflow)
    I[w, t]    = I_pre[w] + net[w] - D[t, w]
    surplus[w] = max(I[w,t], 0)
    backlog[w] = max(-I[w,t], 0)
    I[w, t]    = surplus[w] - backlog[w]               (carry forward)

Task parameters  p = [c, h, b, mu_ship]  of length W*(W+2)
-----------------------------------------------------------
    c[w]         ordering cost per unit, shape (W,)
    h[w]         holding cost (quadratic), shape (W,)
    b[w]         backlog cost (quadratic), shape (W,)
    mu_ship[k]   mean shipment cost for directed edge k, shape (W*(W-1),)

Per-period network costs  C_net[seq, t, w, w']
----------------------------------------------
    Realized shipment costs, drawn each period from a log-normal centered on
    mu_ship.  These are revealed at Stage 2 alongside D[t].

Decision variables (model output)
----------------------------------
    y: (n_sequences, T, W)   orders, one per warehouse per period
    — shipments are not a model output; they are always solved via the Stage-2 QP.

Shipment edge ordering (used throughout)
-----------------------------------------
    Row-major over (w, w') with w ≠ w'.
    Edge index k = w*(W-1) + (w' if w' < w else w'-1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

def task_dim(W: int) -> int:
    """Length of the task parameter vector [c, h, b, mu_ship]."""
    return W * (W + 2)          # c(W) + h(W) + b(W) + mu_ship(W*(W-1))


def order_dim(T: int, W: int) -> int:
    """Length of the flat order decision vector y (model output)."""
    return T * W


def edge_index(w: int, w_prime: int, W: int) -> int:
    """Directed edge (w → w') → flat index k, for w ≠ w'."""
    assert w != w_prime
    return w * (W - 1) + (w_prime if w_prime < w else w_prime - 1)


def unpack_task(task: np.ndarray, W: int):
    """Split flat task vector into (c, h, b, mu_ship)."""
    c      = task[:W]
    h      = task[W:2*W]
    b      = task[2*W:3*W]
    mu_ship = task[3*W:]        # length W*(W-1)
    return c, h, b, mu_ship


def mu_ship_to_matrix(mu_ship: np.ndarray, W: int) -> np.ndarray:
    """Flat edge costs → (W, W) matrix with zeros on diagonal."""
    C = np.zeros((W, W))
    k = 0
    for w in range(W):
        for wp in range(W):
            if w != wp:
                C[w, wp] = mu_ship[k]
                k += 1
    return C


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    n_tasks: int,
    n_sequences: int,
    T: int,
    W: int,
    x_dim: int,
    seed: int = 42,
    W_true: np.ndarray = None,          # (x_dim, W) — share across splits
    tasks: np.ndarray = None,           # (n_tasks, task_dim) — share across splits
    demand_noise_std: float = 1.0,
    net_cost_noise_std: float = 0.2,    # relative log-noise on C_net around mu_ship
) -> dict:
    """
    Generate cross-task multi-warehouse two-stage allocation data.

    Each sequence is assigned one task (drawn uniformly from n_tasks tasks).
    C_net for sequence i is drawn from a log-normal centred on that sequence's
    mu_ship, so the network cost distribution is task-specific.

    Returns
    -------
    dict with:
      X          : (n_sequences, x_dim)       context features
      D          : (n_sequences, T, W)         realized demands
      C_net      : (n_sequences, T, W, W)      per-period realized network costs
      tasks      : (n_tasks, task_dim(W))      task parameter pool [c, h, b, mu_ship]
      task_vecs  : (n_sequences, task_dim)     task assigned to each sequence
      task_idx   : (n_sequences,)              index into tasks pool
      W_true     : (x_dim, W)
      T, W, x_dim : scalars
    """
    rng = np.random.default_rng(seed)

    # --- demand ground-truth weights (shared across train/test splits) ---
    if W_true is None:
        W_true = rng.standard_normal((x_dim, W))

    X = rng.standard_normal((n_sequences, x_dim))
    mean_D = X @ W_true                                        # (n_seq, W)
    noise  = demand_noise_std * rng.standard_normal((n_sequences, T, W))
    D = np.maximum(0.0, mean_D[:, np.newaxis, :] + noise)     # (n_seq, T, W)

    # --- task pool (shared across train/test splits if provided) ---
    if tasks is None:
        n_edges = W * (W - 1)
        c_pool       = rng.uniform(0.1, 1.0, size=(n_tasks, W))
        h_pool       = rng.uniform(0.5, 2.0, size=(n_tasks, W))
        b_pool       = rng.uniform(2.0, 8.0, size=(n_tasks, W))
        mu_ship_pool = rng.uniform(0.1, 1.5, size=(n_tasks, n_edges))
        tasks = np.concatenate([c_pool, h_pool, b_pool, mu_ship_pool], axis=1)

    # --- assign one task per sequence ---
    task_idx  = rng.integers(0, n_tasks, size=n_sequences)    # (n_seq,)
    task_vecs = tasks[task_idx]                                # (n_seq, task_dim)

    # --- per-period network costs from each sequence's assigned mu_ship ---
    C_net = np.zeros((n_sequences, T, W, W))
    edges = [(wf, wt) for wf in range(W) for wt in range(W) if wf != wt]
    for i in range(n_sequences):
        _, _, _, mu_ship_i = unpack_task(task_vecs[i], W)
        mu_mat = mu_ship_to_matrix(mu_ship_i, W)              # (W, W)
        log_noise = rng.normal(0.0, net_cost_noise_std, (T, W, W))
        C_net[i] = mu_mat[np.newaxis, :, :] * np.exp(log_noise)
        for w in range(W):
            C_net[i, :, w, w] = 0.0

    return {
        "X":         X,
        "D":         D,
        "C_net":     C_net,        # (n_seq, T, W, W), diagonal=0
        "tasks":     tasks,        # (n_tasks, task_dim)
        "task_vecs": task_vecs,    # (n_seq, task_dim)
        "task_idx":  task_idx,     # (n_seq,)
        "W_true":    W_true,
        "T":         T,
        "W":         W,
        "x_dim":     x_dim,
    }


# ---------------------------------------------------------------------------
# Stage-2 QP: optimal shipments given inventory positions and realized demand
# ---------------------------------------------------------------------------

def solve_stage2_np(
    I_pre: np.ndarray,   # (W,) inventory before shipments
    D_t: np.ndarray,     # (W,) realized demand this period
    C_net_t: np.ndarray, # (W, W) realized shipment costs, diagonal=0
    h: np.ndarray,       # (W,) holding cost coefficients
    b: np.ndarray,       # (W,) backlog cost coefficients
) -> np.ndarray:
    """
    Solve Stage-2 QP for one period using cvxpy (for evaluation/oracle use).

    Minimises:
        Σ_w  [0.5 h_w S_w^2  +  0.5 b_w B_w^2]  +  Σ_{w≠w'} C_net[w,w'] s[w,w']

    subject to:
        S_w - B_w = I_pre_w - D_w + Σ_{w'} s[w'→w] - Σ_{w'} s[w→w']   ∀w
        s[w,w'] ≥ 0,   S_w ≥ 0,   B_w ≥ 0

    Returns s: (W, W) optimal shipment matrix (diagonal=0).
    """
    import cvxpy as cp

    W = len(I_pre)
    s = cp.Variable((W, W), nonneg=True)
    S = cp.Variable(W, nonneg=True)     # surplus
    B = cp.Variable(W, nonneg=True)     # backlog

    # no self-shipment
    constraints = [cp.diag(s) == 0]

    # inventory balance
    for w in range(W):
        net_in = cp.sum(s[:, w]) - cp.sum(s[w, :])
        constraints.append(S[w] - B[w] == I_pre[w] + net_in - D_t[w])

    cost = (0.5 * cp.sum(cp.multiply(h, cp.square(S))) +
            0.5 * cp.sum(cp.multiply(b, cp.square(B))) +
            cp.sum(cp.multiply(C_net_t, s)))

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    s_val = s.value if s.value is not None else np.zeros((W, W))
    return np.maximum(s_val, 0.0)


# ---------------------------------------------------------------------------
# Full cost evaluation (numpy, calls Stage-2 QP each period)
# ---------------------------------------------------------------------------

def evaluate_cost_np(
    y: np.ndarray,        # (n_sequences, T, W)  first-stage orders
    data: dict,           # output of generate_data
    task: np.ndarray = None,   # (task_dim,) single task OR None → use data["task_vecs"]
    solve_stage2=None,    # callable or None → uses solve_stage2_np
) -> float:
    """
    Simulate all sequences and periods; return mean total cost per sequence.
    If task is None, uses data["task_vecs"] (one task per sequence).
    """
    if solve_stage2 is None:
        solve_stage2 = solve_stage2_np

    n_seq, T, W = data["D"].shape
    D     = data["D"]
    C_net = data["C_net"]

    # Support both single-task and per-sequence-task modes
    if task is not None:
        task_vecs = np.tile(task, (n_seq, 1))
    else:
        task_vecs = data["task_vecs"]

    total = 0.0
    for i in range(n_seq):
        c, h, b, _ = unpack_task(task_vecs[i], W)
        I = np.zeros(W)
        seq_cost = 0.0
        for t in range(T):
            I_pre = I + y[i, t]
            seq_cost += (c * y[i, t]).sum()

            s_opt = solve_stage2(I_pre, D[i, t], C_net[i, t], h, b)

            net     = s_opt.sum(axis=0) - s_opt.sum(axis=1)
            I       = I_pre + net - D[i, t]
            surplus = np.maximum(I, 0.0)
            backlog = np.maximum(-I, 0.0)
            I       = surplus - backlog

            seq_cost += (0.5 * h * surplus**2).sum()
            seq_cost += (0.5 * b * backlog**2).sum()
            seq_cost += (C_net[i, t] * s_opt).sum()

        total += seq_cost

    return total / n_seq


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    T, W, x_dim = 4, 3, 8
    data = generate_data(n_tasks=5, n_sequences=20, T=T, W=W, x_dim=x_dim)

    print("X shape      :", data["X"].shape)
    print("D shape      :", data["D"].shape)
    print("C_net shape  :", data["C_net"].shape)
    print("tasks shape  :", data["tasks"].shape)
    print(f"task_dim(W={W}): {task_dim(W)}")
    print(f"order_dim    : {order_dim(T, W)}")
    print(f"mean demand  : {data['D'].mean():.3f}")
    print(f"C_net range  : [{data['C_net'][data['C_net']>0].min():.3f}, "
          f"{data['C_net'].max():.3f}]")

    # evaluate cost of zero orders (all backlog) for task 0
    y_zero = np.zeros((20, T, W))
    cost = evaluate_cost_np(y_zero, data, data["tasks"][0])
    print(f"\ncost (zero orders, task 0): {cost:.4f}")

    # evaluate cost of generous orders
    y_good = np.ones((20, T, W)) * 3.0
    cost2 = evaluate_cost_np(y_good, data, data["tasks"][0])
    print(f"cost (order=3,  task 0):   {cost2:.4f}")
