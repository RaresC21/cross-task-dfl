"""
Differentiable end-to-end cost for the multi-warehouse two-stage problem.

Stage-2 QP variables:  z = [s_flat  (n_ship),  S  (W),  B  (W)]
  s_flat: directed shipments in row-major order (w_from, w_to), w_from != w_to
  S[w]:   surplus inventory at warehouse w (>= 0)
  B[w]:   backlog at warehouse w (>= 0)

QP for one period:
  min   0.5 h^T S^2  +  0.5 b^T B^2  +  C_net_flat^T s_flat
  s.t.  S[w] - B[w] + net_in[w] = I_pre[w] - D[w]   (Az = b_rhs)
        z >= 0                                         (Gz <= h_ineq)

Gradients w.r.t. y flow through the equality RHS  b_rhs = I_pre - D = I + y[t] - D[t],
and through the carry-forward  I = S* - B*  into the next period.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import numpy as np
from qpth.qp import QPFunction

from data import unpack_task


# ---------------------------------------------------------------------------
# Fixed QP structure (depends on task h, b but not on C_net or I_pre)
# ---------------------------------------------------------------------------

def _edge_order(W: int):
    """Directed edges in row-major order: (w_from, w_to) with w_from != w_to."""
    return [(wf, wt) for wf in range(W) for wt in range(W) if wf != wt]


def build_stage2_matrices(W: int, h: torch.Tensor, b: torch.Tensor, eps: float = 1e-4):
    """
    Return (Q, G, h_ineq, A) as double tensors with leading batch dim of 1.

    Q:      (1, n_z, n_z)   quadratic cost on S and B; eps*I added for SPD
    G:      (1, n_z, n_z)   -I  (nonnegativity)
    h_ineq: (1, n_z)         zeros
    A:      (1,  W,  n_z)   inventory balance
    """
    n_ship = W * (W - 1)
    n_z = n_ship + 2 * W
    edges = _edge_order(W)

    # Q: quadratic cost — only on surplus (h) and backlog (b)
    q_diag = torch.zeros(n_z, dtype=torch.float64)
    for w in range(W):
        q_diag[n_ship + w]     = float(h[w])   # holding
        q_diag[n_ship + W + w] = float(b[w])   # backlog
    Q = torch.diag(q_diag) + eps * torch.eye(n_z, dtype=torch.float64)
    Q = Q.unsqueeze(0)                          # (1, n_z, n_z)

    # G / h_ineq: -I z <= 0  (all variables >= 0)
    G      = -torch.eye(n_z, dtype=torch.float64).unsqueeze(0)  # (1, n_z, n_z)
    h_ineq = torch.zeros(1, n_z, dtype=torch.float64)

    # A: inventory balance
    # For warehouse w: S[w] - B[w] + sum_{w'->w} s_{w'->w} - sum_{w->w'} s_{w->w'} = I_pre[w] - D[w]
    A = torch.zeros(1, W, n_z, dtype=torch.float64)
    for w in range(W):
        A[0, w, n_ship + w]     =  1.0   # S[w]
        A[0, w, n_ship + W + w] = -1.0   # -B[w]
    for k, (wf, wt) in enumerate(edges):
        A[0, wt, k] += 1.0    # inflow  to wt
        A[0, wf, k] -= 1.0    # outflow from wf

    return Q, G, h_ineq, A


# ---------------------------------------------------------------------------
# Differentiable cost (batched over sequences)
# ---------------------------------------------------------------------------

def e2e_cost(
    y: torch.Tensor,         # (batch, T, W)  — model orders, requires_grad
    D: torch.Tensor,         # (batch, T, W)  — realized demands
    C_net: torch.Tensor,     # (batch, T, W, W) — realized network costs
    tasks: torch.Tensor,     # (batch, task_dim) — one task per sequence
    W: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Mean total cost over batch, fully differentiable w.r.t. y via qpth.

    Each sequence has its own task (h, b, c, mu_ship), so Q is built
    per-sample using torch.diag_embed.

    Gradient flow:
      y[:, t, :] → I_pre = I + y_t  →  b_rhs = I_pre - D_t  →  QPFunction
                                    → (S*, B*) → cost_t
                                    → I = S* - B*  → I_pre_{t+1}  → ...
    """
    batch, T, _ = y.shape
    n_ship = W * (W - 1)
    n_z    = n_ship + 2 * W
    edges  = _edge_order(W)

    device = y.device

    # Per-sequence cost parameters
    c_b = tasks[:, :W].float()          # (batch, W) ordering cost
    h_b = tasks[:, W:2*W].float()       # (batch, W) holding cost
    b_b = tasks[:, 2*W:3*W].float()     # (batch, W) backlog cost

    # Q: (batch, n_z, n_z) — quadratic cost on S (h) and B (b), per sequence
    q_diag = torch.zeros(batch, n_z, dtype=torch.float64, device=device)
    q_diag[:, n_ship:n_ship + W] = h_b.double()
    q_diag[:, n_ship + W:]       = b_b.double()
    Q_b = torch.diag_embed(q_diag) + eps * torch.eye(n_z, dtype=torch.float64, device=device)

    # G / h_ineq: nonnegativity, same for all (expand to batch)
    G_b  = (-torch.eye(n_z, dtype=torch.float64, device=device)
             .unsqueeze(0).expand(batch, -1, -1).contiguous())
    hi_b = torch.zeros(batch, n_z, dtype=torch.float64, device=device)

    # A: inventory balance, same structure for all (expand to batch)
    A1   = build_stage2_matrices(W,
                                  torch.ones(W), torch.ones(W), eps=0.0)[3]  # (1,W,n_z)
    A_b  = A1.to(device).expand(batch, -1, -1).contiguous()

    I = y.new_zeros(batch, W)
    total_cost = y.new_zeros(batch)

    for t in range(T):
        y_t   = y[:, t, :]
        I_pre = I + y_t                         # grad flows here

        total_cost = total_cost + (c_b * y_t).sum(dim=1)

        # Linear cost: realized shipment costs for this period
        C_flat = torch.stack(
            [C_net[:, t, wf, wt] for wf, wt in edges], dim=1
        ).float()                               # (batch, n_ship)
        p_vec = torch.cat(
            [C_flat, y.new_zeros(batch, 2 * W)], dim=1
        ).double()                              # (batch, n_z)

        b_rhs = (I_pre - D[:, t, :]).double()  # (batch, W)

        z_opt = QPFunction(verbose=False, check_Q_spd=False)(
            Q_b, p_vec, G_b, hi_b, A_b, b_rhs
        ).float()                               # (batch, n_z)

        S_opt = z_opt[:, n_ship:n_ship + W]
        B_opt = z_opt[:, n_ship + W:]
        s_opt = z_opt[:, :n_ship]

        total_cost = total_cost + (0.5 * h_b * S_opt**2).sum(dim=1)
        total_cost = total_cost + (0.5 * b_b * B_opt**2).sum(dim=1)
        total_cost = total_cost + (C_flat * s_opt).sum(dim=1)

        I = S_opt - B_opt

    return total_cost.mean()
