"""
Models for multi-warehouse two-stage allocation.

All models output orders y: (batch, T, W)  —  relu'd so y >= 0.
Stage-2 QP then finds optimal shipments given realized demand.

Models
------
TaskAgnosticModel  : x           → y   (ignores task p)
ConcatModel        : [x, p]      → y   ("flat" baseline)
FiLMModel          : r(x) ⊙ γ(p) + β(p) → y   (paper approach: f(x;p) = t(r(x);p))
"""

import torch
import torch.nn as nn
from data import task_dim as _task_dim


class TaskAgnosticModel(nn.Module):
    """Linear map x → y.  Ignores task p entirely."""
    def __init__(self, x_dim: int, T: int, W: int):
        super().__init__()
        self.T, self.W = T, W
        self.net = nn.Linear(x_dim, T * W)

    def forward(self, x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor:
        return torch.relu(self.net(x)).reshape(-1, self.T, self.W)


class ConcatModel(nn.Module):
    """
    Flat baseline: concatenate x and p, then linear → y.
    Treats task features identically to context features.
    """
    def __init__(self, x_dim: int, task_dim: int, T: int, W: int, hidden: int = 64):
        super().__init__()
        self.T, self.W = T, W
        self.net = nn.Sequential(
            nn.Linear(x_dim + task_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, T * W),
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(torch.cat([x, p], dim=-1))).reshape(-1, self.T, self.W)


class FiLMModel(nn.Module):
    """
    Paper approach: f(x; p) = t(r(x); p)

    r(x) = W_r x + b_r                      — demand representation
    t(r; p) = diag(γ(p)) r + β(p)           — FiLM modulation by task

    γ(p), β(p) are linear maps from task → (T*W,).
    This separates the x-encoder from the task-conditioner:
    the representation r is shared structure, while γ/β adapt it per task.
    """
    def __init__(self, x_dim: int, task_dim: int, T: int, W: int):
        super().__init__()
        self.T, self.W = T, W
        out_dim = T * W
        self.encoder = nn.Linear(x_dim, out_dim)   # r(x)
        self.gamma    = nn.Linear(task_dim, out_dim)  # scale
        self.beta     = nn.Linear(task_dim, out_dim)  # shift
        # init gamma to 1 so training starts close to pure demand prediction
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        r   = self.encoder(x)                              # (batch, T*W)
        y   = self.gamma(p) * r + self.beta(p)            # FiLM
        return torch.relu(y).reshape(-1, self.T, self.W)


if __name__ == "__main__":
    B, x_dim, T, W = 8, 10, 4, 3
    td = _task_dim(W)
    x = torch.randn(B, x_dim)
    p = torch.randn(B, td)

    for cls, kwargs in [
        (TaskAgnosticModel, dict(x_dim=x_dim, T=T, W=W)),
        (ConcatModel,       dict(x_dim=x_dim, task_dim=td, T=T, W=W)),
        (FiLMModel,         dict(x_dim=x_dim, task_dim=td, T=T, W=W)),
    ]:
        m = cls(**kwargs)
        out = m(x, p) if cls is not TaskAgnosticModel else m(x)
        n = sum(par.numel() for par in m.parameters())
        print(f"{cls.__name__:20s}  out={out.shape}  params={n}")
