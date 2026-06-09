import torch
import torch.nn as nn
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Baseline: flat linear (no decomposition)
# ---------------------------------------------------------------------------

class FlatModel(nn.Module):
    """f(x, p) = Linear([x; p]). No decomposition."""
    def __init__(self, x_dim: int, p_dim: int):
        super().__init__()
        self.net = nn.Linear(x_dim + p_dim, 1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if p.dim() == 1:
            p = p.unsqueeze(0).expand(x.size(0), -1)
        return self.net(torch.cat([x, p], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Option 1: Concat
# ---------------------------------------------------------------------------

class ConcatModel(nn.Module):
    """f(x, p) = Linear([Linear(x); p])."""
    def __init__(self, x_dim: int, p_dim: int, repr_dim: int = 32):
        super().__init__()
        self.repr    = nn.Linear(x_dim, repr_dim)
        self.task_net = nn.Linear(repr_dim + p_dim, 1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        theta = self.repr(x)
        if p.dim() == 1:
            p = p.unsqueeze(0).expand(x.size(0), -1)
        return self.task_net(torch.cat([theta, p], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Option 2: FiLM
# ---------------------------------------------------------------------------

class FiLMModel(nn.Module):
    """
    f(x, p) = Linear(gamma(p) * Linear(x) + beta(p))
    p conditions the representation via elementwise affine transform.
    Perez et al., AAAI 2018.
    """
    def __init__(self, x_dim: int, p_dim: int, repr_dim: int = 32):
        super().__init__()
        self.repr     = nn.Linear(x_dim, repr_dim)
        self.film_net = nn.Linear(p_dim, 2 * repr_dim)   # outputs gamma and beta
        self.decoder  = nn.Linear(repr_dim, 1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        theta = self.repr(x)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        gamma, beta = self.film_net(p).chunk(2, dim=-1)
        return self.decoder(gamma * theta + beta).squeeze(-1)


# ---------------------------------------------------------------------------
# Option 3: Hypernetwork
# ---------------------------------------------------------------------------

class HypernetworkModel(nn.Module):
    """
    f(x, p) = inner_net_p(Linear(x))
    A linear task network maps p -> weights of the inner linear layer.
    Ha et al., ICLR 2017.
    """
    def __init__(self, x_dim: int, p_dim: int, repr_dim: int = 32):
        super().__init__()
        self.repr     = nn.Linear(x_dim, repr_dim)
        self.repr_dim = repr_dim
        self.hyper_net = nn.Linear(p_dim, repr_dim + 1)  # w and bias for repr_dim -> 1

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        theta = self.repr(x)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        params = self.hyper_net(p)
        w = params[..., :self.repr_dim]
        b = params[..., self.repr_dim:]
        out = (theta * w).sum(dim=-1, keepdim=True) / (self.repr_dim ** 0.5) + b
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Option 4: Structured (oracle / upper bound)
# ---------------------------------------------------------------------------

class StructuredModel(nn.Module):
    """
    Linear(x) -> (mu, log_sigma), decision = mu + sigma * Phi^{-1}(b/(h+b)).
    Exploits newsvendor structure analytically.
    """
    def __init__(self, x_dim: int):
        super().__init__()
        self.net = nn.Linear(x_dim, 2)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        mu    = out[:, 0]
        sigma = torch.exp(out[:, 1]).clamp(min=1e-3)

        if p.dim() == 1:
            p = p.unsqueeze(0)
        h, b = p[:, 0], p[:, 1]
        q = b / (h + b)

        phi_inv = torch.tensor(norm.ppf(q.detach().cpu().numpy()),
                               dtype=x.dtype, device=x.device)
        return mu + sigma * phi_inv


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, x_dim, p_dim = 16, 5, 2
    x = torch.randn(B, x_dim)
    p = torch.tensor([0.5, 2.0])

    for name, model in [
        ("Flat",         FlatModel(x_dim, p_dim)),
        ("Concat",       ConcatModel(x_dim, p_dim)),
        ("FiLM",         FiLMModel(x_dim, p_dim)),
        ("Hypernetwork", HypernetworkModel(x_dim, p_dim)),
        ("Structured",   StructuredModel(x_dim)),
    ]:
        out = model(x, p)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s} output shape: {out.shape}  params: {n_params}")
