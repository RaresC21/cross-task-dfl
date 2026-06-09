import numpy as np


def generate_data(
    n_tasks: int,
    n_samples: int,
    x_dim: int = 5,
    seed: int = 42,
    w: np.ndarray = None,
) -> dict:
    """
    Generate cross-task newsvendor data.

    Each task is parameterized by p = (h, b), the holding and backorder costs.
    Demand Z|x is a fixed linear function of x plus noise — independent of p.

    Pass `w` explicitly to share the same demand model across train/test/OOD splits.
    If not provided, w is derived from `seed` (only safe when seed is shared).

    Returns a dict with:
      x        : (n_samples, x_dim)        context features
      z        : (n_samples,)               demand realizations
      tasks    : (n_tasks, 2)              each row is (h, b)
      optimal_q: (n_tasks,)               optimal quantile b/(h+b) per task
      w        : (x_dim,)                 ground-truth demand weights
    """
    rng = np.random.default_rng(seed)

    sigma = 1.0
    if w is None:
        w = rng.standard_normal(x_dim)

    x = rng.standard_normal((n_samples, x_dim))
    mean_demand = x @ w                          # (n_samples,)
    z = np.maximum(0, mean_demand + sigma * rng.standard_normal(n_samples))

    # Tasks: h in (0,1), b in (1,5) so that b > h (backorder costlier)
    h = rng.uniform(0.1, 1.0, size=n_tasks)
    b = rng.uniform(1.0, 5.0, size=n_tasks)
    tasks = np.stack([h, b], axis=1)             # (n_tasks, 2)

    optimal_q = b / (h + b)                      # (n_tasks,)

    return {
        "x": x,
        "z": z,
        "tasks": tasks,
        "optimal_q": optimal_q,
        "w": w,
        "sigma": sigma,
    }


def newsvendor_cost(v: np.ndarray, z: np.ndarray, h: float, b: float) -> np.ndarray:
    """Per-sample newsvendor cost c(v, z) = h*(v-z)^+ + b*(z-v)^+."""
    return h * np.maximum(v - z, 0) + b * np.maximum(z - v, 0)


def optimal_decision(x: np.ndarray, w: np.ndarray, sigma: float, q: float) -> np.ndarray:
    """
    Oracle decision: q-th quantile of N(w^T x, sigma^2).
    Serves as the performance upper bound for a given task.
    """
    from scipy.stats import norm
    mean = x @ w
    return norm.ppf(q, loc=mean, scale=sigma)


if __name__ == "__main__":
    data = generate_data(n_tasks=20, n_samples=1000, x_dim=5)
    print("x shape     :", data["x"].shape)
    print("z shape     :", data["z"].shape)
    print("tasks shape :", data["tasks"].shape)
    print("optimal_q   :", data["optimal_q"].round(3))
