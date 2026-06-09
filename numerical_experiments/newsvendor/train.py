import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from data import generate_data, newsvendor_cost
from models import ConcatModel, FiLMModel, HypernetworkModel, StructuredModel


def newsvendor_loss(pred: torch.Tensor, z: torch.Tensor, h: float, b: float) -> torch.Tensor:
    return (h * torch.clamp(pred - z, min=0) + b * torch.clamp(z - pred, min=0)).mean()


def train(model, x_train, z_train, tasks, lr=1e-3, batch_size=128,
          val_fraction=0.2, patience=10, min_delta=1e-4, check_every=10,
          max_epochs=5000, verbose=False):
    """Train until validation loss stops improving (patience-based early stopping).

    Splits x_train/z_train into train/val, monitors val loss every
    `check_every` epochs, and restores the best-val-loss checkpoint on exit.
    Returns the number of epochs run.
    """
    import copy

    # train/val split
    n_val = max(1, int(len(x_train) * val_fraction))
    x_tr, z_tr = x_train[:-n_val], z_train[:-n_val]
    x_val, z_val = x_train[-n_val:], z_train[-n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_tr, z_tr), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for x_batch, z_batch in loader:
            idx = np.random.randint(len(tasks))
            p = tasks[idx]
            h, b = p[0].item(), p[1].item()
            optimizer.zero_grad()
            newsvendor_loss(model(x_batch, p), z_batch, h, b).backward()
            optimizer.step()

        if (epoch + 1) % check_every == 0:
            val_loss = _val_loss(model, x_val, z_val, tasks)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if verbose:
                print(f"  epoch {epoch+1} val_loss={val_loss:.4f} (no_improve={no_improve}/{patience})", flush=True)
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return epoch + 1


@torch.no_grad()
def _val_loss(model, x_val, z_val, tasks) -> float:
    model.eval()
    total = 0.0
    for p in tasks:
        h, b = p[0].item(), p[1].item()
        total += newsvendor_loss(model(x_val, p), z_val, h, b).item()
    model.train()
    return total / len(tasks)


@torch.no_grad()
def evaluate(model, x_test, z_test, tasks) -> float:
    model.eval()
    costs = []
    for p in tasks:
        h, b = p[0].item(), p[1].item()
        pred = model(x_test, p).numpy()
        costs.append(newsvendor_cost(pred, z_test.numpy(), h, b).mean())
    return float(np.mean(costs))


def oracle_cost(data, x_test, z_test) -> float:
    from scipy.stats import norm
    w, sigma = data["w"], data["sigma"]
    total = 0.0
    for (h, b) in data["tasks"]:
        q = b / (h + b)
        mu = x_test.numpy() @ w
        v = norm.ppf(q, loc=mu, scale=sigma)
        total += newsvendor_cost(v, z_test.numpy(), h, b).mean()
    return total / len(data["tasks"])


if __name__ == "__main__":
    data = generate_data(n_tasks=20, n_samples=2000, x_dim=5, seed=42)
    x = torch.tensor(data["x"], dtype=torch.float32)
    z = torch.tensor(data["z"], dtype=torch.float32)
    tasks = torch.tensor(data["tasks"], dtype=torch.float32)

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    z_train, z_test = z[:split], z[split:]

    models = {
        "Concat":       ConcatModel(x.shape[1], tasks.shape[1]),
        "FiLM":         FiLMModel(x.shape[1], tasks.shape[1]),
        "Hypernetwork": HypernetworkModel(x.shape[1], tasks.shape[1]),
        "Structured":   StructuredModel(x.shape[1]),
    }

    results = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        train(model, x_train, z_train, tasks, verbose=True)
        results[name] = evaluate(model, x_test, z_test, tasks)

    oracle = oracle_cost(data, x_test, z_test)

    print("\n--- Test cost (lower is better) ---")
    for name, cost in results.items():
        print(f"  {name:15s} {cost:.4f}")
    print(f"  {'Oracle':15s} {oracle:.4f}")
