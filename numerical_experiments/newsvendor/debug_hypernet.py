"""
Diagnose why the Hypernetwork underperforms on OOD tasks.

Hypothesis: the hyper_net generates unconstrained weights w.
For in-distribution p values it learns to produce weights that work,
but for OOD p values it extrapolates poorly and the generated weights
blow up in magnitude, destabilizing the dot product with theta.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import generate_data, newsvendor_cost
from models import HypernetworkModel, FiLMModel


def newsvendor_loss(pred, z, h, b):
    return (h * torch.clamp(pred - z, min=0) + b * torch.clamp(z - pred, min=0)).mean()


def train(model, x_train, z_train, tasks, n_epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x_train, z_train), batch_size=128, shuffle=True)
    for _ in range(n_epochs):
        model.train()
        for x_batch, z_batch in loader:
            idx = np.random.randint(len(tasks))
            p = tasks[idx]
            optimizer.zero_grad()
            newsvendor_loss(model(x_batch, p), z_batch, p[0].item(), p[1].item()).backward()
            optimizer.step()


data = generate_data(n_tasks=20, n_samples=2000, x_dim=5, seed=42)
x = torch.tensor(data["x"], dtype=torch.float32)
z = torch.tensor(data["z"], dtype=torch.float32)
tasks = torch.tensor(data["tasks"], dtype=torch.float32)   # in-dist tasks

ood_data = generate_data(n_tasks=20, n_samples=400, x_dim=5, seed=99)
ood_tasks = torch.tensor(ood_data["tasks"], dtype=torch.float32)

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
z_train, z_test = z[:split], z[split:]

hypernet = HypernetworkModel(x_dim=5, p_dim=2)
film     = FiLMModel(x_dim=5, p_dim=2)

print("Training Hypernetwork...")
train(hypernet, x_train, z_train, tasks)
print("Training FiLM...")
train(film, x_train, z_train, tasks)

# -----------------------------------------------------------------------
# Diagnosis 1: weight magnitude for in-dist vs OOD tasks
# -----------------------------------------------------------------------
print("\n--- Hypernetwork: generated weight magnitudes ---")
hypernet.eval()
with torch.no_grad():
    in_norms, ood_norms = [], []
    for p in tasks:
        params = hypernet.hyper_net(p.unsqueeze(0))
        w = params[..., :hypernet.repr_dim]
        in_norms.append(w.norm().item())
    for p in ood_tasks:
        params = hypernet.hyper_net(p.unsqueeze(0))
        w = params[..., :hypernet.repr_dim]
        ood_norms.append(w.norm().item())

print(f"  in-dist  ||w||  mean={np.mean(in_norms):.3f}  std={np.std(in_norms):.3f}  max={np.max(in_norms):.3f}")
print(f"  OOD      ||w||  mean={np.mean(ood_norms):.3f}  std={np.std(ood_norms):.3f}  max={np.max(ood_norms):.3f}")

# -----------------------------------------------------------------------
# Diagnosis 2: prediction variance for in-dist vs OOD tasks
# -----------------------------------------------------------------------
print("\n--- Prediction std across tasks (high std = unstable conditioning) ---")
hypernet.eval()
film.eval()
with torch.no_grad():
    def pred_std(model, task_set):
        preds = [model(x_test, p).numpy() for p in task_set]
        return np.std(np.stack(preds), axis=0).mean()

    print(f"  Hypernetwork in-dist std: {pred_std(hypernet, tasks):.4f}")
    print(f"  Hypernetwork OOD     std: {pred_std(hypernet, ood_tasks):.4f}")
    print(f"  FiLM         in-dist std: {pred_std(film, tasks):.4f}")
    print(f"  FiLM         OOD     std: {pred_std(film, ood_tasks):.4f}")

# -----------------------------------------------------------------------
# Diagnosis 3: sensitivity — how much does output change per unit change in p?
# -----------------------------------------------------------------------
print("\n--- Output sensitivity to p (d output / d p) ---")
def grad_norm(model, x_test, p):
    p = p.clone().requires_grad_(True)
    out = model(x_test[:32], p)
    out.sum().backward()
    return p.grad.norm().item()

hypernet.train(); film.train()   # need grad
in_grads_h, ood_grads_h, in_grads_f, ood_grads_f = [], [], [], []
for p in tasks:
    in_grads_h.append(grad_norm(hypernet, x_test, p))
    in_grads_f.append(grad_norm(film, x_test, p))
for p in ood_tasks:
    ood_grads_h.append(grad_norm(hypernet, x_test, p))
    ood_grads_f.append(grad_norm(film, x_test, p))

print(f"  Hypernetwork in-dist |d/dp|: {np.mean(in_grads_h):.3f}")
print(f"  Hypernetwork OOD     |d/dp|: {np.mean(ood_grads_h):.3f}")
print(f"  FiLM         in-dist |d/dp|: {np.mean(in_grads_f):.3f}")
print(f"  FiLM         OOD     |d/dp|: {np.mean(ood_grads_f):.3f}")

print("\n--- Summary ---")
print("If OOD ||w|| >> in-dist ||w||: the hyper_net is extrapolating weight magnitudes.")
print("If OOD sensitivity >> in-dist sensitivity: small p shifts cause large output swings.")
print("FiLM avoids this because gamma/beta modulate a bounded representation additively,")
print("not via an unconstrained dot product.")
