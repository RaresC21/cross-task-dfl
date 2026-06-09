"""
Compare all architectures across a variety of scenarios.

Scenarios vary along three axes:
  - n_tasks:   how many tasks are seen during training
  - n_samples: training set size
  - x_dim:     feature dimensionality

Each scenario is evaluated on:
  - in-distribution tasks (seen during training)
  - out-of-distribution tasks (held-out tasks, unseen during training)
"""

import csv
import itertools
import os
import numpy as np
import torch

from data import generate_data, newsvendor_cost
from models import FlatModel, ConcatModel, FiLMModel, HypernetworkModel, StructuredModel
from train import train, evaluate, seed_everything


SCENARIOS = list(itertools.product(
    [1, 10, 100, 500],   # n_tasks
    [100, 200, 500, 750, 1000, 2000],         # n_samples
    [5, 20, 50, 100],           # x_dim
))

LR         = 1e-3
BATCH_SIZE = 32
TRAIN_SEED = 42
OOD_SEED   = 99
TEST_SEED  = 77
N_TEST     = 2000   # fixed test set size, same across all scenarios
N_OOD_TASKS = 1000   # fixed number of OOD tasks regardless of n_tasks
VERBOSE    = False

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")
MODEL_NAMES  = ["Flat", "Concat", "FiLM", "Hypernetwork", "Structured"]


def make_models(x_dim, p_dim=2):
    return {
        "Flat":         FlatModel(x_dim, p_dim),
        "Concat":       ConcatModel(x_dim, p_dim),
        "FiLM":         FiLMModel(x_dim, p_dim),
        "Hypernetwork": HypernetworkModel(x_dim, p_dim),
        "Structured":   StructuredModel(x_dim),
    }


def run():
    header = f"{'n_tasks':>8} {'n_samples':>10} {'x_dim':>6} | " + \
             " ".join(f"{n:>14}" for n in MODEL_NAMES) + \
             " | " + " ".join(f"OOD-{n:>10}" for n in MODEL_NAMES)
    print(header)
    print("-" * len(header))

    csv_rows = []

    # generate one fixed w (demand model) and test set per x_dim,
    # shared across ALL scenarios so Z|x is the same everywhere
    x_dims = sorted(set(x_dim for _, _, x_dim in SCENARIOS))
    demand_models = {}   # x_dim -> w
    test_sets = {}
    for x_dim in x_dims:
        w_seed = np.random.default_rng(TEST_SEED).standard_normal(x_dim)
        demand_models[x_dim] = w_seed
        td = generate_data(n_tasks=1, n_samples=N_TEST, x_dim=x_dim,
                           seed=TEST_SEED, w=w_seed)
        test_sets[x_dim] = (
            torch.tensor(td["x"], dtype=torch.float32),
            torch.tensor(td["z"], dtype=torch.float32),
        )

    # fixed OOD tasks (same regardless of n_tasks)
    ood_tasks_by_dim = {}
    for x_dim in x_dims:
        ood_d = generate_data(n_tasks=N_OOD_TASKS, n_samples=1,
                              x_dim=x_dim, seed=OOD_SEED,
                              w=demand_models[x_dim])
        ood_tasks_by_dim[x_dim] = torch.tensor(ood_d["tasks"], dtype=torch.float32)

    for (n_tasks, n_samples, x_dim) in SCENARIOS:
        print(f"\n>>> scenario: n_tasks={n_tasks}, n_samples={n_samples}, x_dim={x_dim}")

        w = demand_models[x_dim]
        x_test, z_test = test_sets[x_dim]
        ood_tasks = ood_tasks_by_dim[x_dim]

        data = generate_data(n_tasks=n_tasks, n_samples=n_samples,
                             x_dim=x_dim, seed=TRAIN_SEED, w=w)
        x_train = torch.tensor(data["x"], dtype=torch.float32)
        z_train = torch.tensor(data["z"], dtype=torch.float32)
        tasks   = torch.tensor(data["tasks"], dtype=torch.float32)

        in_costs, ood_costs = {}, {}
        for i, (name, model) in enumerate(make_models(x_dim).items()):
            seed_everything(TRAIN_SEED + i)
            print(f"  [{name}]", flush=True)
            epochs_run = train(model, x_train, z_train, tasks, lr=LR, batch_size=BATCH_SIZE, verbose=VERBOSE)
            in_costs[name]  = evaluate(model, x_test, z_test, tasks)
            ood_costs[name] = evaluate(model, x_test, z_test, ood_tasks)
            print(f"  -> converged in {epochs_run} epochs | in={in_costs[name]:.4f} ood={ood_costs[name]:.4f}")

        row_str = f"{n_tasks:>8} {n_samples:>10} {x_dim:>6} | "
        row_str += " ".join(f"{in_costs[n]:>14.4f}" for n in MODEL_NAMES)
        row_str += " | "
        row_str += " ".join(f"{ood_costs[n]:>14.4f}" for n in MODEL_NAMES)
        print(row_str)

        csv_rows.append({
            "n_tasks": n_tasks, "n_samples": n_samples, "x_dim": x_dim,
            **{f"in_{n}":  in_costs[n]  for n in MODEL_NAMES},
            **{f"ood_{n}": ood_costs[n] for n in MODEL_NAMES},
        })

    fieldnames = ["n_tasks", "n_samples", "x_dim"] + \
                 [f"in_{n}"  for n in MODEL_NAMES] + \
                 [f"ood_{n}" for n in MODEL_NAMES]
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run()
