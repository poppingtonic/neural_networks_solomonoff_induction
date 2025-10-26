from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


def save_mswag(mswag: Any, path: str) -> None:
    """Save a MultiSWAG object to disk.

    Tries torch.save with pickle. The object must be importable to load later.
    """
    torch.save(mswag, path)


def load_mswag(path: str, map_location: Optional[str | torch.device] = None) -> Any:
    """Load a MultiSWAG object saved with save_mswag()."""
    return torch.load(path, map_location=map_location)


# --- Optional helpers for regression-style visualization (toy demos) ---

def posterior_predict_regression(
    mswag: Any,
    x_grid: torch.Tensor,
    loss_fn: torch.nn.Module,
    *,
    num_samples: int = 20,
    modes: Sequence[str] = ("mean", "std", "pred"),
    f_reg: bool = True,
):
    """Thin wrapper around mswag.posterior_pred for regression demos."""
    return mswag.posterior_pred(x_grid, loss_fn, num_samples=num_samples, mode=list(modes), f_reg=f_reg)


def plot_regression_predictions(outputs: dict, X: torch.Tensor, Y: torch.Tensor, x_grid: torch.Tensor, num_models: int, title: str = "MultiSWAG") -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[12, 6])

    # Predictive mean and std
    axs[0].plot(X, Y, "kx", label="Toy data", markersize=1)
    axs[0].set_xlim(-0.5, 1)
    axs[0].set_ylim(-2, 2)
    axs[0].plot(x_grid, outputs["mean"], "r--", linewidth=1)
    axs[0].fill_between(x_grid.reshape(1, -1)[0], (outputs["mean"] - outputs["std"]).squeeze(), (outputs["mean"] + outputs["std"]).squeeze(), alpha=0.5, color="red")
    axs[0].fill_between(
        x_grid.reshape(1, -1)[0], (outputs["mean"] + 2 * outputs["std"]).squeeze(), (outputs["mean"] - 2 * outputs["std"]).squeeze(), alpha=0.2, color="red"
    )
    axs[0].set_title("Predictive Mean and Std")

    # Individual model predictions
    axs[1].plot(X, Y, "kx", label="Toy data", markersize=1)
    axs[1].set_xlim(-0.5, 1)
    axs[1].set_ylim(-1.5, 2)
    axs[1].set_title("Individual Model output")

    for j in range(num_models):
        axs[1].plot(x_grid, torch.tensor([sublist[j] for sublist in outputs["pred"]]), linestyle="--", linewidth=1)

    fig.suptitle(title, fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()
