from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import push.bayes.swag as push_swag  # type: ignore
except Exception as e:  # noqa: BLE001
    push_swag = None

from mswag_utils import save_mswag, posterior_predict_regression, plot_regression_predictions


def target_toy(x: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    epsilons = torch.randn(3) * 0.02
    return (
        x + 0.3 * torch.sin(2 * torch.pi * (x + epsilons[0]))
        + 0.3 * torch.sin(4 * torch.pi * (x + epsilons[1]))
        + epsilons[2]
    )


class GenericNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_toy_data(num_samples: int = 100, input_dim: int = 1):
    torch.manual_seed(0)
    X = torch.rand(num_samples, input_dim) * 0.5
    Y = torch.stack([target_toy(x, seed) for x, seed in zip(X, range(X.shape[0]))])
    dataset = TensorDataset(X, Y)
    return X, Y, dataset


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--x_grid_min", type=float, default=-5.0)
    parser.add_argument("--x_grid_max", type=float, default=5.0)
    parser.add_argument("--x_grid_points", type=int, default=1000)

    parser.add_argument("--pretrain_epochs", type=int, default=2000)
    parser.add_argument("--swag_epochs", type=int, default=1000)
    parser.add_argument("--cov_mat_rank", type=int, default=20)
    parser.add_argument("--num_models", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--save_path", type=str, default="mswag_toy.pt")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args(argv)

    if push_swag is None:
        print("push.bayes.swag is not available. Please install to run this demo.")
        return

    X, Y, dataset = make_toy_data(args.num_samples, args.input_dim)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train MultiSWAG
    mswag = push_swag.train_mswag(
        train_loader,
        torch.nn.MSELoss(),
        args.pretrain_epochs,
        args.swag_epochs,
        GenericNet,
        args.input_dim,
        num_devices=args.num_devices,
        num_models=args.num_models,
        cov_mat_rank=args.cov_mat_rank,
        lr=args.lr,
        mswag_state={},
    )

    # Save
    if args.save_path:
        save_mswag(mswag, args.save_path)
        print(f"Saved MultiSWAG to {args.save_path}")

    # Visualize posterior predictive uncertainty
    if args.plot:
        x_grid = torch.linspace(args.x_grid_min, args.x_grid_max, args.x_grid_points).reshape(-1, args.input_dim)
        outputs = posterior_predict_regression(
            mswag, x_grid, torch.nn.MSELoss(), num_samples=20, modes=("mean", "std", "pred"), f_reg=True
        )
        plot_regression_predictions(outputs, X, Y, x_grid, num_models=args.num_models, title="MultiSWAG Toy Regression")


if __name__ == "__main__":
    main()
