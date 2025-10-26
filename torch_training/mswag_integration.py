from __future__ import annotations

from typing import Callable, Optional

import torch

from utm_dataset import UTMIterableDataset, IGNORE_INDEX


def make_train_loader(data_generator, batches_per_epoch: int = 1):
    """Build a DataLoader yielding (tokens, targets) per step.

    We yield a single (B, T) batch per iteration to match the online generator.
    """
    ds = UTMIterableDataset(data_generator, batches_per_epoch=batches_per_epoch)
    # Unwrap the single sample per batch from IterableDataset
    def _collate(batch):
        return batch[0]

    return torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=_collate)


def default_lm_loss_fn(pred_log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Language modeling NLL with ignore index.

    pred_log_probs: (B, T, V) log-probabilities
    targets: (B, T) with IGNORE_INDEX where padded
    """
    B, T, V = pred_log_probs.shape
    return torch.nn.functional.nll_loss(
        pred_log_probs.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )


def train_with_mswag(
    data_generator,
    model_factory: Callable[[], torch.nn.Module],
    *,
    pretrain_epochs: int,
    swag_epochs: int,
    cov_mat_rank: int,
    num_models: int,
    lr: float,
    num_devices: int = 1,
    batches_per_epoch: int = 1,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """Attempt to train with MultiSWAG via push.bayes.swag; returns mswag or None.

    Falls back to None if the package is not available.
    """
    try:
        import push.bayes.swag as push_swag  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(
            "[MultiSWAG] push.bayes.swag not available: "
            f"{e}. Skipping MultiSWAG and returning None."
        )
        return None

    # Wrap our factory into a no-arg class to satisfy MultiSWAG API expectations.
    class _GenericNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = model_factory()

        def forward(self, x):
            return self.net(x)

    train_loader = make_train_loader(data_generator, batches_per_epoch=batches_per_epoch)
    loss_fn = loss_fn or default_lm_loss_fn

    mswag = push_swag.train_mswag(
        train_loader,
        loss_fn,
        pretrain_epochs,
        swag_epochs,
        _GenericNet,  # NN template
        *[],
        num_devices=num_devices,
        num_models=num_models,
        cov_mat_rank=cov_mat_rank,
        lr=lr,
        mswag_state={},
    )
    return mswag
