from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data import utm_data_generator as utm_dg
from data import utms as utms_lib
from torch_models.transformer import TransformerConfig, TransformerDecoderLM
from utm_dataset import IGNORE_INDEX
from mswag_integration import train_with_mswag
from mswag_utils import save_mswag


def build_data_generator(
    *,
    batch_size: int,
    seq_length: int,
    memory_size: int,
    maximum_steps: int,
    tokenizer: str,
    maximum_program_length: int,
    seed: int,
):
    rng = np.random.default_rng(seed=seed)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)
    tokenizer_enum = utm_dg.Tokenizer.ASCII if tokenizer.lower() == "ascii" else utm_dg.Tokenizer.SEQ_POSITION
    return utm_dg.UTMDataGenerator(
        batch_size=batch_size,
        seq_length=seq_length,
        rng=rng,
        utm=utm,
        memory_size=memory_size,
        maximum_steps=maximum_steps,
        tokenizer=tokenizer_enum,
        maximum_program_length=maximum_program_length,
    )


def train_epoch_step(model: torch.nn.Module, opt: torch.optim.Optimizer, data_generator) -> float:
    model.train()
    sequences, log_dict = data_generator.sample()
    sequences = np.asarray(sequences)
    tokens = np.argmax(sequences, axis=-1).astype(np.int64)
    if 'loss_mask' in log_dict:
        loss_mask = np.asarray(log_dict['loss_mask']).astype(bool)
    else:
        loss_mask = np.zeros(tokens.shape, dtype=bool)
    targets = tokens.copy()
    targets[loss_mask] = IGNORE_INDEX

    x = torch.from_numpy(tokens)
    y = torch.from_numpy(targets)

    x = x.to(next(model.parameters()).device)
    y = y.to(x.device)

    opt.zero_grad(set_to_none=True)
    log_probs = model(x)
    B, T, V = log_probs.shape
    loss = F.nll_loss(
        log_probs.reshape(B * T, V),
        y.reshape(B * T),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    loss.backward()
    opt.step()
    return float(loss.detach().cpu().item())


def run_standard_training(
    *,
    vocab_size: int,
    training_steps: int,
    log_every: int,
    device: str,
    data_generator,
    lr: float,
    model_dim: int,
    num_layers: int,
    num_heads: int,
    widening_factor: int,
    save_path: Optional[str] = None,
) -> tuple[TransformerDecoderLM, float]:
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        widening_factor=widening_factor,
    )
    model = TransformerDecoderLM(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = 0.0
    for step in range(training_steps):
        loss = train_epoch_step(model, opt, data_generator)
        last_loss = loss
        if log_every > 0 and step % log_every == 0:
            print(f"step={step} loss={loss:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, save_path)

    return model, last_loss


def turn_the_swag_on(
    *,
    enable_mswag: bool,
    data_generator,
    vocab_size: int,
    model_dim: int,
    num_layers: int,
    num_heads: int,
    widening_factor: int,
    mswag_pretrain_epochs: int,
    mswag_swag_epochs: int,
    mswag_cov_rank: int,
    mswag_num_models: int,
    mswag_num_devices: int,
    mswag_lr: float,
    mswag_batches_per_epoch: int,
):
    if not enable_mswag:
        return None

    def model_factory():
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            embedding_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            widening_factor=widening_factor,
        )
        return TransformerDecoderLM(cfg)

    return train_with_mswag(
        data_generator=data_generator,
        model_factory=model_factory,
        pretrain_epochs=mswag_pretrain_epochs,
        swag_epochs=mswag_swag_epochs,
        cov_mat_rank=mswag_cov_rank,
        num_models=mswag_num_models,
        lr=mswag_lr,
        num_devices=mswag_num_devices,
        batches_per_epoch=mswag_batches_per_epoch,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=10)
    parser.add_argument("--maximum_steps", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default="ascii", choices=["ascii", "seq_position"])
    parser.add_argument("--maximum_program_length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--widening_factor", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="torch_params.pt")

    parser.add_argument("--enable_mswag", action="store_true")
    parser.add_argument("--mswag_pretrain_epochs", type=int, default=200)
    parser.add_argument("--mswag_swag_epochs", type=int, default=100)
    parser.add_argument("--mswag_cov_rank", type=int, default=20)
    parser.add_argument("--mswag_num_models", type=int, default=4)
    parser.add_argument("--mswag_num_devices", type=int, default=1)
    parser.add_argument("--mswag_lr", type=float, default=3e-2)
    parser.add_argument("--mswag_batches_per_epoch", type=int, default=1)
    parser.add_argument("--mswag_save_path", type=str, default="mswag.pt")

    args = parser.parse_args(argv)

    dg = build_data_generator(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        memory_size=args.memory_size,
        maximum_steps=args.maximum_steps,
        tokenizer=args.tokenizer,
        maximum_program_length=args.maximum_program_length,
        seed=args.seed,
    )

    mswag_obj = turn_the_swag_on(
        enable_mswag=args.enable_mswag,
        data_generator=dg,
        vocab_size=dg.feature_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        widening_factor=args.widening_factor,
        mswag_pretrain_epochs=args.mswag_pretrain_epochs,
        mswag_swag_epochs=args.mswag_swag_epochs,
        mswag_cov_rank=args.mswag_cov_rank,
        mswag_num_models=args.mswag_num_models,
        mswag_num_devices=args.mswag_num_devices,
        mswag_lr=args.mswag_lr,
        mswag_batches_per_epoch=args.mswag_batches_per_epoch,
    )

    if mswag_obj is None:
        model, last_loss = run_standard_training(
            vocab_size=dg.feature_size,
            training_steps=args.training_steps,
            log_every=args.log_every,
            device=args.device,
            data_generator=dg,
            lr=args.lr,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            widening_factor=args.widening_factor,
            save_path=args.save_path,
        )
        print(f"final_loss={last_loss:.6f}; saved={args.save_path}")
    else:
        print("MultiSWAG training completed. Use the returned object for posterior predictions.")
        if args.mswag_save_path:
            save_mswag(mswag_obj, args.mswag_save_path)
            print(f"MultiSWAG object saved to {args.mswag_save_path}")


if __name__ == "__main__":
    main()
