from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from data import utm_data_generator as utm_dg
from data import utms as utms_lib
from torch_models.transformer import TransformerConfig, TransformerDecoderLM


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


IGNORE_INDEX = -100


def train_one_config(
    *,
    name: str,
    device: str,
    vocab_size: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    widening_factor: int,
    lr: float,
    data_generator,
    training_steps: int,
    log_every: int,
    save_dir: str,
) -> Dict[str, List[float]]:
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        widening_factor=widening_factor,
    )
    model = TransformerDecoderLM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    perplexities: List[float] = []
    grad_norms: List[float] = []

    for step in range(training_steps):
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

        x = torch.from_numpy(tokens).to(device)
        y = torch.from_numpy(targets).to(device)

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
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += float(param_norm.item() ** 2)
        total_norm = math.sqrt(total_norm_sq)
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        perplexities.append(float(math.exp(loss_val)))
        grad_norms.append(total_norm)

        if log_every > 0 and step % log_every == 0:
            print(f"[{name}] step={step} loss={loss_val:.6f} ppl={perplexities[-1]:.3f} grad_norm={total_norm:.3f}")

    out_path = os.path.join(save_dir, f"torch_params_{name}.pt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, out_path)

    return {"loss": losses, "perplexity": perplexities, "grad_norm": grad_norms}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=10)
    parser.add_argument("--maximum_steps", type=int, default=100)
    parser.add_argument("--tokenizer", type=str, default="ascii", choices=["ascii", "seq_position"])
    parser.add_argument("--maximum_program_length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--training_steps", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--widening_factor", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="runs/sml")
    parser.add_argument("--plot", action="store_true")

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

    configs = [
        ("S", 16, 2, 2),
        ("M", 64, 4, 4),
        ("L", 256, 4, 6),
    ]

    all_metrics: Dict[str, Dict[str, List[float]]] = {}

    for name, dim, heads, layers in configs:
        print(f"=== Training {name} (dim={dim}, heads={heads}, layers={layers}) ===")
        metrics = train_one_config(
            name=name,
            device=args.device,
            vocab_size=dg.feature_size,
            model_dim=dim,
            num_heads=heads,
            num_layers=layers,
            widening_factor=args.widening_factor,
            lr=args.lr,
            data_generator=dg,
            training_steps=args.training_steps,
            log_every=args.log_every,
            save_dir=args.output_dir,
        )
        all_metrics[name] = metrics

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "sml_metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib not available: {e}")
            return

        steps = range(args.training_steps)

        plt.figure(figsize=(10, 6))
        for name in ["S", "M", "L"]:
            plt.plot(steps, all_metrics[name]["loss"], label=f"{name}")
        plt.xlabel("step")
        plt.ylabel("loss (NLL)")
        plt.title("Training Loss Trajectories (S/M/L)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sml_losses.png"))

        plt.figure(figsize=(10, 6))
        for name in ["S", "M", "L"]:
            plt.plot(steps, all_metrics[name]["perplexity"], label=f"{name}")
        plt.xlabel("step")
        plt.ylabel("perplexity")
        plt.title("Training Perplexity Trajectories (S/M/L)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sml_perplexity.png"))

        plt.figure(figsize=(10, 6))
        for name in ["S", "M", "L"]:
            plt.plot(steps, all_metrics[name]["grad_norm"], label=f"{name}")
        plt.xlabel("step")
        plt.ylabel("grad_norm (L2)")
        plt.title("Gradient Norm Trajectories (S/M/L)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sml_grad_norm.png"))

        print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
