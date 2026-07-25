"""
Bloem+CTW Supervised Fine-Tuning for Tinker API.

This recipe fine-tunes pretrained LLMs on universal data (Bloem LSTM + CTW trees)
for Solomonoff Induction approximation. Based on train_pytorch_hooks.ipynb.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         LOCAL (CPU)                              │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
    │  │ LSTM Source │───▶│ BloemBuffer │───▶│ Batch Conversion    │  │
    │  │ (reweight)  │    │ (30k seqs)  │    │ (tokens → Datum)    │  │
    │  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
    │                                                    │             │
    │  ┌─────────────┐                                   │             │
    │  │ CTWGenerator│───────────────────────────────────┤             │
    │  │ (curriculum)│                                   │             │
    │  └─────────────┘                                   │             │
    └────────────────────────────────────────────────────┼─────────────┘
                                                         │
                         ┌───────────────────────────────▼───────────────┐
                         │                 TINKER API (GPU)               │
                         │  ┌─────────────────────────────────────────┐  │
                         │  │ forward_backward(datums, "cross_entropy")│  │
                         │  │ optim_step(adam_params)                  │  │
                         │  │ save_weights_for_sampler() ──────────────┼──┼──┐
                         │  └─────────────────────────────────────────┘  │  │
                         └───────────────────────────────────────────────┘  │
                                                                            │
    ┌───────────────────────────────────────────────────────────────────────┘
    │  MODEL SAMPLING (for Bloem buffer updates)
    │  ┌─────────────────────────────────────────────────────────────────┐
    │  │ sampling_client.sample(prompt, max_tokens) → new sequences      │
    │  │ Update BloemBuffer with model-generated sequences              │
    │  └─────────────────────────────────────────────────────────────────┘

Key Concepts:
    - BloemBuffer: Stores 30k sequences, iteratively refined by LSTM source
    - CTWGenerator: Generates M-ary CTW tree sequences with curriculum
    - Tinker LoRA: Fine-tunes large models (Llama-3.2-3B) efficiently
    - Model sampling: Periodically updates Bloem buffer with model generations

Usage:
    python bloem_ctw_sft.py --base_model meta-llama/Llama-3.2-3B --num_steps 5000

References:
    - Peter Bloem (2025): "Universal pre-training by iterated random computation"
      (See: neural_networks_solomonoff_induction/universal_pretraining_solomonoff_induction/)
    - CTW: Context Tree Weighting for sequence modeling
    - Notebook: train_pytorch_hooks.ipynb
"""

import logging
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import chz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# =============================================================================
# Configuration (matches train_pytorch_hooks.ipynb TrainingConfig)
# =============================================================================

@chz.chz
class BloemCTWConfig:
    """Configuration for Bloem+CTW fine-tuning on Tinker."""
    
    # Tinker settings
    base_url: str | None = None
    log_path: str = "/tmp/tinker-bloem-ctw"
    base_model: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 64
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    num_steps: int = 10000
    save_every: int = 500
    log_every: int = 50
    
    # Sequence settings
    seq_length: int = 2048  # n_ctx in notebook
    d_vocab: int = 256  # Token vocabulary size
    
    # Bloem buffer (CRITICAL: must be large enough!)
    bloem_buffer_size: int = 30000  # 6x larger than default 5000
    bloem_weight_mult_range: Tuple[float, float] = (1.0, 3.5)
    bloem_temperature: float = 0.3  # Sampling temperature
    bloem_update_interval: int = 100  # Steps between model sampling updates
    
    # CTW settings
    ctw_max_depth: int = 16
    ctw_min_arity: int = 2
    ctw_max_arity: int = 256
    ctw_curriculum_mode: str = "weighted"
    ctw_curriculum_steps: int = 5000
    ctw_curriculum_temperature: float = 2.0
    ctw_dirichlet_alpha: float = 0.5
    ctw_use_sparse_dirichlet: bool = True
    
    # Data mixing
    data_source: str = "bloem_ctw"  # "bloem", "ctw", or "bloem_ctw"
    bloem_ratio: float = 0.5  # Fraction of batch from Bloem (rest from CTW)
    
    # W&B (optional)
    wandb_project: str | None = "sol"
    wandb_entity: str | None = "nlug"


# =============================================================================
# LSTM Source Generator (from notebook)
# =============================================================================

@dataclass
class BloemSourceConfig:
    """Configuration for LSTM source generator."""
    num_tokens: int = 256
    emb: int = 256
    lstm_hidden: int = 512
    lstm_layers: int = 2
    weight_mult_range: Tuple[float, float] = (1.0, 3.5)
    emb_mult: float = 0.3
    temperature: float = 0.3


class LSTMSourceGenerator(nn.Module):
    """
    LSTM source generator for Bloem-style pretraining.
    
    Generates sequences by conditioning on buffer sequences and sampling
    from the LSTM's output distribution. The LSTM is randomly reweighted
    each batch to produce diverse computable sequences.
    
    From train_pytorch_hooks.ipynb.
    """
    
    def __init__(self, config: BloemSourceConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.num_tokens, config.emb)
        self.lstm = nn.LSTM(
            config.emb * 2, config.lstm_hidden,
            config.lstm_layers, batch_first=True
        )
        self.to_logits = nn.Linear(config.lstm_hidden, config.num_tokens)
    
    def reset_and_scale(self, weight_mult: float):
        """Reset parameters and scale LSTM weights."""
        self.token_embedding.reset_parameters()
        self.lstm.reset_parameters()
        self.to_logits.reset_parameters()
        for n, p in self.lstm.named_parameters():
            if 'weight' in n:
                p.data *= weight_mult
        self.token_embedding.weight.data *= self.config.emb_mult
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, 
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass with conditioning."""
        combined = torch.cat([
            self.token_embedding(x),
            self.token_embedding(cond)
        ], dim=-1)
        out, hidden = self.lstm(combined, hidden)
        return self.to_logits(out), hidden
    
    @torch.no_grad()
    def sample_sequence(self, seed: torch.Tensor, length: int,
                       temperature: float = 0.5,
                       conditional: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample a sequence given a seed and optional conditioning."""
        b, sl = seed.shape
        dev = seed.device
        if conditional is None:
            conditional = torch.zeros(b, length, dtype=torch.long, device=dev)
        
        seq = seed.clone()
        hid = None
        
        if sl > 1:
            _, hid = self.forward(seq[:, :-1], conditional[:, :sl-1], hid)
        
        for i in range(length - sl):
            pos = sl + i - 1
            logits, hid = self.forward(
                seq[:, -1:], conditional[:, pos:pos+1], hid
            )
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            seq = torch.cat([seq, torch.multinomial(probs, 1)], dim=1)
        
        return seq


# =============================================================================
# Bloem Buffer (from notebook)
# =============================================================================

class BloemBuffer:
    """
    Buffer for Bloem-style iterative conditioning.
    
    Stores sequences that are iteratively refined by the LSTM source.
    The buffer is sampled from during training, and updated with new
    sequences generated by conditioning on existing buffer contents.
    
    CRITICAL: Buffer size should be large (30k+) for long training runs
    to prevent curriculum exhaustion (iterations > 5 too early).
    """
    
    def __init__(self, size: int, seq_length: int, num_tokens: int):
        self.size = size
        self.seq_length = seq_length
        self.num_tokens = num_tokens
        # Initialize with random sequences
        self.buffer = torch.randint(0, num_tokens, (size, seq_length))
        self.update_counts = torch.zeros(size, dtype=torch.long)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, List[int]]:
        """Sample batch_size sequences and their indices."""
        idx = random.sample(range(self.size), batch_size)
        return self.buffer[idx], idx
    
    def update(self, idx: List[int], seqs: torch.Tensor):
        """Update buffer at indices with new sequences."""
        self.buffer[idx] = seqs.cpu()
        self.update_counts[idx] += 1
    
    def avg_iterations(self) -> float:
        """Average number of times sequences have been updated."""
        return self.update_counts.float().mean().item()
    
    def get_state(self) -> dict:
        """Get buffer state for checkpointing."""
        return {
            'buffer': self.buffer.clone(),
            'update_counts': self.update_counts.clone(),
        }
    
    def load_state(self, state: dict):
        """Restore buffer state from checkpoint."""
        self.buffer = state['buffer']
        self.update_counts = state['update_counts']


# =============================================================================
# CTW Generator (from notebook)
# =============================================================================

@dataclass
class CTWNodeConfig:
    """Configuration for CTW nodes."""
    max_depth: int = 16
    spawn_prob: float = 0.5
    dirichlet_alpha: float = 0.5
    use_sparse_dirichlet: bool = True


def sparse_dirichlet_sample(rng: np.random.Generator, alpha: float,
                           size: int, active: Optional[int] = None) -> np.ndarray:
    """Sample from sparse Dirichlet (most mass on few symbols)."""
    if active is None:
        active = rng.integers(2, size + 1)
    idx = rng.choice(size, size=active, replace=False)
    g = rng.gamma(alpha, 1.0, size=active)
    probs = np.zeros(size)
    probs[idx] = g / g.sum()
    return probs


class CTWNode:
    """Single node in CTW tree with Dirichlet-sampled distribution."""
    
    def __init__(self, depth: int, config: CTWNodeConfig,
                 rng: np.random.Generator, arity: int):
        self.depth = depth
        self.config = config
        self.rng = rng
        self.arity = arity
        self.children = {}
        
        if config.use_sparse_dirichlet:
            self.probs = sparse_dirichlet_sample(
                rng, config.dirichlet_alpha, arity
            )
        else:
            self.probs = rng.dirichlet([config.dirichlet_alpha] * arity)
    
    def get_child(self, sym: int) -> Optional['CTWNode']:
        """Get or create child node for symbol."""
        if sym not in self.children:
            if (self.depth < self.config.max_depth and 
                self.rng.random() < self.config.spawn_prob):
                self.children[sym] = CTWNode(
                    self.depth + 1, self.config, self.rng, self.arity
                )
            else:
                self.children[sym] = None
        return self.children[sym]
    
    def sample(self) -> int:
        """Sample symbol from node's distribution."""
        if self.probs.sum() > 0:
            return self.rng.choice(self.arity, p=self.probs)
        return self.rng.integers(0, self.arity)


class CTWGenerator:
    """
    CTW tree sequence generator with curriculum learning.
    
    Generates sequences from M-ary CTW trees with weighted curriculum
    that gradually increases arity from min_arity to max_arity.
    """
    
    def __init__(self, batch_size: int, seq_length: int,
                 config: BloemCTWConfig, seed: int = 42):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        self.arities = np.arange(config.ctw_min_arity, config.ctw_max_arity + 1)
        self.num_arities = len(self.arities)
        self.step = 0
        
        self.node_config = CTWNodeConfig(
            max_depth=config.ctw_max_depth,
            dirichlet_alpha=config.ctw_dirichlet_alpha,
            use_sparse_dirichlet=config.ctw_use_sparse_dirichlet,
        )
    
    def _get_weighted_dist(self) -> np.ndarray:
        """Get curriculum-weighted arity distribution."""
        prog = min(1.0, self.step / self.config.ctw_curriculum_steps)
        target = self.config.ctw_min_arity + prog * (
            self.config.ctw_max_arity - self.config.ctw_min_arity
        )
        w = np.exp(
            -self.config.ctw_curriculum_temperature * 
            np.abs(self.arities - target) / self.num_arities
        )
        # 90% curriculum, 10% uniform exploration
        return 0.9 * (w / w.sum()) + 0.1 / self.num_arities
    
    def _get_arities(self) -> List[int]:
        """Get arities for current batch."""
        if self.config.ctw_curriculum_mode == "weighted":
            return self.rng.choice(
                self.arities, self.batch_size,
                p=self._get_weighted_dist()
            ).tolist()
        return self.rng.integers(
            self.config.ctw_min_arity,
            self.config.ctw_max_arity + 1,
            self.batch_size
        ).tolist()
    
    def generate_sequence(self, arity: int) -> np.ndarray:
        """Generate single sequence from CTW tree with given arity."""
        root = CTWNode(0, self.node_config, self.rng, arity)
        seq = []
        ctx = []
        
        for _ in range(self.seq_length):
            node = root
            for sym in reversed(ctx[-self.node_config.max_depth:]):
                child = node.get_child(sym)
                if child is None:
                    break
                node = child
            s = node.sample()
            seq.append(s)
            ctx.append(s)
        
        return np.array(seq, dtype=np.int64)
    
    def __call__(self) -> torch.Tensor:
        """Generate batch of CTW sequences."""
        arities = self._get_arities()
        seqs = [self.generate_sequence(a) for a in arities]
        self.step += 1
        return torch.tensor(np.stack(seqs), dtype=torch.long)
    
    def set_step(self, step: int):
        """Set curriculum step."""
        self.step = step


# =============================================================================
# Tinker Data Conversion
# =============================================================================

def batch_to_tinker_datums(batch: torch.Tensor,
                           tokenizer=None) -> List[types.Datum]:
    """
    Convert batch of token sequences to Tinker Datum format.
    
    Args:
        batch: Tensor of shape (batch_size, seq_length) with token IDs
        tokenizer: Optional tokenizer for token remapping (if d_vocab != LLM vocab)
    
    Returns:
        List of Tinker Datum objects for forward_backward()
    
    Note:
        If using a pretrained LLM with different vocabulary, tokens should be
        remapped. For byte-level training (d_vocab=256), tokens can map directly
        to byte values in the LLM's vocabulary.
    """
    datums = []
    
    for seq in batch:
        tokens = seq.tolist()
        
        # For pretrained LLMs, we may need to remap tokens
        # Option 1: Use byte tokens directly (if LLM supports byte-level)
        # Option 2: Encode as text and re-tokenize
        # For now, assume direct token mapping works
        
        input_tokens = tokens[:-1]  # All but last
        target_tokens = tokens[1:]  # All but first
        
        # Create Tinker Datum
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.long)
                ),
                "weights": TensorData.from_torch(
                    torch.ones(len(target_tokens), dtype=torch.float32)
                ),
            },
        )
        datums.append(datum)
    
    return datums


def compute_mean_nll(logprobs: List, weights: List) -> float:
    """Compute mean negative log-likelihood from forward_backward outputs."""
    total_nll = 0.0
    total_weight = 0.0
    
    for lp, w in zip(logprobs, weights):
        if isinstance(lp, TensorData):
            lp = lp.to_torch()
        if isinstance(w, TensorData):
            w = w.to_torch()
        
        total_nll += (-lp * w).sum().item()
        total_weight += w.sum().item()
    
    return total_nll / max(total_weight, 1e-8)


# =============================================================================
# Main Training Loop
# =============================================================================

def main(config: BloemCTWConfig):
    """Main training loop for Bloem+CTW fine-tuning on Tinker."""
    
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=f"bloem_ctw_{config.base_model.split('/')[-1]}",
        config=config,
        do_configure_logging_module=True,
    )
    
    logger.info(f"Starting Bloem+CTW fine-tuning on {config.base_model}")
    logger.info(f"Buffer size: {config.bloem_buffer_size}, Steps: {config.num_steps}")
    
    # Initialize local components (CPU)
    # ---------------------------------
    
    # LSTM source for Bloem
    bloem_source_config = BloemSourceConfig(
        num_tokens=config.d_vocab,
        weight_mult_range=config.bloem_weight_mult_range,
        temperature=config.bloem_temperature,
    )
    bloem_source = LSTMSourceGenerator(bloem_source_config)
    
    # Bloem buffer
    bloem_buffer = BloemBuffer(
        config.bloem_buffer_size,
        config.seq_length,
        config.d_vocab,
    )
    logger.info(f"Bloem buffer initialized: {config.bloem_buffer_size} sequences")
    
    # CTW generator
    ctw_gen = CTWGenerator(config.batch_size, config.seq_length, config)
    logger.info(f"CTW generator initialized: arity {config.ctw_min_arity}-{config.ctw_max_arity}")
    
    # Initialize Tinker (remote GPU)
    # ------------------------------
    
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    # Check for checkpoint resumption
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_step = resume_info["batch"]
        
        # Restore Bloem buffer state
        if "bloem_buffer" in resume_info:
            bloem_buffer.load_state(resume_info["bloem_buffer"])
            logger.info(f"Restored Bloem buffer (avg iterations: {bloem_buffer.avg_iterations():.1f})")
        
        # Restore CTW step
        if "ctw_step" in resume_info:
            ctw_gen.set_step(resume_info["ctw_step"])
        
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.base_model,
            rank=config.lora_rank,
        )
        start_step = 0
    
    # Training loop
    # -------------
    
    logger.info(f"Training for {config.num_steps} steps")
    
    for step in range(start_step, config.num_steps):
        t_start = time.time()
        metrics = {}
        
        # Learning rate schedule (linear decay)
        if step < config.warmup_steps:
            lr_mult = step / config.warmup_steps
        else:
            lr_mult = max(0.0, 1.0 - (step - config.warmup_steps) / 
                         (config.num_steps - config.warmup_steps))
        current_lr = config.learning_rate * lr_mult
        adam_params = tinker.AdamParams(
            learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
        )
        
        # Generate batch (local CPU)
        # --------------------------
        
        if config.data_source == "bloem":
            batch = _get_bloem_batch(bloem_source, bloem_buffer, config)
        elif config.data_source == "ctw":
            batch = ctw_gen()
        else:  # bloem_ctw
            n_bloem = int(config.batch_size * config.bloem_ratio)
            n_ctw = config.batch_size - n_bloem
            
            bloem_batch = _get_bloem_batch(bloem_source, bloem_buffer, config, n_bloem)
            ctw_batch = ctw_gen()[:n_ctw] if n_ctw > 0 else torch.empty(0, config.seq_length)
            
            batch = torch.cat([bloem_batch, ctw_batch], dim=0)
        
        # Convert to Tinker format
        datums = batch_to_tinker_datums(batch)
        
        # Training step (remote GPU)
        # --------------------------
        
        fwd_bwd_future = training_client.forward_backward(
            datums, loss_fn="cross_entropy"
        )
        optim_step_future = training_client.optim_step(adam_params)
        
        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()
        
        # Compute metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in datums]
        train_nll = compute_mean_nll(train_logprobs, train_weights)
        
        # Update Bloem buffer with model samples (periodically)
        # -----------------------------------------------------
        
        if (step > 0 and step % config.bloem_update_interval == 0 and 
            config.data_source in ["bloem", "bloem_ctw"]):
            _update_bloem_with_model_samples(
                training_client, service_client, bloem_buffer, config, step
            )
        
        # Logging
        # -------
        
        metrics.update(
            step=step,
            train_nll=train_nll,
            learning_rate=current_lr,
            bloem_iterations=bloem_buffer.avg_iterations(),
            ctw_step=ctw_gen.step,
            time_total=time.time() - t_start,
            num_sequences=len(datums),
            num_tokens=sum(d.model_input.length for d in datums),
        )
        
        if step % config.log_every == 0:
            logger.info(
                f"Step {step}: NLL={train_nll:.4f}, LR={current_lr:.2e}, "
                f"Bloem iters={bloem_buffer.avg_iterations():.1f}"
            )
            ml_logger.log_metrics(metrics=metrics, step=step)
        
        # Checkpointing
        # -------------
        
        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={
                    "batch": step,
                    "bloem_buffer": bloem_buffer.get_state(),
                    "ctw_step": ctw_gen.step,
                },
            )
            logger.info(f"Checkpoint saved at step {step}")
    
    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={
            "batch": config.num_steps,
            "bloem_buffer": bloem_buffer.get_state(),
            "ctw_step": ctw_gen.step,
        },
    )
    
    ml_logger.close()
    logger.info("Training completed!")


def _get_bloem_batch(source: LSTMSourceGenerator, buffer: BloemBuffer,
                     config: BloemCTWConfig, 
                     batch_size: Optional[int] = None) -> torch.Tensor:
    """Generate batch from Bloem buffer with LSTM conditioning."""
    if batch_size is None:
        batch_size = config.batch_size
    
    # Randomly reweight LSTM
    weight_mult = random.uniform(*config.bloem_weight_mult_range)
    source.reset_and_scale(weight_mult)
    
    # Sample conditioning sequences from buffer
    cond, idx = buffer.sample(batch_size)
    
    # Generate new sequences conditioned on buffer
    seed = torch.randint(0, config.d_vocab, (batch_size, 1))
    new_seqs = source.sample_sequence(
        seed, config.seq_length,
        temperature=config.bloem_temperature,
        conditional=cond,
    )
    
    # Update buffer with new sequences
    buffer.update(idx, new_seqs)
    
    # Sample final batch from updated buffer
    batch, _ = buffer.sample(batch_size)
    return batch


def _update_bloem_with_model_samples(
    training_client,
    service_client,
    buffer: BloemBuffer,
    config: BloemCTWConfig,
    step: int,
    num_samples: int = 100,
):
    """
    Update Bloem buffer with samples from the fine-tuned model.
    
    This helps the buffer evolve toward sequences the model finds
    interesting, creating a curriculum that adapts to model capabilities.
    """
    try:
        # Save weights for sampling
        sampling_path = training_client.save_weights_for_sampler(
            name=f"step_{step}"
        ).result().path
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )
        
        sampling_params = types.SamplingParams(
            max_tokens=config.seq_length,
            temperature=config.bloem_temperature,
        )
        
        # Sample sequences
        new_seqs = []
        for _ in range(min(num_samples, buffer.size // 10)):
            # Start with random seed token
            prompt = types.ModelInput.from_ints(
                tokens=[random.randint(0, config.d_vocab - 1)]
            )
            response = sampling_client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()
            
            tokens = response.sequences[0].tokens
            if len(tokens) >= config.seq_length:
                new_seqs.append(tokens[:config.seq_length])
        
        # Update random buffer positions with model samples
        if new_seqs:
            idx = random.sample(range(buffer.size), len(new_seqs))
            seqs_tensor = torch.tensor(new_seqs, dtype=torch.long)
            buffer.update(idx, seqs_tensor)
            logger.debug(f"Updated buffer with {len(new_seqs)} model samples")
    
    except Exception as e:
        logger.warning(f"Failed to update buffer with model samples: {e}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
