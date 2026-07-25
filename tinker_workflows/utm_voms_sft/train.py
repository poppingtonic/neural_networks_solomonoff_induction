import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tinker
from tinker import TensorData
from tinker.types import AdamParams

# Import from tinker_cookbook if available
try:
    from tinker_cookbook.checkpoint_utils import get_last_checkpoint, save_checkpoint
    from tinker_cookbook.cli_utils import check_log_dir
    from tinker_cookbook.tokenizer_utils import get_tokenizer
except ImportError:
    # Fallback implementations if tinker_cookbook is not available
    def get_tokenizer(model_name: str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def get_last_checkpoint(log_dir: str, required_key: str = "state_path"):
        checkpoint_path = os.path.join(log_dir, "checkpoints.jsonl")
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, 'r') as f:
            checkpoints = [json.loads(line) for line in f if line.strip()]
        checkpoints = [c for c in checkpoints if required_key in c]
        return checkpoints[-1] if checkpoints else None
    
    def save_checkpoint(training_client, name: str, log_path: str, loop_state: dict, **kwargs):
        checkpoint_path = os.path.join(log_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save the model state
        saved = training_client.save_state(name=name).result()
        
        # Update checkpoints file
        checkpoint_info = {
            "name": name,
            "step": loop_state.get("global_step", 0),
            "stage": loop_state.get("current_stage", ""),
            "state_path": saved.path,
            **loop_state
        }
        
        checkpoints_file = os.path.join(log_path, "checkpoints.jsonl")
        with open(checkpoints_file, "a") as f:
            f.write(json.dumps(checkpoint_info) + "\n")
        
        return saved
    
    def check_log_dir(log_dir: str, behavior_if_exists: str = "ask"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            return
            
        if behavior_if_exists == "delete":
            import shutil
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)
        elif behavior_if_exists == "ask":
            while True:
                resp = input(f"Log directory {log_dir} exists. [D]elete/[R]esume/[Q]uit? ").lower()
                if resp == 'd':
                    import shutil
                    shutil.rmtree(log_dir)
                    os.makedirs(log_dir)
                    break
                elif resp == 'r':
                    break
                elif resp == 'q':
                    sys.exit(0)

logger = logging.getLogger(__name__)

from neural_networks_solomonoff_induction.data import ctw_data_generator as ctw_dg
from neural_networks_solomonoff_induction.data import utm_data_generator as utm_dg
from neural_networks_solomonoff_induction.data import utms as utms_lib


@dataclass
class AdamConfig:
    learning_rate: float
    beta1: float
    beta2: float
    eps: float


@dataclass
class StageConfig:
    name: str
    steps: int
    utm: Optional[Dict[str, Any]] = None
    ctw: Optional[Dict[str, Any]] = None


@dataclass
class Config:
    base_model: str
    lora_rank: int
    adam: AdamConfig
    batch_size: int
    seq_length: int
    save_every: int
    log_every: int
    output_dir: str
    seed: int
    stages: List[StageConfig]
    resume_from_state: str = ""


def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    """Append a JSON object to a JSONL file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = json.load(f)
    adam = AdamConfig(**raw["adam"])
    stages = [StageConfig(**s) for s in raw["stages"]]
    return Config(
        base_model=raw["base_model"],
        lora_rank=int(raw["lora_rank"]),
        adam=adam,
        batch_size=int(raw["batch_size"]),
        seq_length=int(raw["seq_length"]),
        save_every=int(raw["save_every"]),
        log_every=int(raw["log_every"]),
        output_dir=raw["output_dir"],
        seed=int(raw.get("seed", 1)),
        stages=stages,
        resume_from_state=raw.get("resume_from_state", ""),
    )


def _pad_to_length(ids: List[int], length: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = ids[:length]
    weights = np.ones(len(ids), dtype=np.float32)
    if len(ids) > 0:
        weights[0] = 0.0
    if len(ids) < length:
        pad_len = length - len(ids)
        ids = ids + [pad_id] * pad_len
        pad_w = np.zeros(pad_len, dtype=np.float32)
        weights = np.concatenate([weights, pad_w], axis=0)
    return np.asarray(ids, dtype=np.int64), weights.astype(np.float32)


def build_datum(tokens_1d: np.ndarray, weights_1d: np.ndarray) -> tinker.Datum:
    model_input = tinker.ModelInput.from_ints(tokens_1d.tolist())
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(tokens_1d.astype(np.int64)),
            "weights": TensorData.from_numpy(weights_1d.astype(np.float32)),
        },
    )


def make_utm_generator(cfg: Config, stage: StageConfig) -> utm_dg.UTMDataGenerator:
    rng = np.random.default_rng(seed=cfg.seed)
    program_sampler = utms_lib.FastSampler(rng=rng)
    utm = utms_lib.BrainPhoqueUTM(program_sampler)

    if stage.utm is None:
        raise ValueError("UTM stage missing 'utm' config")

    return utm_dg.UTMDataGenerator(
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        rng=rng,
        utm=utm,
        memory_size=int(stage.utm.get("memory_size", 10)),
        maximum_steps=int(stage.utm.get("maximum_steps", 100)),
        tokenizer=utm_dg.Tokenizer.ASCII,
        maximum_program_length=int(stage.utm.get("maximum_program_length", 100)),
    )


def make_ctw_generator(cfg: Config, stage: StageConfig) -> ctw_dg.CTWGenerator:
    rng = np.random.default_rng(seed=cfg.seed)
    if stage.ctw is None:
        raise ValueError("CTW stage missing 'ctw' config")
    return ctw_dg.CTWGenerator(
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        rng=rng,
        max_depth=int(stage.ctw.get("max_depth", 5)),
        with_contexts=False,
    )


def utm_batch_to_token_ids(
    generator: utm_dg.UTMDataGenerator,
    tokenizer,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences, log_dict = generator.sample()
    sequences = np.asarray(sequences)
    ascii_vals = np.argmax(sequences, axis=-1).astype(np.uint8)

    loss_mask = np.asarray(log_dict.get("loss_mask", np.zeros(ascii_vals.shape, dtype=np.uint8))).astype(bool)
    batch_tokens: List[np.ndarray] = []
    batch_weights: List[np.ndarray] = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for i in range(ascii_vals.shape[0]):
        valid = ascii_vals[i][~loss_mask[i]]
        text = "".join(chr(int(x)) for x in valid.tolist())
        ids = tokenizer.encode(text, add_special_tokens=False)
        tok_1d, w_1d = _pad_to_length(ids, seq_length, pad_id)
        batch_tokens.append(tok_1d)
        batch_weights.append(w_1d)

    return np.stack(batch_tokens, axis=0), np.stack(batch_weights, axis=0)


def ctw_batch_to_token_ids(
    generator: ctw_dg.CTWGenerator,
    tokenizer,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences, _ = generator.sample()
    sequences = np.asarray(sequences)
    bits = np.argmax(sequences, axis=-1).astype(np.int64)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batch_tokens: List[np.ndarray] = []
    batch_weights: List[np.ndarray] = []
    for i in range(bits.shape[0]):
        text = "".join("1" if b == 1 else "0" for b in bits[i].tolist())
        ids = tokenizer.encode(text, add_special_tokens=False)
        tok_1d, w_1d = _pad_to_length(ids, seq_length, pad_id)
        batch_tokens.append(tok_1d)
        batch_weights.append(w_1d)
    return np.stack(batch_tokens, axis=0), np.stack(batch_weights, axis=0)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with the specified log level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log")
        ]
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model on UTM and CTW data.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Check for API key
    if os.getenv("TINKER_API_KEY") is None:
        logger.error("TINKER_API_KEY is not set in the environment")
        sys.exit(1)
    
    # Set up directories
    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    
    # Handle existing log directory
    if args.resume:
        logger.info(f"Resuming training from {output_dir}")
        check_log_dir(str(output_dir), "resume")
    else:
        logger.info(f"Starting new training run in {output_dir}")
        check_log_dir(str(output_dir), "delete" if not args.resume else "resume")
    
    # Set up paths
    metrics_path = output_dir / "metrics.jsonl"
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer for {cfg.base_model}")
    tokenizer = get_tokenizer(cfg.base_model)

    # Initialize Tinker clients
    logger.info("Initializing Tinker service client")
    service_client = tinker.ServiceClient()
    
    # Check for existing checkpoint to resume from
    resume_from = None
    if args.resume:
        last_ckpt = get_last_checkpoint(str(output_dir))
        if last_ckpt:
            resume_from = last_ckpt.get("state_path")
            logger.info(f"Resuming from checkpoint: {resume_from}")
    
    # Initialize training client
    logger.info(f"Initializing LoRA training client with rank {cfg.lora_rank}")
    training_client = service_client.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
    )
    
    # Load state if resuming
    if resume_from:
        logger.info(f"Loading model state from {resume_from}")
        training_client.load_state(resume_from)
    elif cfg.resume_from_state:
        logger.info(f"Loading model state from config: {cfg.resume_from_state}")
        training_client.load_state(cfg.resume_from_state)
    
    # Set up optimizer
    optim_params = AdamParams(
        learning_rate=cfg.adam.learning_rate,
        beta1=cfg.adam.beta1,
        beta2=cfg.adam.beta2,
        eps=cfg.adam.eps,
    )
    
    # Initialize training state
    loop_state = {
        "global_step": 0,
        "current_stage": "",
        "stage_step": 0,
        "total_steps": sum(s.steps for s in cfg.stages),
    }
    # Training loop
    for stage in cfg.stages:
        loop_state["current_stage"] = stage.name
        loop_state["stage_step"] = 0
        
        logger.info(f"Starting stage: {stage.name} for {stage.steps} steps")
        
        # Set up data generator for this stage
        if stage.name == "utm":
            gen = make_utm_generator(cfg, stage)
            batch_fn = lambda: utm_batch_to_token_ids(gen, tokenizer, cfg.seq_length)
        elif stage.name == "ctw":
            gen = make_ctw_generator(cfg, stage)
            batch_fn = lambda: ctw_batch_to_token_ids(gen, tokenizer, cfg.seq_length)
        else:
            raise ValueError(f"Unknown stage name: {stage.name}")

        # Stage training loop
        for local_step in range(stage.steps):
            loop_state["stage_step"] = local_step
            loop_state["global_step"] = loop_state.get("global_step", 0)
            
            # Get batch
            try:
                batch_tokens, batch_weights = batch_fn()
                data = [
                    build_datum(batch_tokens[i], batch_weights[i])
                    for i in range(batch_tokens.shape[0])
                ]
                
                # Forward/backward pass
                fwd_out = training_client.forward_backward(
                    data, 
                    loss_fn="cross_entropy"
                ).result()
                
                # Optimization step
                optim_out = training_client.optim_step(optim_params).result()
                
                # Log metrics
                if loop_state["global_step"] % cfg.log_every == 0:
                    row = {
                        "step": loop_state["global_step"],
                        "stage": stage.name,
                        "stage_step": local_step,
                        "learning_rate": optim_params.learning_rate,
                    }
                    
                    # Add forward pass metrics
                    if fwd_out.metrics:
                        row.update({
                            f"train/{k}": float(v) 
                            for k, v in fwd_out.metrics.items()
                        })
                    
                    # Add optimizer metrics
                    if optim_out.metrics:
                        row.update({
                            f"optim/{k}": float(v)
                            for k, v in optim_out.metrics.items()
                        })
                    
                    # Log to file and console
                    append_jsonl(str(metrics_path), row)
                    logger.info(
                        f"Step {loop_state['global_step']} | "
                        f"Stage: {stage.name} ({local_step}/{stage.steps}) | "
                        f"Loss: {fwd_out.metrics.get('loss', 'N/A'):.4f} | "
                        f"LR: {optim_params.learning_rate:.2e}"
                    )
                
                # Save checkpoint
                if (loop_state["global_step"] + 1) % cfg.save_every == 0:
                    ckpt_name = f"step_{loop_state['global_step'] + 1:06d}_stage_{stage.name}"
                    saved = save_checkpoint(
                        training_client,
                        name=ckpt_name,
                        log_path=str(output_dir),
                        loop_state=loop_state
                    )
                    logger.info(f"Saved checkpoint: {saved.path}")
                
                loop_state["global_step"] += 1
                
            except Exception as e:
                logger.error(f"Error during training step {loop_state['global_step']}", exc_info=True)
                # Save checkpoint on error to allow resuming
                if loop_state["global_step"] > 0:
                    ckpt_name = f"error_step_{loop_state['global_step']:06d}_stage_{stage.name}"
                    try:
                        saved = save_checkpoint(
                            training_client,
                            name=ckpt_name,
                            log_path=str(output_dir),
                            loop_state=loop_state
                        )
                        logger.info(f"Saved recovery checkpoint: {saved.path}")
                    except Exception as save_error:
                        logger.error("Failed to save recovery checkpoint", exc_info=True)
                raise


if __name__ == "__main__":
    main()
