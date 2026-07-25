"""
Tinker workflows for Solomonoff Induction approximation.

Available recipes:
- bloem_ctw_sft: Fine-tune LLMs on Bloem+CTW universal data
- train: Sequential UTM + VOMS fine-tuning
"""

from .bloem_ctw_sft import (
    BloemCTWConfig,
    BloemBuffer,
    BloemSourceConfig,
    LSTMSourceGenerator,
    CTWGenerator,
    CTWNode,
    batch_to_tinker_datums,
)

__all__ = [
    "BloemCTWConfig",
    "BloemBuffer",
    "BloemSourceConfig", 
    "LSTMSourceGenerator",
    "CTWGenerator",
    "CTWNode",
    "batch_to_tinker_datums",
]
