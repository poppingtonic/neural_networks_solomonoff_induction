#!/usr/bin/env python3
"""FastAI DataLoader wrappers for UTM and CTW data generators.

Provides fastai-compatible data loaders that wrap the existing data generators
for use with the fastai training framework.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from torch.utils.data import IterableDataset

from utm_dataset import IGNORE_INDEX


class UTMFastAIDataset(IterableDataset):
    """FastAI-compatible iterable dataset for UTM data."""
    
    def __init__(self, data_generator, batches_per_epoch: int = 100):
        """Initialize UTM dataset.
        
        Args:
            data_generator: UTM data generator instance
            batches_per_epoch: Number of batches to generate per epoch
        """
        super().__init__()
        self._dg = data_generator
        self._batches_per_epoch = batches_per_epoch
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate batches of (inputs, targets)."""
        for _ in range(self._batches_per_epoch):
            sequences, log_dict = self._dg.sample()
            sequences = np.asarray(sequences)
            tokens = np.argmax(sequences, axis=-1).astype(np.int64)
            
            # Handle loss mask for padding
            if 'loss_mask' in log_dict:
                loss_mask = np.asarray(log_dict['loss_mask']).astype(bool)
            else:
                loss_mask = np.zeros(tokens.shape, dtype=bool)
            
            targets = tokens.copy()
            targets[loss_mask] = IGNORE_INDEX
            
            # Convert to tensors
            x = torch.from_numpy(tokens)
            y = torch.from_numpy(targets)
            
            yield (x, y)


class CTWFastAIDataset(IterableDataset):
    """FastAI-compatible iterable dataset for CTW data."""
    
    def __init__(self, data_generator, batches_per_epoch: int = 100):
        """Initialize CTW dataset.
        
        Args:
            data_generator: CTW data generator instance
            batches_per_epoch: Number of batches to generate per epoch
        """
        super().__init__()
        self._dg = data_generator
        self._batches_per_epoch = batches_per_epoch
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate batches of (inputs, targets)."""
        for _ in range(self._batches_per_epoch):
            sequences, log_dict = self._dg.sample()
            sequences = np.asarray(sequences)
            tokens = np.argmax(sequences, axis=-1).astype(np.int64)
            
            # For CTW, we predict the sequence itself
            x = torch.from_numpy(tokens)
            y = x.clone()
            
            yield (x, y)


def create_utm_dataloaders(
    data_generator,
    batches_per_epoch: int = 100,
    device: Optional[str] = None,
) -> DataLoaders:
    """Create fastai DataLoaders for UTM data.
    
    Args:
        data_generator: UTM data generator instance
        batches_per_epoch: Number of batches per epoch
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        FastAI DataLoaders object
    """
    train_ds = UTMFastAIDataset(data_generator, batches_per_epoch)
    
    # Create train dataloader (no validation for now)
    train_dl = DataLoader(
        train_ds,
        batch_size=None,  # Dataset yields batches
        shuffle=False,
        pin_memory=True if device == 'cuda' else False,
    )
    
    # Use same dataloader for validation (or create separate with different seed)
    valid_dl = DataLoader(
        UTMFastAIDataset(data_generator, batches_per_epoch // 10),
        batch_size=None,
        shuffle=False,
        pin_memory=True if device == 'cuda' else False,
    )
    
    return DataLoaders(train_dl, valid_dl)


def create_ctw_dataloaders(
    data_generator,
    batches_per_epoch: int = 100,
    device: Optional[str] = None,
) -> DataLoaders:
    """Create fastai DataLoaders for CTW data.
    
    Args:
        data_generator: CTW data generator instance
        batches_per_epoch: Number of batches per epoch
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        FastAI DataLoaders object
    """
    train_ds = CTWFastAIDataset(data_generator, batches_per_epoch)
    
    # Create train dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=None,  # Dataset yields batches
        shuffle=False,
        pin_memory=True if device == 'cuda' else False,
    )
    
    # Create validation dataloader
    valid_dl = DataLoader(
        CTWFastAIDataset(data_generator, batches_per_epoch // 10),
        batch_size=None,
        shuffle=False,
        pin_memory=True if device == 'cuda' else False,
    )
    
    return DataLoaders(train_dl, valid_dl)
