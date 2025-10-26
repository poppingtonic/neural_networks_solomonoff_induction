from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset

IGNORE_INDEX = -100


class UTMIterableDataset(IterableDataset):
    def __init__(self, data_generator, batches_per_epoch: int):
        super().__init__()
        self._dg = data_generator
        self._batches_per_epoch = batches_per_epoch

    def __iter__(self):
        for _ in range(self._batches_per_epoch):
            sequences, log_dict = self._dg.sample()
            sequences = np.asarray(sequences)
            tokens = np.argmax(sequences, axis=-1).astype(np.int64)
            if 'loss_mask' in log_dict:
                loss_mask = np.asarray(log_dict['loss_mask']).astype(bool)
            else:
                loss_mask = np.zeros(tokens.shape, dtype=bool)
            targets = tokens.copy()
            targets[loss_mask] = IGNORE_INDEX
            yield torch.from_numpy(tokens), torch.from_numpy(targets)
