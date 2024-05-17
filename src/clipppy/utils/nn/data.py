import random
from os import cpu_count
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset

import phytorchx


def multiprocess_batch(dataset, batch_size, num_workers=None,
                       shuffle=True, pin_memory=True, drop_last=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=(num_workers if num_workers is not None else min(batch_size, cpu_count())),
        shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last,
        worker_init_fn=lambda *args, **kwargs: torch.set_default_tensor_type(torch.FloatTensor),
        generator=torch.Generator(device=phytorchx.get_default_device()),
    )


def random_subsets(dataset, lengths, seed=None) -> Iterable[Subset]:
    random.seed(seed)
    indices = random.sample(range(len(dataset)), k=sum(lengths))
    return (Subset(dataset, indices[i-length:i]) for i in [0] for length in lengths for i in [i+length])
