from __future__ import annotations

import torch

if not hasattr(torch.Tensor, '_numpy'):
    torch.Tensor._numpy = torch.Tensor.numpy
    torch.Tensor.numpy = lambda self: self.detach().cpu()._numpy()
