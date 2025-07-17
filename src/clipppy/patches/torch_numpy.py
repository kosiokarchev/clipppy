from __future__ import annotations

import torch


def _torch_numpy(self: torch.Tensor, **kwargs):
    self = self.detach().cpu()
    return (self.float() if self.dtype in (torch.float16, torch.bfloat16) else self)._numpy(**kwargs)


if not hasattr(torch.Tensor, '_numpy'):
    torch.Tensor._numpy = torch.Tensor.numpy
    torch.Tensor.numpy = _torch_numpy
