import torch
from pytest import raises


def test_patch_torch_numpy():
    t = torch.ones(10, requires_grad=True)

    with raises(RuntimeError, match=r'Can\'t call numpy\(\) on Tensor that requires grad\.'):
        t.numpy()

    from clipppy.patches import torch_numpy
    assert (torch.ones(10, requires_grad=True).numpy() == 1).all()
