import torch
from ruamel import yaml as yaml

from .constructor import YAMLConstructor
from ..stochastic import stochastic


class PrefixedTensorYAMLConstructor(YAMLConstructor):
    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        if suffix == 'default':
            kwargs = {'dtype': torch.get_default_dtype(),
                      'device': torch._C._get_default_device(),
                      **kwargs}
        else:
            kwargs = {'dtype': getattr(torch, suffix), **kwargs}
        if not isinstance(kwargs['dtype'], torch.dtype):
            raise ValueError(f'In tag \'!torch:{suffix}\', \'{suffix}\' is not a valid torch.dtype.')

        return super().construct(torch.tensor, loader, node, **kwargs)


class PrefixedStochasticYAMLConstructor(YAMLConstructor):
    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        return super().construct(stochastic, loader, node, name=suffix)
