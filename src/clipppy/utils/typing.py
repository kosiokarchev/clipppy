import sys
import typing as tp

import torch
from pyro import distributions as dist
from pyro.poutine.indep_messenger import CondIndepStackFrame


__all__ = '_T', '_KT', '_VT', '_Tout', '_Tin', '_Site', '_Model', '_Guide'


_T = tp.TypeVar('_T')
_KT = tp.TypeVar('_KT')
_VT = tp.TypeVar('_VT')
_Tin = tp.TypeVar('_Tin')
_Tout = tp.TypeVar('_Tout')


if sys.version_info < (3, 8):
    def TypedDict(name: str, fields: tp.Dict[str, tp.Any], total: bool = True):
        return tp.NewType(name, tp.Dict[str, tp.Any])
    tp.TypedDict = TypedDict


    class Literal(type):
        @classmethod
        def __getitem__(mcs, item):
            typ = type(f'{mcs.__name__}[{repr(item)[1:-1]}]', (mcs,), {'__origin__': mcs})
            typ.__args__ = item
            return typ
    tp.Literal = Literal

    tp.get_origin = lambda generic: getattr(generic, '__origin__', None)
    tp.get_args = lambda generic: getattr(generic, '__args__', ())


_Site = tp.TypedDict('_Site', {
    'name': str, 'fn': dist.TorchDistribution, 'mask': torch.Tensor,
    'value': torch.Tensor, 'type': str, 'infer': dict,
    'cond_indep_stack': tp.Iterable[CondIndepStackFrame]
}, total=False)
_Model = tp.NewType('_Model', tp.Callable)
_Guide = tp.NewType('_Guide', tp.Callable)
