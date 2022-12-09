from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from functools import update_wrapper, WRAPPER_ASSIGNMENTS
from typing import Any, Callable, ClassVar, Container, Generic, Literal, Mapping, Type, TypeVar, Union

import forge
from torch.utils.data import DataLoader, Dataset

from ....sbi._typing import _OptimizerT, _SchedulerT, DEFAULT_LOSS_NAME
from ....utils import _T, merge_if_not_skip, Sentinel


_C = TypeVar('_C')
__T = TypeVar('__T')


def params_except(clbl, exclude: Container[str]) -> OrderedDict[str, forge.FParameter]:
    return OrderedDict(filter(
        lambda keyval: keyval[0] not in exclude,
        forge.fsignature(clbl).parameters.items()))


class BaseConfig(Generic[_T]):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> _T: ...



@dataclass
class Config(BaseConfig[_T], Generic[_T]):
    cls: Union[Type[_T], Callable[..., _T], _T] = lambda *args, **kwargs: None
    kwargs: Union[Mapping[str, Any], Literal[Sentinel.no_call]] = field(default_factory=dict)

    def instantiate(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> _T:
        return (self.cls if self.kwargs is Sentinel.no_call
                else self.instantiate(*args, **merge_if_not_skip(self.kwargs, kwargs)))

    _wrap_cls_params_except: ClassVar[Container[str]] = {'cls', 'kwargs'}
    _wrap_params_except: ClassVar[Container[str]] = set()

    @classmethod
    def wrap(cls, clbl):
        return update_wrapper(forge.sign(**(
            clbl_kwargs := params_except(clbl, cls._wrap_params_except)
        ), **params_except(cls, cls._wrap_cls_params_except))(
            lambda **kwargs: cls(clbl, {
                key: kwargs.pop(key) for key in clbl_kwargs.keys()
                if key in kwargs
            }, **kwargs)), clbl, assigned=tuple(set(WRAPPER_ASSIGNMENTS) - {'__annotations__'}))


class DatasetConfig(Config[Dataset]):
    pass


@dataclass
class DataLoaderConfig(Config[DataLoader]):
    cls: Union[Type[DataLoader], Callable[[...], DataLoader], DataLoader] = DataLoader


class OptimizerConfig(Config[_OptimizerT]):
    pass


@dataclass
class BaseSchedulerConfig(BaseConfig[_SchedulerT], Generic[_SchedulerT], ABC):
    interval: str = 'step'
    frequency: int = 1
    monitor: str = DEFAULT_LOSS_NAME
    strict: bool = True
    name: str = None

    @abstractmethod
    def _scheduler(self, optimizer: _OptimizerT, **kwargs): ...

    def __call__(self, optimizer: _OptimizerT, **kwargs):
        return (res := self._scheduler(optimizer, **kwargs)) and {
            'scheduler': res, **{key.name: getattr(self, key.name) for key in
                                 set(fields(SchedulerConfig)) - set(fields(Config))}
        }


@dataclass
class SchedulerConfig(BaseSchedulerConfig[_SchedulerT], Config[_SchedulerT], Generic[_SchedulerT]):
    _wrap_params_except = {'optimizer'}

    def _scheduler(self, optimizer: _OptimizerT, **kwargs):
        return super(BaseSchedulerConfig, self).__call__(optimizer, **kwargs)
