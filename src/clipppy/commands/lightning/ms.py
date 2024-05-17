from __future__ import annotations

from abc import abstractmethod, ABC
from functools import partialmethod
from math import log
from typing import Any, Sequence, TypedDict, Mapping, TypeVar, Generic, Callable, Union, TYPE_CHECKING, Iterable, cast
from warnings import warn

# import attr
import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from torch import Tensor, Size, LongTensor
from torch.nn import CrossEntropyLoss, Module
from torchmetrics.classification import MulticlassROC
from typing_extensions import Unpack

from .callbacks import DiagnosticFigureMixin
from .command import AbstractLightningSBICommand
from .config import Config
from .utils import if_not_sanity_checking
from ...utils import Sentinel
from ...utils.importing.attr import attr
from ...utils.metrics import ClassificationMetric

_T = TypeVar('_T')


class _MMS_OutT(TypedDict):
    loss: Tensor
    pred: Tensor
    target: LongTensor


class MultiModelSelection(AbstractLightningSBICommand[CrossEntropyLoss], Generic[_T]):
    class _KwargsT(AbstractLightningSBICommand[CrossEntropyLoss]._KwargsT, total=False):
        models: Sequence[Any]
        net: Union[Module, Callable[[_T], Tensor]]

    if TYPE_CHECKING:
        # noinspection PyMissingConstructor
        def __init__(self, **kwargs: Unpack[_KwargsT]): ...

    loss_config: Config = Config(CrossEntropyLoss(), Sentinel.no_call)

    net: Union[Module, Callable[[_T], Tensor]]
    models: Sequence[Any]

    @property
    def output_size(self):
        return len(self.models)

    def forward(self, x: _T) -> Tensor:
        return self.net(x)

    def _step(self, batch: tuple[_T, Tensor], *args, _log_name, **kwargs) -> _MMS_OutT:
        pred, target = self.forward(batch[0]), batch[1]
        loss: Tensor = self.lossfunc(pred, target) / log(pred.shape[-1])
        self.log_loss(loss, loss_name=getattr(self, _log_name))
        return {'loss': loss, 'pred': pred, 'target': target}

    training_step = partialmethod(_step, _log_name='_loss_name')
    validation_step = partialmethod(_step, _log_name='_val_name')

    _ckpt_state_name = 'clipppy_state'

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint[self._ckpt_state_name] = self.__getstate__()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if self._ckpt_state_name in checkpoint:
            self.__setstate__(checkpoint[self._ckpt_state_name])
        else:
            warn(f'"{self._ckpt_state_name}" not found in checkpoint.', RuntimeWarning)


class AutoregressiveMultiModelSelection(MultiModelSelection):
    models_shape: Size

    @property
    def res_sizes(self):
        return tuple(np.cumprod(self.models_shape))

    @property
    def output_size(self):
        return sum(self.res_sizes)

    def forward(self, x):
        res: Tensor = self.net(x)
        return sum(
            r.unflatten(-1, (*self.models_shape[:i], nmods, *(1,)*(len(self.models_shape)-i-1)))
            for i, (r, nmods) in enumerate(zip(res.split(self.res_sizes, -1), self.models_shape))
        ).flatten(-len(self.models_shape))


def confusion_matrix(pred: Tensor, target: LongTensor):
    return pred.new_zeros(2*pred.shape[-1:]).scatter_reduce_(
        0, target.unsqueeze(-1).expand(*target.shape, pred.shape[-1]),
        pred.softmax(-1), 'mean')


class MSCallback(ABC):
    @abstractmethod
    def __call__(self, trainer: Trainer, pl_module: MultiModelSelection, preds: Tensor, targets: LongTensor): ...


@attr.s(kw_only=True)
class FuzzyConfusionCallback(DiagnosticFigureMixin, MSCallback):
    labels: Sequence[str] = None
    heatmap_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(
        annot=True, vmin=0, cbar=False, square=True, cmap='viridis'
    ).__or__)
    tick_params: Mapping[str, Any] = attr.field(default={}, converter=dict(
        top=True, bottom=False, labeltop=True, labelbottom=False
    ).__or__)
    xtick_params: Mapping[str, Any] = attr.field(default={}, converter=dict(labelrotation=0).__or__)
    ytick_params: Mapping[str, Any] = attr.field(default={}, converter=dict(labelrotation=0).__or__)
    ax_params: Mapping[str, Any] = attr.field(default={}, converter=dict(xlabel='predicted class', ylabel='true class').__or__)

    def __call__(self, trainer: Trainer, pl_module: MultiModelSelection, preds: Tensor, targets: LongTensor):
        if trainer.sanity_checking:
            return

        from matplotlib import pyplot as plt
        import seaborn as sns

        labels = self.labels or pl_module.models

        fig = plt.figure()

        ax = sns.heatmap(confusion_matrix(preds, targets).numpy(force=True), **{
            'xticklabels': labels, 'yticklabels': labels, **self.heatmap_kwargs})
        ax.tick_params(**self.tick_params)
        ax.tick_params(axis='x', **self.xtick_params)
        ax.tick_params(axis='y', **self.ytick_params)
        ax.set(**self.ax_params)

        self.log_figure('confusion', fig, trainer.global_step)
        plt.close(fig)


@attr.s(kw_only=True)
class ROCCallback(DiagnosticFigureMixin, MSCallback):
    roc_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(score=True).__or__)

    def __call__(self, trainer: Trainer, pl_module: MultiModelSelection, preds: Tensor, targets: LongTensor):
        roc = MulticlassROC(len(pl_module.models))
        roc.update(preds, targets)

        from matplotlib import pyplot as plt

        fig = cast(plt.Figure, roc.plot(**{'labels': pl_module.models, **self.roc_kwargs})[0])
        self.log_figure('roc', fig, trainer.global_step)
        plt.close(fig)


class MSCallbackCollection(Callback):
    def __init__(self, *callbacks: MSCallback):
        self.callbacks = callbacks
        self.metric = ClassificationMetric()

    @if_not_sanity_checking
    def on_validation_epoch_start(self, *args, **kwargs):
        self.metric.reset()

    @if_not_sanity_checking
    def on_validation_batch_end(self, trainer: Trainer, pl_module: MultiModelSelection, outputs: _MMS_OutT, *args, **kwargs):
        self.metric.update(outputs['pred'], outputs['target'])

    @if_not_sanity_checking
    def on_validation_epoch_end(self, *args, **kwargs):
        kwargs['preds'] = self.metric.preds.compute()
        kwargs['targets'] = self.metric.targets.compute()

        for cb in self.callbacks:
            cb(*args, **kwargs)