from __future__ import annotations

from functools import partialmethod
from itertools import chain
from typing import Mapping, Union, Generic

import attr
from more_itertools import always_iterable
from torch import Size

from .command import LightningSBICommand
from .config import Config
from .loss import NPELoss
from ...sbi._typing import SBIBatch, _KT, _MultiKT, _SBIObsT, _MultiMappingT
from ...sbi.nn import _HeadOoutT
from ...sbi.nn.npe import NPEResult
from ...utils import Sentinel


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class NPE(LightningSBICommand[NPELoss, Union[NPEResult, Mapping[_MultiKT, NPEResult]], _HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    loss_config: Config[NPELoss] = attr.ib(factory=lambda: Config(NPELoss(), Sentinel.no_call))

    def _step(self, batch: SBIBatch, *args, _log_name, **kwargs):
        ret = self.lossfunc(self(batch))
        self.log_loss(
            ret.loss, tree if isinstance(tree := ret.unflatten(), Mapping) else None,
            loss_name=getattr(self, _log_name)
        )
        return ret.loss

    training_step = partialmethod(_step, _log_name='_loss_name')
    validation_step = partialmethod(_step, _log_name='_val_name')

    def posterior(self, obs: _SBIObsT):
        _, x = self.head({}, obs)

        from ...sbi.nn import MultiSBITail
        from ...sbi.nn.npe import NPETail

        if isinstance(self.tail, NPETail):
            return self.tail.get_dist(x)
        elif isinstance(self.tail, MultiSBITail):
            return {key: tail.get_dist(x) for key, tail in self.tail.tails.items()}
        raise TypeError

    def sample(self, obs: _SBIObsT, shape=Size()) -> _MultiMappingT:
        return dict(chain.from_iterable(
            zip(always_iterable(key), (s.unsqueeze(-1) if not d.event_shape else s).unbind(-1))
            for key, d in self.posterior(obs).items() for s in [d.sample(shape)]
        ))
