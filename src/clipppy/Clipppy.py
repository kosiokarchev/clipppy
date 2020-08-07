import inspect
import typing as tp
from abc import abstractmethod, ABC
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from pyro.infer import Trace_ELBO, SVI

# TODO: better IPython checking
try:
    from IPython import get_ipython
    if get_ipython() is None:
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from .globals import register_globals, init_msgr, _Model, _Guide, noop, dict_union, depoutine
from .guide import Guide


__all__ = ('Clipppy', 'Fit', 'Mock', 'Command')


class ProxyDict(dict):
    def __init__(self, obj, keys):
        super().__init__()
        self._keys = keys
        self.obj = obj

    def keys(self):
        return self._keys

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __getitem__(self, item):
        return getattr(self.obj, item)


class Command(ABC):
    """
    An abstract base class for commands.

    Defines an interface that allows manipulating parameters of the command in
    three distinct ways. Firstly, the *class* object itself provides the
    defaults. Next, when an instance is created, the constructor takes
    arbitrary keyword arguments that will be set as *instance* attributes
    overriding the defaults. Finally, when a class instance is called, any
    possible parameters that are included in the ``**kwargs`` parameter are
    extracted and set on the instance before execution of the `forward` method,
    which should be overridden to provide the actual command implementation.

    Commands also support "binding" keyword parameters in much the same way
    as `functools.partial`, through the `boundkwargs` property. The provided
    keywords are then always included in the `forward` call, if the forward
    call has explicit parameters with the same names.

    Attributes
    ----------
    boundkwargs
        A dictionary of values to be forwarded to each call of `forward`,
        if there are exact name matches in its signature.
    """

    no_call = object()
    """Special value to be used to indicate that instead of calling an object,
       it should be returned as is."""

    @property
    @lru_cache()
    def attr_names(self) -> tp.List[str]:
        return list(tp.get_type_hints(type(self)).keys())

    def setattr(self, kwargs: tp.Dict[str, tp.Any]):
        for key in list(key for key in kwargs if key in self.attr_names):
            setattr(self, key, kwargs.pop(key))
        return kwargs

    def __init__(self, **kwargs):
        self.setattr(kwargs)
        self.boundkwargs: tp.Dict = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        oldkwargs = {name: getattr(self, name) for name in self.attr_names}
        kwargs = self.setattr(kwargs)
        allowed = inspect.signature(self.forward).parameters
        try:
            ret = self.forward(*args, **dict_union(
                {key: value for key, value in self.boundkwargs.items() if key in allowed},
                kwargs))
        finally:
            self.setattr(oldkwargs)

        return ret

    def __getattribute__(self, name):
        # Needed to avoid binding functions that are saved as properties
        # upon attribute access. Properties should be annotated!
        cls = type(self)
        if name != '__dict__' and name not in self.__dict__ and name in tp.get_type_hints(cls):
            return getattr(cls, name)
        return super().__getattribute__(name)

    # Common attributes:
    plate_stack: tp.Union[tp.Iterable[int], tp.ContextManager] = nullcontext()
    """A stack of plates or an iterable of ints.

       Either one or multiple plates (as returned by `pyro.plate
       <pyro.primitives.plate>` or `pyro.plate_stack
       <pyro.primitives.plate_stack>`) or an iterable of ints that
       will be converted to a stack of plates (named ``plate_0``, etc. and
       aligned to ``rightmost_dim = -1``) for batch mock generation.
    """

    @property
    def plate(self) -> tp.Union[tp.ContextManager, pyro.primitives.plate]:
        return (self.plate_stack if isinstance(self.plate_stack, tp.ContextManager)
                else pyro.plate_stack('plate', self.plate_stack))


class Fit(Command):
    """Fit using ELBO maximisation."""

    n_steps: int = 1000
    """Run at most ``n_steps`` steps."""

    min_steps: int = 1
    """Run at least ``min_steps`` steps."""

    n_write: int = -1
    """Save the guide each ``n_write`` steps."""

    lr: float = 1e-3
    """Learning rate (passed to the optimizer)."""

    conv_th: float = -float('inf')
    """Convergence threshold. Should be positive or ``-float('inf')`` to turn
       off convergence-based termination.

       See `converged`."""

    avgwindow: int = 100
    """Number of most recents steps to use for calculating various
       averaged quantities (average rate, mean loss...)."""

    optimizer_cls: tp.Union[tp.Type[pyro.optim.PyroOptim],
                            tp.Callable[[tp.Dict], pyro.optim.PyroOptim]]\
        = pyro.optim.Adam
    """The class of `PyroOptim <pyro.optim.optim.PyroOptim>` to
       instantiate (or a function that acts like it)."""

    optimizer_args: tp.Optional[tp.Dict] = {}
    """The (keyword!) arguments to pass to ``optimizer_cls``.

       Will be updated with the standalone ``lr`` passed. Defaults are
       ``{'amsgrad': False, 'weight_decay': 0.}``."""

    loss_cls: tp.Union[tp.Type[pyro.infer.ELBO], tp.Callable[..., torch.Tensor]]\
        = Trace_ELBO
    """The loss class to instantiate."""

    loss_args: tp.Dict = {}
    """Arguments for the ``loss_cls`` constructor.

       Pass the special value `Command.no_call` to avoid instantiating
       ``load_cls`` and use it directly."""

    @property
    def optimizer(self) -> pyro.optim.PyroOptim:
        """Construct an optimizer as ``optimizer_cls(optimizer_args)``."""
        return self.optimizer_cls(dict_union(
            {'amsgrad': False, 'weight_decay': 0.}, self.optimizer_args, {'lr': self.lr}
        ))

    @property
    def lossfunc(self) -> tp.Union[pyro.infer.ELBO, tp.Callable[..., torch.Tensor]]:
        """
        Construct a loss as ``loss_cls(**loss_args)`` or simply return
        `loss_cls` if `loss_args` is `Command.no_call`.
        """
        return (self.loss_cls if self.loss_args is self.no_call
                else self.loss_cls(**self.loss_args))

    def converged(self, slope: float, windowed_losses: tp.Iterable[float] = None):
        """
        Indicate whether the fit is considered converged.

        This implementation returns `True` if the averaged slope ``slope``,
        defined as the fitted slope of losses over the last `avgwindow`
        iterations, divided by the learning rate, is shallower than `conv_th`,
        which should be given as positive. If the ``slope`` is positive, this
        will always be true, so turning this check off requires setting
        ``conv_th = -float('inf')``.

        This method can be overridden in subclasses to provide more
        sophisticated checks.

        """
        return slope > -self.conv_th

    def forward(self, model: _Model, guide: _Guide, *args, **kwargs):
        svi = SVI(model, guide, self.optimizer, self.lossfunc)

        xavg = np.arange(self.avgwindow)
        slope = np.NaN
        losses = []
        minloss = None
        try:
            with tqdm(range(self.n_steps)) as tq:
                for i in tq:
                    loss = svi.step(*args, **kwargs)
                    losses.append(loss)

                    windowed_losses = losses[-self.avgwindow:]
                    avgloss = np.mean(windowed_losses)
                    minloss = min(minloss, loss) if minloss is not None else loss

                    if len(windowed_losses) >= self.avgwindow:
                        slope = np.polyfit(xavg, windowed_losses, deg=1)[0] / self.lr

                    tq.set_postfix_str(f'loss={loss:.3f} (avg={avgloss:.3f}, min={minloss:.3f}, slope={slope:.3e})')

                    if 0 < self.min_steps <= i and self.converged(slope, windowed_losses):
                        break
        except KeyboardInterrupt:
            pass


class SamplingCommand(Command, ABC):
    savename: str = None
    """Filename to save the sample to (or `None` to skip saving)."""

    conditioning: bool = False
    """Whether to retain any conditioning already applied to the model.
       If a false value, `pyro.poutine.handlers.uncondition` will
       be applied to ``model`` before evaluating."""

    @property
    def uncondition(self):
        return pyro.poutine.uncondition() if not self.conditioning else nullcontext()


class Mock(SamplingCommand):
    """Generate mock data from the model prior."""
    def forward(self, model: _Model, *args, **kwargs) -> pyro.poutine.Trace:
        with pyro.poutine.trace() as trace, init_msgr, self.plate, self.uncondition:
            model(*args, **kwargs)

        if self.savename:
            torch.save(trace.trace, self.savename)

        return trace.trace


class PPD(SamplingCommand):
    """Sample from the guide and optionally generate the corresponding data."""

    observations: bool = True
    """Sample also the observations corresponding to the drawn parameters."""

    guidefile: str = None
    """File to load a guide from, or `None` to use the guide in the config."""

    def forward(self, model: _Model, guide: Guide, *args, **kwargs)\
            -> tp.TypedDict('ppd', {'guide_trace': pyro.poutine.Trace,
                                    'model_trace': pyro.poutine.Trace}, total=False):
        # TODO: better guide loading
        if self.guidefile is not None:
            guide = torch.load(self.guidefile)

        guide_is_trainable = hasattr(guide, 'training')

        if guide_is_trainable:
            was_training = guide.training
            guide.eval()
        with pyro.poutine.trace() as guide_tracer, self.plate:
            guide(*args, **kwargs)
        if guide_is_trainable:
            guide.train(was_training)
        ret = {'guide_trace': guide_tracer.trace}

        if self.observations:
            with pyro.poutine.trace() as model_tracer, \
                    pyro.poutine.replay(trace=ret['guide_trace']), \
                    self.uncondition:
                model(*args, **kwargs)
            ret['model_trace'] = model_tracer.trace

        if self.savename:
            torch.save(ret, self.savename)

        return ret


class Commandable:
    @property
    @lru_cache()
    def commands(self) -> tp.Mapping[str, tp.Any]:
        return tp.get_type_hints(type(self))

    def get_cmd_cls(self, name: str) -> tp.Optional[tp.Type[Command]]:
        # sys.version_info >= (3, 8)
        cmd = self.commands.get(name, None)
        return cmd if cmd is not None and issubclass(cmd, Command) else None

    def register_cmd_cls(self, name: str, cls: tp.Type[Command]):
        type(self).__annotations__[name] = cls

    def __setattr__(self, key, value):
        if isinstance(value, Command):
            value.boundkwargs = ProxyDict(self, ('model', 'guide'))
            # value.boundkwargs = dict(model=self.model, guide=self.guide)
        super().__setattr__(key, value)

    def __getattr__(self, name: str):
        # sys.version_indo > (3, 8)
        cmd = self.get_cmd_cls(name)
        if cmd is not None:
            setattr(self, name, cmd())
            return getattr(self, name)

        raise AttributeError(name)


class Clipppy(Commandable):
    """
    Attributes
    ----------
    model
        A callable that acts as a generative model.
    guide
        A callable that acts as a guide. Preferably a :any:`Guide`.
    """
    def __init__(self,
                 model: _Model = noop,
                 guide: Guide = Guide(),
                 conditioning: tp.Dict[str, torch.Tensor] = None,
                 **kwargs):
        # Conditions the model and sets it on the guide, if it doesn't have a model already.
        self.conditioning = conditioning
        self._model = model
        self.guide = guide

        if isinstance(self.guide, Guide) and self.guide.model is None:
            self.guide.model = self.model

        self.kwargs = {}
        for key, val in kwargs.items():
            if self.get_cmd_cls(key) is not None:
                setattr(self, key, val)
            else:
                self.kwargs[key] = val

    @property
    def umodel(self):
        return depoutine(self._model)

    @property
    def model(self):
        return self._model if self.conditioning is None else pyro.condition(self._model, data=self.conditioning)

    fit: Fit
    mock: Mock
    ppd: PPD


register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
