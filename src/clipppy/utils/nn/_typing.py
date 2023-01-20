from typing import Callable, Union

from torch import Tensor
from torch.nn import Module
from typing_extensions import TypeAlias


_ff_module_like: TypeAlias = Union[Module, Callable[[Tensor], Tensor]]
