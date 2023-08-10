from operator import itemgetter
from warnings import warn
from abc import abstractmethod
from typing import TypeVar, Union, Any, Mapping, Optional, Collection, Sequence, Iterator

from more_itertools import all_equal, one, unique_everseen
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypeAlias


_T = TypeVar('_T')
_CTensorT: TypeAlias = Union[Tensor, Any]
_CValuesT: TypeAlias = Mapping[str, _CTensorT]


class PersistentDataset(Dataset[_CValuesT]):
    @property
    @abstractmethod
    def keys(self) -> Optional[Collection[str]]: ...

    def get_values_for_keys(self, values: Mapping[str, _T]) -> Mapping[str, _T]:
        return values if self.keys is None else {key: values[key] for key in self.keys}

    @abstractmethod
    def _extend_batch(self, values: Mapping[str, Sequence[Tensor]]): ...

    def extend_batch(self, values: Mapping[str, Sequence[Tensor]]):
        values = self.get_values_for_keys(values)

        if not all_equal(
            # this supports Nestedtensor, for which len raises an error...
            v.size(0) if isinstance(v, Tensor) else len(v)
            for v in values.values()
        ):
            warn(f'Appending unequal-length batches to {type(self).__name__}', RuntimeWarning)

        return self._extend_batch(values)

    def extend(self, values: Mapping[str, Tensor]):
        self.extend_batch({
            key: val.unsqueeze(0)
            for key, val in self.get_values_for_keys(values).items()})

    @property
    @abstractmethod
    def variables(self) -> Iterator[tuple[str, Sequence[_CTensorT]]]: ...

    def __getitem__(self, item) -> _CValuesT:
        return {key: val[item] for key, val in self.variables}

    def __len__(self) -> int:
        return one(unique_everseen(map(len, map(itemgetter(1), self.variables))))
