import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Callable, Sequence

from clipppy.sbi.persistent.netcdf_data import NetCDFDataset
from torch import Tensor
from tqdm.auto import tqdm


@dataclass
class NCDatagen:
    trainsize: int
    valsize: int

    basename: str
    generate_batch: Callable[[], Mapping[str, Sequence[Tensor]]]

    var_dimensions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    dimlens: Mapping[str, int] = field(default_factory=dict)

    def run(self, verbose=True):
        for suffix, total in (('train', self.trainsize), ('val', self.valsize)):
            name = self.basename.format(suffix=suffix)
            Path(name).parent.mkdir(exist_ok=True, parents=True)

            if verbose:
                print('Appending to' if os.path.exists(name) else 'Creating', name)

            d = NetCDFDataset(name, 'r+', var_dimensions=defaultdict(tuple, self.var_dimensions))

            for dimname, dimlen in self.dimlens.items():
                if dimname not in d.group.dimensions:
                    d.group.createDimension(dimname, dimlen)

            with tqdm(total=total-len(d)) as tq:
                while len(d) < total:
                    d.extend_batch(self.generate_batch())
                    tq.update(1)

            d.group.close()

        if verbose:
            print('Done.')

        return self
