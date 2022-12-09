from __future__ import annotations

import io
from os import PathLike
from pathlib import Path
from typing import Union

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def parse(fname: Union[str, PathLike[str]], **kwargs):
    fname = Path(fname)
    return io.StringIO(Environment(
        loader=FileSystemLoader(fname.parent),
        undefined=StrictUndefined
    ).from_string(open(fname).read()).render(**kwargs))
