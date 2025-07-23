from __future__ import annotations

import io
from os import PathLike
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def parse(fname: str | PathLike[str], **kwargs):
    fname = Path(fname)
    return io.StringIO(Environment(
        loader=FileSystemLoader(fname.parent),
        undefined=StrictUndefined,
        trim_blocks=True, lstrip_blocks=True
    ).from_string(open(fname).read()).render(**kwargs))
