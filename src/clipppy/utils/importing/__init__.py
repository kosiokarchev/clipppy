from __future__ import annotations

import importlib.util
import sys


def get_pure_python_module(name: str):
    orig_mods = sys.modules[name], sys.modules[f'_{name}']
    try:
        sys.modules[f'_{name}'] = None
        (spec := importlib.util.find_spec(name)).loader.exec_module(
            mod := importlib.util.module_from_spec(spec))
        return mod
    finally:
        sys.modules[name], sys.modules[f'_{name}'] = orig_mods
