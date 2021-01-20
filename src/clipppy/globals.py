import builtins
import typing as tp


# GLOBAL REGISTRY
# ---------------
def get_global(name: str, default=None, scope: dict = None) -> tp.Any:
    if scope is not None and name in scope:
        return scope[name]
    elif name in globals():
        return globals()[name]
    elif hasattr(builtins, name):
        return getattr(builtins, name)
    else:
        return default


def register_globals(**kwargs):
    vars(builtins).update(kwargs)
