from functools import partial

import attr


if not hasattr(attr, '_unpatched_s'):
    attr._unpatched_s = attr.s
    attr.s = partial(attr.s, eq=False, auto_attribs=True)
