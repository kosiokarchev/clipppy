from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # BEGIN API
    from .command import Command
    from .commandable import Commandable
    from .fit import Fit
    from .lightning.npe import NPE as LightningNPE
    from .lightning.nre import NRE as LightningNRE
    from .mock import Mock
    from .ppd import PPD
    # END API

import re, ast, apipkg

apipkg.initpkg(__name__, {
    alias.asname or alias.name: stmt.level*'.'+f'{stmt.module}:{alias.name}'
    for res in re.findall(r'# BEGIN API\n(.*?)\s*# END API', open(__file__).read(), re.DOTALL)
    for line in res.split('\n') for line in [line.lstrip()]
    if line
    for stmt in ast.parse(line, mode='single').body
    if isinstance(stmt, ast.ImportFrom)
    for alias in stmt.names
})
