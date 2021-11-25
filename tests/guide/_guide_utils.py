import pyro
from pyro.distributions import Normal, Uniform
from pytest import fixture


unit_normal = Normal(0, 1)
some_uniform = Uniform(0.41, 0.43)


class ClearParamStore:
    @fixture(autouse=True, scope='function')
    def clear_param_store(self):
        pyro.clear_param_store()
