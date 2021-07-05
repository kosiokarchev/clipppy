import pyro
from pyro.distributions import Normal


def linear(a, b, x, err):
    y = a.unsqueeze(-1) * x + b.unsqueeze(-1)
    pyro.deterministic('y', y, event_dim=1)
    return pyro.sample('obs', Normal(y, err).to_event(1))
