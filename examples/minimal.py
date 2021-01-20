import pyro
import pyro.distributions as dist
import torch


##################################################
# Simple linear regression via registered function
##################################################

def linear(a, b, x):
    return pyro.sample('y', dist.Normal(a + b*x, 1.0))


#######################################
# Simple quadratic regression via class
#######################################

class Quadratic:
    def __init__(self, x):
        self.x = x

    def __call__(self, a, b, c):
        print(a, b, c)
        return pyro.sample('y', dist.Normal(a + b*self.x + c*self.x**2, 1.0))


########################################################################
# More complex model with several "Source" instances + linear regression
########################################################################

class Source:
    def __init__(self, xgrid: torch.Tensor):
        self.xgrid = xgrid

    def __call__(self, h0, x0, w0):
        """Gaussian source with free height, width and position."""
        return h0.unsqueeze(-1) * torch.exp(-0.5 * (self.xgrid-x0.unsqueeze(-1))**2 / w0.unsqueeze(-1)**2)


class SpecModel:
    def __init__(self, xgrid: torch.Tensor, sources):
        self.xgrid = xgrid
        self.sources = sources

    def __call__(self, a: torch.Tensor, b: torch.Tensor):
        spec = sum([source() for source in self.sources])
        spec += self.xgrid*b.unsqueeze(-1) + a.unsqueeze(-1)

        # The whole spectrum is a single "event", so
        # event_dim=1 and .to_event(1) are necessary to ensure proper
        # interaction with plates.
        pyro.deterministic('spec_nonoise', spec, event_dim=1)
        return pyro.sample('spec', dist.Normal(spec, 1.0).to_event(1))
