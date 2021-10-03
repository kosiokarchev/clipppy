from pyro.distributions import Delta


# TODO: Python 3.10 ArgSpec
class ExtraDelta(Delta):
    def __init__(self, *args, extra_event_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum_over = tuple(range(-extra_event_dim, 0))

    def log_prob(self, x):
        return super().log_prob(x).sum(self.sum_over)