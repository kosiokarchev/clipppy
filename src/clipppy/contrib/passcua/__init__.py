from dataclasses import field
from typing import Callable, Any, Mapping

from torch.distributions import Categorical

from phytorchx import broadcast_gather
from .bounds import *
from .samples import SamplesMetric, SampleBatch


@dataclass
class Passcua:
    """Parallel Slice Sampling with Calibrated Uniform Acceptance"""

    ndim: int
    log_prob_fn: Callable[[Tensor], Tensor]
    bound: Bound

    samples: SamplesMetric = field(init=False, default_factory=SamplesMetric)


    def random_direction(self, shape: Union[Size, tuple[int, ...]]) -> Tensor:
        d = torch.randn(shape+(self.ndim,))
        return d / norm(d, dim=-1, keepdim=True)

    def random_slices(self, p: Tensor, ndirs: int, nsamples: int) -> Tensor:
        p = p.unsqueeze(-2)
        d = self.random_direction(p.shape[:-2]+(ndirs,))
        tmax = self.bound.intersection_dist(p, d)
        dmax = broadcast_gather(
            d*tmax.unsqueeze(-1), -2,
            Categorical(probs=tmax**self.ndim).sample((nsamples,)).movedim(0, -1)
        )
        rands = dmax.new_empty(dmax.shape[:-1]).uniform_().pow_(1/self.ndim)
        return p + dmax * rands.unsqueeze(-1)

    def gen_samples(self, p: Tensor, ndirs: int, nsamples: int, y: Tensor = None):
        if y is None:
            y = self.log_prob_fn(p)
        pnew = self.random_slices(p, ndirs, nsamples)
        ynew = self.log_prob_fn(pnew)
        accept = ynew >= (torch.rand_like(y).log_() + y).unsqueeze(-1)
        return SampleBatch(
            pnew.flatten(end_dim=-2)[accept.flatten()],
            ynew[accept],
            (accept.shape[-1] / accept.sum(-1, keepdim=True)).expand_as(accept)[accept],
        )

    def run(
        self, nbatch: int, ndirs: int = 100,
        p_init: Tensor = None, nseed: int = None,
        progress: Union[bool, Mapping[str, Any]] = True,
        target_ess: Real = float('inf'),
        max_iter: Union[Real, float, int] = float('inf')
    ):
        if p_init is None:
            p_init = self.bound.get_point().unsqueeze(0)
        if nseed is None:
            nseed = len(p_init)

        if progress:
            from tqdm.auto import tqdm
            tq = tqdm(total=max_iter, **{**dict(leave=False), **({} if progress is True else progress)})

        y = None
        i = 0

        while i < max_iter:
            self.samples.update(**self.gen_samples(p_init, ndirs, nbatch, y=y)._asdict())
            ess = self.samples.ess()
            if progress:
                tq.set_postfix_str(f'n: {len(self.samples)}, ess: {ess:.0f} / {target_ess}')
                tq.update(1)
            if ess >= target_ess:
                break
            if ess >= nseed:
                p_init, y = self.samples.sample(nseed, return_y=True)
            i += 1

        if progress:
            tq.close()

        return self.samples
