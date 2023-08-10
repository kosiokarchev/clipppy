import torch.nn


class LeakyPOP(torch.nn.Module):
    def __init__(self, alpha: float = 2., beta: float = 1., **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor):
        return self.beta * x + torch.sign(x) * torch.abs(x)**self.alpha
