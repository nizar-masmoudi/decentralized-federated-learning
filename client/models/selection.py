import torch
import torch.nn as nn


class SelectionModel(nn.Module):
    def __init__(self, n_samples: int, alpha: float = 1, theta: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attributes
        self.alpha = alpha
        self.theta = theta
        self.n_samples = n_samples
        # Optimization parameters
        self.weights = nn.Parameter(torch.rand(self.n_samples))

    def betas(self) -> torch.Tensor:
        return torch.sigmoid(self.weights)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return (
            self.alpha
            * torch.dot(torch.sigmoid(self.weights), a)
            / torch.dot(torch.sigmoid(self.weights), b)
        ) + self.theta / torch.sigmoid(self.weights).sum()

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, theta={self.theta})"
