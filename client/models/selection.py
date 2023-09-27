import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import itertools
from client.logger import ConsoleLogger
import logging

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class PeerSelectionModel(nn.Module):
    def __init__(self, n_samples: int, alpha: float = 1, theta: float = 0.5):
        super().__init__()
        # Attributes
        self.alpha = alpha
        self.theta = theta
        self.n_samples = n_samples
        # Optimization parameters
        self.weights = nn.Parameter(torch.rand(self.n_samples))

    def forward(self, gain: torch.Tensor, energy: torch.Tensor):
        return ((self.alpha * torch.dot(torch.sigmoid(self.weights), gain) /
                 torch.dot(torch.sigmoid(self.weights), energy)) + (self.theta * torch.sigmoid(self.weights).sum()))

    def get_betas(self):
        return torch.sigmoid(self.weights)


class LightningPeerSelection(LightningModule):
    inc = itertools.count(start=1)

    def __init__(self, n_samples: int, alpha: float = 1, theta: float = 0.5):
        super().__init__()
        self.id_ = next(LightningPeerSelection.inc)

        self.model = PeerSelectionModel(n_samples, alpha, theta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, maximize=True)

        self.training_step_output = {'rew': float('nan')}

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'n_samples={self.model.n_samples}, '
                f'alpha={self.model.alpha}, '
                f'theta={self.model.theta})')

    def forward(self, gain: torch.Tensor, energy: torch.Tensor):
        return self.model(gain, energy)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.float:
        inputs, targets = batch
        reward = self(inputs)
        self.training_step_output['rew'] = reward.item()
        return reward

    def configure_optimizers(self):
        return self.optimizer

    # Hooks
    def on_train_epoch_end(self) -> None:
        # Gather and report
        logger.info('Epoch [{:>2}/{:>2}] - Reward = {:.3f}'.format(self.current_epoch + 1, self.trainer.max_epochs, self.training_step_output['rew']))
