import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import client as cl
from client.loggers import ConsoleLogger
from client.selection.selector import PeerSelector

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


def minmaxscale(feature: list, min_: float, max_: float):
    scaled = np.ones(len(feature))
    if min_ != max_:
        scaled = (np.array(feature) - min_) / max_ - min_
    return scaled.tolist()


class CustomEarlyStopping:
    def __init__(self, patience: int, delta: float):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.prev_loss = None

    def stop(self, loss: float):
        if self.prev_loss is not None:
            if abs(loss - self.prev_loss) < self.delta:
                if self.counter < self.patience:
                    self.counter += 1
                else:
                    return True
        self.prev_loss = loss
        return False


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
        return ((self.alpha * torch.dot(torch.sigmoid(self.weights), energy) /
                 torch.dot(torch.sigmoid(self.weights), gain)) + (self.theta / torch.sigmoid(self.weights).sum()))

    def get_betas(self):
        return torch.sigmoid(self.weights)


class EfficientPeerSelector(PeerSelector):
    def __init__(self, alpha: float = 1, theta: float = .5, log_interval: int = 20):
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.log_interval = log_interval

    def select(self, client: 'cl.Client') -> List['cl.Client']:
        selected_neighbors = list(filter(
            lambda neighbor: (neighbor.model.loss_history[-1] - client.model.loss_history[-1]) > 0,
            client.neighbors
        ))
        if len(selected_neighbors) == 0:
            return []

        # Get positive knowledge gain for each neighbor
        kgain = [client.knowledge_gain(neighbor) for neighbor in selected_neighbors]

        # Get communication energy for each neighbor + Scaling
        energy = [client.communication_energy(neighbor) for neighbor in selected_neighbors]
        energy = minmaxscale(energy, 0, client.communication_energy(distance=client.lookup_dist))

        # Get tensors
        t_kgain = torch.tensor(kgain, dtype=torch.float)
        t_energy = torch.tensor(energy, dtype=torch.float)
        # Setup model + optimizer
        model = PeerSelectionModel(len(selected_neighbors), self.alpha, self.theta)
        early_stopping = CustomEarlyStopping(3, 1e-4)
        optimizer = Adam(model.parameters(), lr=.1)
        # Optimize
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = model(t_kgain, t_energy)
            loss.backward()
            optimizer.step()

            # Log
            if epoch % self.log_interval == 0 or epoch == 1000:
                report_str = 'Epoch [{:>3}/{:>3}] - Loss = {:.3f}'.format(epoch + 1, 1000, loss)
                logger.info(report_str, extra={'id': self.id_})
            if early_stopping.stop(loss):
                logger.info('Early stopping at epoch [{:>3}/{:>3}] - Loss = {:.3f}'.format(epoch + 1, 1000, loss),
                            extra={'id': self.id_})
                break

        mask = model.get_betas().detach().numpy().round().astype(bool)
        peers = [selected_neighbors[i] for i in range(len(selected_neighbors)) if mask[i]]
        return peers
