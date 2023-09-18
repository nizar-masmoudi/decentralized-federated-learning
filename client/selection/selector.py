import random
import logging
from client.loggers.system import SystemLogger
import itertools
from client.models import SelectionModel
import numpy as np
from torch.optim import Adam
import torch

logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class PeerSelector:
    inc = itertools.count(start=1)

    def __init__(self) -> None:
        self.id_ = next(PeerSelector.inc)  # Auto-increment ID

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            + ', '.join(
                [
                    f'{attr}={value}'
                    for attr, value in vars(self).items()
                    if attr != 'id_'
                ]
            )
            + ')'
        )

    def select(self, *args, **kwargs):
        ...


class RandomPeerSelector(PeerSelector):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def select(self, neighbors: list) -> list:
        return [neighbor for neighbor in neighbors if random.random() < self.p]


class EfficientPeerSelector(PeerSelector):
    MAX_GAIN = 2
    MAX_ENERGY = 2
    MAX_EPOCHS = 1000

    def __init__(self):
        super().__init__()

    def select(self, client: any, neighbors: list, alpha: float, theta: float, log_interval: int = 10) -> list:
        # TODO - Change object_size
        gain_list = [neighbor.config.loss_history[1] for neighbor in neighbors]
        energy_list = [neighbor.communication_energy(client, neighbor, 1e6) for neighbor in neighbors]
        scaled_gain = EfficientPeerSelector.minmaxscale(gain_list, 0, EfficientPeerSelector.MAX_GAIN)
        scaled_energy = EfficientPeerSelector.minmaxscale(energy_list, 0, EfficientPeerSelector.MAX_ENERGY)
        gain_tensor = torch.tensor(scaled_gain)
        energy_tensor = torch.tensor(scaled_energy)

        model = SelectionModel(len(neighbors), alpha, theta)
        optimizer = Adam(model.parameters(), lr=0.1)

        for epoch in range(EfficientPeerSelector.MAX_ENERGY):
            optimizer.zero_grad()
            loss = model(energy_tensor, gain_tensor)
            loss.backward()
            optimizer.step()
            # Early stopping

            # Report
            if epoch % log_interval == 0:
                logger.info(
                    'Epoch [{:>3}/{:>3}] - Loss = {:.3f}'.format(epoch + 1, EfficientPeerSelector.MAX_EPOCHS, loss),
                    extra={'client': self.id_}
                )
        return []

    @staticmethod
    def minmaxscale(feature: list, min_: float, max_: float):
        scaled = (np.array(feature) - min_) / max_ - min_
        return scaled.tolist()

