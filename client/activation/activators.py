from enum import IntEnum
import random
import logging
from client.loggers.system import SystemLogger
import itertools
from typing import overload

logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class Activator:
    inc = itertools.count(start=1)

    def __init__(self):
        self.id_ = next(Activator.inc)  # Auto-increment ID

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id_})'

    def activate(self, *args, **kwargs):
        ...


class RandomActivator(Activator):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def activate(self) -> bool:
        return random.random() < self.p


class EfficientActivator(Activator):
    def __init__(self, threshold: float, alpha: float):
        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid 'alpha'. Value must be within 0 and 1.")
        if threshold < 0 or threshold > 1:
            raise ValueError("Invalid 'threshold'. Value must be within 0 and 1.")

        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def activate(self, client, neighbors: list) -> bool:
        # Computation energy consumption
        energy = client.computation_energy()
        energies = [neighbor.computation_energy() for neighbor in neighbors] + [energy]
        if max(energies) == min(energies):
            scaled_energy = 0.
        else:
            scaled_energy = (energy - min(energies)) / (max(energies) - min(energies))
        logger.debug(f'Activator.efficient_activation.scaled_energy={scaled_energy}', extra={'client': self.id_})
        # Learning slope
        slope = client.learning_slope()
        slopes = [neighbor.learning_slope() for neighbor in neighbors] + [slope]
        if max(slopes) == min(slopes):
            scaled_slope = 1.
        else:
            scaled_slope = (energy - min(slopes)) / (max(slopes) - min(slopes))
        logger.debug(f'Activator.efficient_activation.scaled_slope={scaled_slope}', extra={'client': self.id_})
        # Activation cost
        cost = self.alpha * scaled_energy + (1 - self.alpha) * (1 - scaled_slope)
        logger.debug(f'Activator.efficient_activation.cost={cost}', extra={'client': self.id_})
        return self.threshold > cost
