from enum import IntEnum
import random
import logging
from client.loggers.system import SystemLogger
import itertools

logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class Activator:
    inc = itertools.count(start=1)

    class ActivationPolicy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, policy: ActivationPolicy):
        self.id_ = next(Activator.inc)  # Auto-increment ID
        self.policy = policy
        logger.debug(repr(self), extra={'client': self.id_})

    def __repr__(self):
        return f'Activator(id={self.id_}, policy={self.policy.name})'

    def activate(self):
        if self.policy == Activator.ActivationPolicy.FULL:
            return True
        elif self.policy == Activator.ActivationPolicy.RANDOM:
            return random.random() < 0.7
        elif self.policy == Activator.ActivationPolicy.EFFICIENT:
            raise NotImplementedError
        else:
            raise ValueError(f'Policy {self.policy} not recognized!')

    @staticmethod
    def efficient_activation(client, neighbors: list):
        pass

