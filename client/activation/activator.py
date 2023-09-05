from enum import IntEnum
import random
import logging
from client.loggers.console import ConsoleLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Activator:
    class ActivationPolicy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, id_: int, policy: ActivationPolicy):
        self.id_ = id_
        self.policy = policy

    def activate(self):
        if self.policy == Activator.ActivationPolicy.FULL:
            return True
        elif self.policy == Activator.ActivationPolicy.RANDOM:
            return random.random() < .7
        elif self.policy == Activator.ActivationPolicy.EFFICIENT:
            raise NotImplementedError
        else:
            raise ValueError(f'Policy {self.policy} not recognized!')