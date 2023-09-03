import random
from enum import IntEnum
import logging
from client.loggers.console import ConsoleLogger
from client.configs import ComputationConfig

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class ClientActivator:
    class Policy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, id_: int, policy: Policy, comp_cfg: ComputationConfig) -> None:
        self.id_ = id_
        self.policy = policy
        self.comp_cfg = comp_cfg

    def activate(self, p: float = None) -> bool:
        """Activate client for round participation according to a predefined policy"""
        if self.policy == ClientActivator.Policy.FULL:
            is_active = ClientActivator.full_activation()
            logger.info('Client set to {}'.format('active' if is_active else 'inactive'), extra={'client': self.id_})
            return is_active
        elif self.policy == ClientActivator.Policy.RANDOM:
            is_active = ClientActivator.random_activation(p=p)
            logger.info('Client set to {}'.format('active' if is_active else 'inactive'), extra={'client': self.id_})
            return is_active
        elif self.policy == ClientActivator.Policy.EFFICIENT:
            is_active = ClientActivator.efficient_activation()
            logger.info('Client set to {}'.format('active' if is_active else 'inactive'), extra={'client': self.id_})
            return is_active
        else:
            raise ValueError(f'Policy {self.policy} not recognized!')

    @staticmethod
    def full_activation() -> bool:
        return True

    @staticmethod
    def random_activation(p: float) -> bool:
        return random.random() < p

    @staticmethod
    def efficient_activation() -> bool:
        # TODO: Implement efficient activation
        raise NotImplementedError
