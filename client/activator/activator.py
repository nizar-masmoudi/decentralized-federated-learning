import random
from enum import IntEnum
import logging
from client.loggers.console import ConsoleLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Activator:
    class Policy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, client_id: int, policy: Policy) -> None:
        self.client_id = client_id
        self.policy = policy

    def activate(self, p: float = None) -> bool:
        """Activate client for round participation according to a predefined policy.

    Args:
        p (float, optional): Probability of activation. This parameter is only relevant with Random Activation policy. Defaults to None.

    Raises:
        ValueError: Unrecognized policy.

    Returns:
        bool: Indicates whether client is active or not.
    """
        if self.policy == Activator.Policy.FULL:
            is_active = Activator.full_activation()
            logger.info('Client set to {}'.format('active' if is_active else 'inactive'), extra={'client': self.client_id})
            return is_active
        elif self.policy == Activator.Policy.RANDOM:
            is_active = Activator.random_activation(p=p)
            logger.info('Client set to {}'.format('active' if is_active else 'inactive'), extra={'client': self.client_id})
            return is_active
        elif self.policy == Activator.Policy.EFFICIENT:
            is_active = Activator.efficient_activation()
            logger.info('Client set to {}'.format('active' if is_active else 'inactive'), extra={'client': self.client_id})
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
        raise NotImplementedError
