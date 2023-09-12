from enum import IntEnum
import random
from typing import List
import logging
from client.loggers.system import SystemLogger
import itertools


class Client:
    pass


logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class PeerSelector:
    inc = itertools.count(start=1)

    class SelectionPolicy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, policy: SelectionPolicy) -> None:
        self.id_ = next(PeerSelector.inc)  # Auto-increment ID
        self.policy = policy
        logger.debug(repr(self), extra={'client': self.id_})

    def __repr__(self):
        return f'PeerSelector(id={self.id_}, policy={self.policy.name})'

    def select(self, neighbors: List[Client]) -> List[Client]:
        if self.policy == PeerSelector.SelectionPolicy.FULL:
            return neighbors
        elif self.policy == PeerSelector.SelectionPolicy.RANDOM:
            return random.sample(neighbors, int(len(neighbors)*.7))
        elif self.policy == PeerSelector.SelectionPolicy.EFFICIENT:
            peers = PeerSelector.efficient_selection(neighbors)
            return peers
        else:
            raise ValueError(f'Policy {self.policy} not recognized!')

    @staticmethod
    def efficient_selection(neighbors: List[Client]) -> List[Client]:
        raise NotImplementedError
