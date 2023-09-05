from enum import IntEnum
import random
from typing import List
import logging
from client.loggers.console import ConsoleLogger
from client.configs import TransmissionConfig


class Client:
    pass


logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class PeerSelector:
    class SelectionPolicy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, id_: int, policy: SelectionPolicy) -> None:
        self.id_ = id_
        self.policy = policy

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
