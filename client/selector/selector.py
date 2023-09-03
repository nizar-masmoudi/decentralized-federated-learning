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
    class Policy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

        def __repr__(self):
            return f'PeerSelectionPolicy=PeerSelector.{self.__class__.__name__}.{self.name}'

    def __init__(self, id_: int, policy: Policy, trans_cfg: TransmissionConfig) -> None:
        self.id_ = id_
        self.policy = policy
        self.trans_cfg = trans_cfg

    def select_peers(self, neighbors: List[Client], k: int = None) -> List[Client]:
        """Select peers from neighboring nodes using a predefined policy"""
        if self.policy == PeerSelector.Policy.FULL:
            peers = PeerSelector.full_selection(neighbors)
            logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra={'client': self.id_})
            return peers
        elif self.policy == PeerSelector.Policy.RANDOM:
            peers = PeerSelector.random_selection(neighbors, k=k)
            logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra={'client': self.id_})
            return peers
        elif self.policy == PeerSelector.Policy.EFFICIENT:
            peers = PeerSelector.efficient_selection(neighbors)
            logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra={'client': self.id_})
            return peers
        else:
            raise ValueError(f'Policy {self.policy} not recognized!')

    @staticmethod
    def full_selection(neighbors: List[Client]) -> List[Client]:
        """Select all neighboring nodes as peers"""
        return neighbors

    @staticmethod
    def random_selection(neighbors: List[Client], k: int) -> List[Client]:
        """Select a random subset of neigbhbors as peers"""
        return random.sample(neighbors, k)

    @staticmethod
    def efficient_selection(neighbors: List[Client]) -> List[Client]:
        """Select a subset of neighbors using an energy-efficient policy based on knowledge gain"""
        # TODO: Implement efficient peer selection
        raise NotImplementedError
