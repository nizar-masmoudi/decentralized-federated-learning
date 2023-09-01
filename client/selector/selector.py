from enum import IntEnum
import random
from typing import List
import logging
from client.loggers.console import ConsoleLogger


class Client:
    pass


logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class PeerSelector:
    class Policy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self, client_id: int, policy: Policy) -> None:
        self.client_id = client_id
        self.policy = policy

    def select_peers(self, neighbors: List[Client], k: int = None) -> List[Client]:
        """Select peers from neighboring nodes using a predefined policy"""
        if self.policy == PeerSelector.Policy.FULL:
            peers = PeerSelector.full_selection(neighbors)
            logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra={'client': self.client_id})
            return peers
        elif self.policy == PeerSelector.Policy.RANDOM:
            peers = PeerSelector.random_selection(neighbors, k=k)
            logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra={'client': self.client_id})
            return peers
        elif self.policy == PeerSelector.Policy.EFFICIENT:
            peers = PeerSelector.efficient_selection(neighbors)
            logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra={'client': self.client_id})
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
