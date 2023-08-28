from enum import IntEnum
import random
from typing import Sequence
import logging
class Client:
  pass

logger = logging.getLogger(__name__)

class PeerSelector:
  class Policy(IntEnum):
    RANDOM = 0
    FULL = 1
    EFFICIENT = 2

  def __init__(self, id: int, policy: Policy) -> None:
    self.id = id
    self.policy = policy
  
  def select_peers(self, neighbors: Sequence[Client], k: float = None) -> Sequence[Client]:
    """Select peers from neighboring nodes using a predefined policy.


    Args:
        neighbors (Sequence[Client]): Sequence of neighboring nodes.
        k (float, optional): Number of peers to select. This parameter is relevant only when using Random Selection policy. Defaults to None.

    Raises:
        ValueError: Unrecognized policy.
        
    Returns:
        Sequence[Client]: Sequence of selected peers.
    """    
    if self.policy == PeerSelector.Policy.FULL:
      peers = PeerSelector.full_selection(neighbors)
      logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra = {'client': self.id})
      return peers
    elif self.policy == PeerSelector.Policy.RANDOM:
      peers = PeerSelector.random_selection(neighbors, k = k)
      logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra = {'client': self.id})
      return peers
    elif self.policy == PeerSelector.Policy.EFFICIENT:
      peers = PeerSelector.efficient_selection(neighbors)
      logger.info('Selected peers: {}'.format(', '.join([str(peer) for peer in peers])), extra = {'client': self.id})
      return peers
    else:
      raise ValueError(f'Policy {self.policy} not recognized!')

  @staticmethod
  def full_selection(neighbors: Sequence[Client]) -> Sequence[Client]:
    """Select all neighboring nodes as peers.

    Args:
        neighbors (Sequence[Client]): Sequence of neighboring nodes.

    Returns:
        Sequence[Client]: Sequence of selected peers.
    """    
    return neighbors
  
  @staticmethod
  def random_selection(neighbors: Sequence[Client], k: int) -> Sequence[Client]:
    """Select a random subset of neigbhbors as peers.

    Args:
        neighbors (Sequence[Client]): Sequence of neighboring nodes.
        k (int): Size of subset.

    Returns:
        Sequence[Client]: Sequence of selected peers.
    """    
    return random.sample(neighbors, k)
  
  @staticmethod
  def efficient_selection(neighbors: Sequence[Client]) -> Sequence[Client]:
    """Select a subset of neighbors using an energy-efficient policy based on knowledge gain.

    Args:
        neighbors (Sequence[Client]): Sequence of neighboring nodes.

    Returns:
        Sequence[Client]: Sequence of selected peers.
    """    
    raise NotImplementedError