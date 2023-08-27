from enum import Enum
import random
from typing import Sequence

class Client:
  pass

class PeerSelector:
  class Policy(str, Enum):
    RANDOM = 'Random'
    FULL = 'Full'
    EFFICIENT = 'Efficient'

  def __init__(self, policy: Policy) -> None:
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
      return PeerSelector.full_selection(neighbors)
    elif self.policy == PeerSelector.Policy.RANDOM:
      return PeerSelector.random_selection(neighbors, k = k)
    elif self.policy == PeerSelector.Policy.EFFICIENT:
      return PeerSelector.efficient_selection(neighbors)
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