import random
from enum import Enum
import logging

class Activator:
  class Policy(str, Enum):
    RANDOM = 'Random'
    FULL = 'Full'
    EFFICIENT = 'Efficient'
  
  def __init__(self, policy: Policy) -> None:
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
      return Activator.full_activation()
    elif self.policy == Activator.Policy.RANDOM:
      return Activator.random_activation(p = p)
    elif self.policy == Activator.Policy.EFFICIENT:
      return Activator.efficient_activation()
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