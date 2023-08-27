from enum import Enum
from typing import List, OrderedDict

import torch


class Aggregator:
  class Policy(str, Enum):
    FEDAVG = 'FedAvg'
    MIXING = 'Mixing'
    
  def __init__(self, policy: Policy) -> None:
    self.policy = policy
    
  def aggregate(self, state_dicts: List[OrderedDict]) -> OrderedDict:
    if self.policy == Aggregator.Policy.FEDAVG:
      return Aggregator.fedavg(state_dicts)
  
  @staticmethod
  def fedavg(state_dicts: List[OrderedDict]) -> OrderedDict:
    agg_state = state_dicts[0].copy()
    for key in agg_state:
      agg_state[key] = torch.stack([state[key] for state in state_dicts if key in state.keys()]).mean(0)
    return agg_state