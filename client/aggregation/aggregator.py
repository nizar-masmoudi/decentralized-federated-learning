from enum import IntEnum
from typing import List, Dict
import logging
import torch
from client.loggers.system import SystemLogger
import itertools

logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class Aggregator:
    inc = itertools.count(start=1)

    class AggregationPolicy(IntEnum):
        FEDAVG = 0
        MIXING = 1

    def __init__(self, policy: AggregationPolicy) -> None:
        self.id_ = next(Aggregator.inc)  # Auto-increment ID
        self.policy = policy
        logger.debug(repr(self), extra={'client': self.id_})

    def __repr__(self):
        return f'Aggregator(id={self.id_}, policy={self.policy.name})'

    def aggregate(self, state_dicts: List[Dict]) -> Dict:
        if self.policy == Aggregator.AggregationPolicy.FEDAVG:
            agg_state = Aggregator.fedavg(state_dicts)
            return agg_state
        elif self.policy == Aggregator.AggregationPolicy.MIXING:
            agg_state = Aggregator.mixing(state_dicts)
            return agg_state
        else:
            raise ValueError(f'Policy {self.policy} not recognized!')

    @staticmethod
    def fedavg(state_dicts: List[Dict]) -> Dict:
        agg_state = state_dicts[0].copy()
        for key in agg_state:
            agg_state[key] = torch.stack([state[key] for state in state_dicts if key in state.keys()]).mean(0)
        return agg_state

    @staticmethod
    def mixing(state_dicts: List[Dict]) -> Dict:
        raise NotImplementedError
