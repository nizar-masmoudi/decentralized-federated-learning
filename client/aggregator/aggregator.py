from enum import IntEnum
from typing import List, Dict
import logging
import torch
from client.loggers.console import ConsoleLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Aggregator:
    class Policy(IntEnum):
        FEDAVG = 0
        MIXING = 1

        def __repr__(self):
            return f'AggregationPolicy=Aggregator.{self.__class__.__name__}.{self.name}'

    def __init__(self, id_: int, policy: Policy) -> None:
        self.id_ = id_
        self.policy = policy

    def aggregate(self, state_dicts: List[Dict]) -> Dict:
        if self.policy == Aggregator.Policy.FEDAVG:
            agg_state = Aggregator.fedavg(state_dicts)
            logger.info(f'{len(state_dicts)} states of {len(agg_state)} layers were aggregated', extra={'client': self.id_})
            return agg_state

    @staticmethod
    def fedavg(state_dicts: List[Dict]) -> Dict:
        agg_state = state_dicts[0].copy()
        for key in agg_state:
            agg_state[key] = torch.stack([state[key] for state in state_dicts if key in state.keys()]).mean(0)
        return agg_state

    @staticmethod
    def mixing(state_dicts: List[Dict]) -> Dict:
        # TODO: Implement mixing aggregation
        raise NotImplementedError
