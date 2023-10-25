from typing import List, Dict

import torch

from client.aggregation.aggregator import Aggregator


class FedAvg(Aggregator):
    def __init__(self):
        super().__init__()

    def aggregate(self, state_dicts: List[Dict]) -> Dict:
        agg_state = state_dicts[0].copy()
        for key in agg_state:
            agg_state[key] = torch.stack([state[key] for state in state_dicts if key in state.keys()]).mean(0)
        return agg_state
