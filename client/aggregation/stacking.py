from client.aggregation.aggregator import Aggregator
from typing import List, Dict


class FedStack(Aggregator):
    def __init__(self):
        super().__init__()

    def aggregate(self, state_dicts: List[Dict]) -> Dict:
        raise NotImplementedError
