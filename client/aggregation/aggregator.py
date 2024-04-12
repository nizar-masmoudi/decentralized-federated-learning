from abc import ABC, abstractmethod
from typing import Dict, List


class Aggregator(ABC):
    """
    Abstract class used to create custom aggregators.
    """
    def __init__(self):
        ...

    def __repr__(self):
        params = ', '.join([f'{key}={value}' for key, value in self.__dict__.values()])
        return f'{self.__class__.__name__}({params})'

    @abstractmethod
    def aggregate(self, state_dicts: List[Dict]) -> Dict:
        """
        Aggregation function.
        :param state_dicts: Models' state dicts
        :rtype: OrderedDict
        """
        ...
