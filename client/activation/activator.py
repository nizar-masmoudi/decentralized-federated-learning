import itertools
from abc import ABC, abstractmethod


class Activator(ABC):
    """
    Abstract class used to create custom activators.
    """
    inc = itertools.count(start=1)

    def __init__(self):
        self.id_ = next(Activator.inc)

    def __repr__(self):
        params = ', '.join([f'{key}={value}' for key, value in self.__dict__.values()])
        return f'{self.__class__.__name__}({params})'

    @abstractmethod
    def activate(self, *args, **kwargs) -> bool:
        """
        Activation function
        :param args: Activation arguments. It depends on implmented policy.
        :param kwargs: Arbitrary additional keyword arguments.
        """
        ...


class FullActivator(Activator):
    """
    Full activation policy class. All clients are active during each round.
    """
    def __init__(self):
        super().__init__()

    def activate(self, *args) -> bool:
        return True
