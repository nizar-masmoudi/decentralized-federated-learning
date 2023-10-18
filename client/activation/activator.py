from abc import ABC, abstractmethod
import random
import itertools


class Activator(ABC):
    inc = itertools.count(start=1)

    """
    Abstract class of activator.
    """
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
    def __init__(self):
        super().__init__()

    def activate(self, *args) -> bool:
        return True


class RandActivator(Activator):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def activate(self, *args) -> bool:
        return random.random() < self.p
