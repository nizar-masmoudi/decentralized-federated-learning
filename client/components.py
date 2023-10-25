import dataclasses
from typing import ClassVar

import numpy as np


class RandomFactory:
    def __init__(self, min_: float, max_: float, step: float, dtype: type = int):
        """
        A random number factory.
        :param min_: Start of interval. The interval includes this value.
        :param max_: End of interval. The interval includes this value.
        :param step: Spacing between values.
        :param dtype: The type of the output value.
        """
        self.min_ = min_
        self.max_ = max_
        self.step = step
        self.dtype = dtype

    def __call__(self):
        """
        Generate random value.
        """
        return self.dtype(np.random.choice(np.arange(self.min_, self.max_ + self.step, self.step)))


@dataclasses.dataclass
class Transmitter:
    power: float = dataclasses.field(default_factory=RandomFactory(10, 21, 5))
    """Transmission power."""
    bandwidth: float = dataclasses.field(default_factory=RandomFactory(5e6, 20e6, 5e6))
    """Signal bandwidth."""
    psd: float = dataclasses.field(default=-174, init=False)
    """Power spectral density of the Gaussian noise."""
    signal_frequency: float = dataclasses.field(default=1e9)
    """Signal frequency."""
    transmit_gain: float = dataclasses.field(default=0)
    """Transmit gain. Used to calculate channel gain."""
    receive_gain: float = dataclasses.field(default=0)
    """Receive gain. Used to calculate channel gain."""

    def __repr__(self):
        return (f'Transmitter(power={self.power}, bandwidth={self.bandwidth:.0e}, psd={self.psd}, '
                f'signal_frequency={self.signal_frequency:.0e}, transmit_gain={self.transmit_gain}, '
                f'receive_gain={self.receive_gain})')

    def to_dict(self):
        return {
            'power': self.power,
            'bandwidth': self.bandwidth,
            'psd': self.psd,
            'signal_frequency': self.signal_frequency,
            'transmit_gain': self.transmit_gain,
            'receive_gain': self.receive_gain
        }


@dataclasses.dataclass
class CPU:
    MIN_FREQ: ClassVar[int] = 1e9
    MAX_FREQ: ClassVar[int] = 5e9

    fpc: int = dataclasses.field(default=4)
    """FLOPs per CPU cycle."""
    frequency: float = dataclasses.field(default_factory=RandomFactory(MIN_FREQ, MAX_FREQ, 1e9))
    """CPU clock frequency."""
    kappa: float = dataclasses.field(default=1e-28, init=False)
    """Effective capacitance."""

    def __repr__(self):
        return f'CPU(fpc={self.fpc}, frequency={self.frequency:.0e}, kappa={self.kappa:.0e})'

    def to_dict(self):
        return {
            'fpc': self.fpc,
            'frequency': self.frequency,
            'kappa': self.kappa
        }
