import dataclasses
import random
import math


def frange(start: float = 0, stop: float = 1, step: float = 0.1):
    value = start
    while value <= stop:
        yield value
        value += step


POWER_RANGE = list(range(10, 21, 5))
BANDWIDTH_RANGE = list(frange(5e6, 20e6, 5e6))
CPU_FREQUENCY_RANGE = list(frange(1e9, 5e9, 1e9))


def random_factory(values: list):
    return random.choice(values)


@dataclasses.dataclass
class Transmitter:
    power: float = dataclasses.field(default_factory=lambda: random_factory(POWER_RANGE))
    bandwidth: float = dataclasses.field(default_factory=lambda: random_factory(BANDWIDTH_RANGE))
    psd: float = dataclasses.field(default=-174, init=False)
    signal_frequency: float = dataclasses.field(default=1e9)
    transmit_gain: float = dataclasses.field(default=0)
    receive_gain: float = dataclasses.field(default=0)

    def __repr__(self):
        return (f'Transmitter(power={self.power}, bandwidth={self.bandwidth:.0e}, psd={self.psd}, '
                f'signal_frequency={self.signal_frequency:.0e}, transmit_gain={self.transmit_gain}, '
                f'receive_gain={self.receive_gain})')


@dataclasses.dataclass
class CPU:
    flops_per_cycle: int = dataclasses.field(default=4)
    frequency: float = dataclasses.field(default_factory=lambda: random_factory(CPU_FREQUENCY_RANGE))
    kappa: float = dataclasses.field(default=1e-28, init=False)

    def __repr__(self):
        return f'CPU(flops_per_cycle={self.flops_per_cycle}, frequency={self.frequency:.0e}, kappa={self.kappa:.0e})'


@dataclasses.dataclass
class ClientConfig:
    geo_limits: tuple
    transmitter: Transmitter
    cpu: CPU
    location: tuple = dataclasses.field(init=False)
    neighbors: list = dataclasses.field(default_factory=list, init=False)
    peers: list = dataclasses.field(default_factory=list, init=False)
    active: bool = dataclasses.field(default=False, init=False)
    loss_history: tuple = dataclasses.field(default=(math.inf, -math.inf), init=False)

    def __post_init__(self):
        # Set initial location
        (lat_min, lon_min), (lat_max, lon_max) = self.geo_limits
        self.location = (
            random.uniform(lat_min, lat_max),
            random.uniform(lon_min, lon_max),
        )

    def __repr__(self):
        return (f'ClientConfig(location={self.location}, neighbors={self.neighbors}, peers={self.peers}, '
                f'active={self.active}, loss_history={self.loss_history})')
