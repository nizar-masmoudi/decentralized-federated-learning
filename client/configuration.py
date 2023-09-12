import dataclasses
from typing import Sequence
import random
import math


@dataclasses.dataclass
class Transmitter:
    power: float
    bandwidth: float
    psd: float = dataclasses.field(default=-174, init=False)
    signal_frequency: float = dataclasses.field(default=1e9)
    transmit_gain: float = dataclasses.field(default=0)
    receive_gain: float = dataclasses.field(default=0)


@dataclasses.dataclass
class CPU:
    flops_per_cycle: int
    frequency: float
    kappa: float = dataclasses.field(default=1e-28, init=False)


@dataclasses.dataclass
class ClientConfig:
    geo_limits: Sequence[Sequence]
    transmitter: Transmitter
    cpu: CPU
    location: tuple = dataclasses.field(default_factory=tuple, init=False)
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
