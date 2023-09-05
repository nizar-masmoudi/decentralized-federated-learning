import dataclasses
from typing import Sequence


@dataclasses.dataclass
class Transmitter:
    power: float
    bandwidth: float
    psd: float = dataclasses.field(default=-174, init=False)


@dataclasses.dataclass
class CPU:
    cycles: int
    frequency: float
    kappa: float = dataclasses.field(default=10e-28, init=False)


@dataclasses.dataclass
class ClientConfig:
    geo_limits: Sequence[Sequence]
    transmitter: Transmitter
    cpu: CPU
    location: tuple = dataclasses.field(default_factory=tuple, init=False)
    neighbors: list = dataclasses.field(default_factory=list, init=False)
    peers: list = dataclasses.field(default_factory=list, init=False)
    active: bool = dataclasses.field(default=False, init=False)
