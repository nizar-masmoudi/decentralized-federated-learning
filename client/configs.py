import dataclasses
from enum import IntEnum
import torch
from typing import Callable, Sequence
import random
import logging
from client.loggers import ConsoleLogger
import torch.nn as nn
import math

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class AggregationPolicy(IntEnum):
    FEDAVG = 0
    MIXING = 1


class ClientActivationPolicy(IntEnum):
    RANDOM = 0
    FULL = 1
    EFFICIENT = 2


class PeerSelectionPolicy(IntEnum):
    RANDOM = 0
    FULL = 1
    EFFICIENT = 2


@dataclasses.dataclass
class Configuration:
    pass


@dataclasses.dataclass
class TrainerConfig(Configuration):
    opt_class: any  # FIXME
    batch_size: int
    loss_fn: Callable
    local_epochs: int
    validation_split: float
    optimizer: torch.optim.Optimizer = dataclasses.field(default=None, init=False)
    opt_params: dict = dataclasses.field(default_factory=dict)
    device: torch.device = dataclasses.field(default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), init=False)

    def setup_optim(self, model: nn.Module):
        self.optimizer = self.opt_class(model.parameters(), **self.opt_params)


@dataclasses.dataclass
class TransmissionConfig(Configuration):
    transmission_power: float
    bandwidth: float
    psd: float = dataclasses.field(default=-174, init=False)


@dataclasses.dataclass
class ComputationConfig(Configuration):
    cpu_cycles: int
    computation_capacity: float
    effective_capacitance: float = dataclasses.field(default=10e-28, init=False)


@dataclasses.dataclass
class NodeConfig(Configuration):
    geo_limits: Sequence[Sequence]
    learning_slope: float = dataclasses.field(default=999., init=False)
    location: tuple = dataclasses.field(default_factory=tuple, init=False)
    neighbors: list = dataclasses.field(default_factory=list, init=False)
    peers: list = dataclasses.field(default_factory=list, init=False)
    active: bool = dataclasses.field(default=False, init=False)

    def __post_init__(self):
        # Setup initial location
        (lat_min, lon_min), (lat_max, lon_max) = self.geo_limits
        self.location = (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))
        # super().__post_init__()
