import itertools
import logging
import math
from enum import IntEnum
from typing import List, Dict

import torch.optim
from torch.utils.data import Dataset
from client.dataset.sampling import DataChunk

from client.activation import Activator
from client.aggregation import Aggregator
from client.selection import PeerSelector
from client.training import Trainer
from client.loggers import ConsoleLogger, WandbLogger
from client.configuration import ClientConfig
import random

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Client:
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

    inc = itertools.count(start=1)

    def __init__(self,
                 model: torch.nn.Module,
                 datachunk: DataChunk,
                 testset: torch.utils.data.Dataset,
                 trainer: Trainer,
                 aggregator: Aggregator,
                 activator: Activator,
                 selector: PeerSelector,
                 config: ClientConfig,
                 wandb_logger: WandbLogger,
                 ) -> None:
        self.id_ = next(Client.inc)  # Auto-increment ID
        self.model = model
        self.datachunk = datachunk
        self.testset = testset
        self.trainer = trainer
        self.aggregator = aggregator
        self.activator = activator
        self.selector = selector
        self.config = config
        self.wandb_logger = wandb_logger

    def __eq__(self, other: 'Client') -> bool:
        return self.id_ == other.id_

    def __repr__(self) -> str:
        return f'Client(id={self.id_})'

    def relocate(self):
        (lat_min, lon_min), (lat_max, lon_max) = self.config.geo_limits
        location = self.config.location
        self.config.location = (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))
        logger.info(f'Client relocated from {location} to {self.config.location}', extra={'client': self.id_})

    def lookup(self, clients: List['Client'], max_dist: float) -> List['Client']:
        neighbors = [client for client in clients if (client != self) and Client.distance(self, client) < max_dist]
        self.config.neighbors = neighbors
        logger.info('{}=[{}]'.format('Client.lookup(...)', ', '.join(repr(neighbor) for neighbor in self.config.neighbors)), extra={'client': self.id_})
        return neighbors

    def computation_energy(self):
        i = self.trainer.args.local_epochs
        k = self.config.cpu.kappa
        c = self.config.cpu.cycles
        f = self.config.cpu.frequency
        d = len(self.datachunk)
        energy = i*k*c*d*(f**2)
        logger.debug(f'Client.computation_energy(...)={energy}', extra={'client': self.id_})
        return energy

    def communication_energy(self, peer: 'Client', object_size: float):
        g = 1/Client.distance(self, peer)**2
        p = self.config.transmitter.power
        b = self.config.transmitter.bandwidth
        n = self.config.transmitter.psd
        s = object_size
        energy = (s*p) / (b*math.log2(1 + (p*g) / (n*b)))
        logger.debug(f'Client.communication_energy(...)={energy}', extra={'client': self.id_})
        return energy

    def train(self, round_: int):
        self.trainer.train(model=self.model, datachunk=self.datachunk, round_=round_)

    def evaluate(self):
        self.trainer.evaluate(model=self.model, testset=self.testset)

    def aggregate(self, state_dicts: List[Dict]):
        state_dicts.append(self.model.state_dict())
        agg_state = self.aggregator.aggregate(state_dicts)
        self.model.load_state_dict(agg_state)

    def select_peers(self):
        self.config.peers = self.selector.select(self.config.neighbors)

    def activate(self):
        self.config.active = self.activator.activate()

    @staticmethod
    def distance(client1: 'Client', client2: 'Client') -> float:
        radius = 6371.0  # Radius of the Earth in kilometers

        lat1, lon1 = client1.config.location
        lat2, lon2 = client2.config.location

        # Convert degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        # Differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = radius * c

        return distance
