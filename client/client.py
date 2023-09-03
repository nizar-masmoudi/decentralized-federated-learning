import itertools
import logging
import math
from enum import IntEnum
from typing import List, Dict

import torch.optim
from torch.utils.data import Dataset
from client.dataset.sampling import DataChunk

from client.activation import ClientActivator
from client.aggregator import Aggregator
from client.selector import PeerSelector
from client.trainer import Trainer
from client.loggers import ConsoleLogger, WandbLogger
import random
from client.configs import TrainerConfig, NodeConfig, TransmissionConfig, ComputationConfig

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Client:
    inc = itertools.count(start=1)

    def __init__(self,
                 metadata: NodeConfig,
                 dataset: DataChunk,
                 testset: torch.utils.data.Dataset,
                 model: torch.nn.Module,
                 trainer_cfg: TrainerConfig,
                 aggregation_policy: Aggregator.Policy,
                 selection_policy: PeerSelector.Policy,
                 activation_policy: ClientActivator.Policy,
                 trans_cfg: TransmissionConfig,
                 comp_cfg: ComputationConfig,
                 wandb_logger: WandbLogger,
                 ) -> None:
        self.id_ = next(Client.inc)  # Auto-increment ID
        self.dataset = dataset
        self.testset = testset
        self.model = model
        self.metadata = metadata
        self.trainer_cfg = trainer_cfg
        self.trans_cfg = trans_cfg
        self.comp_cfg = comp_cfg
        self.wandb_logger = wandb_logger

        # Modules
        self.trainer = Trainer(self.id_, self.model, self.dataset, self.testset, self.trainer_cfg, self.wandb_logger)
        self.aggregator = Aggregator(self.id_, aggregation_policy)
        self.selector = PeerSelector(self.id_, selection_policy, trans_cfg)
        self.activator = ClientActivator(self.id_, activation_policy, comp_cfg)

    def __eq__(self, other: 'Client') -> bool:
        return self.id_ == other.id_

    def __repr__(self) -> str:
        return f'Client(id={self.id_})'

    def relocate(self):
        (lat_min, lon_min), (lat_max, lon_max) = self.metadata.geo_limits
        location = self.metadata.location
        self.metadata.location = (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))
        logger.info(f'Client relocated from {location} to {self.metadata.location}', extra={'client': self.id_})

    def lookup(self, clients: List['Client'], max_dist: float) -> List['Client']:
        """Lookup neighbors within a certain distance (communication reach)"""
        neighbors = [client for client in clients if (client != self) and Client.distance(self, client) < max_dist]
        self.metadata.neighbors = neighbors
        logger.info('{}=[{}]'.format('Client.lookup(...)', ', '.join(repr(neighbor) for neighbor in self.metadata.neighbors)), extra={'client': self.id_})
        return neighbors

    def computation_energy(self):
        energy = self.trainer_cfg.local_epochs * self.comp_cfg.effective_capacitance * self.comp_cfg.cpu_cycles * len(self.dataset) * self.comp_cfg.computation_capacity**2
        logger.debug(f'Client.computation_energy(...)={energy}', extra={'client': self.id_})
        return energy

    def communication_energy(self, peer: 'Client', object_size: float):
        channel_gain = Client.distance(self, peer)
        energy = (object_size * self.trans_cfg.transmission_power) / (self.trans_cfg.bandwidth * math.log2(1 + (self.trans_cfg.transmission_power*channel_gain)/(self.trans_cfg.psd*self.trans_cfg.bandwidth)))
        logger.debug(f'Client.communication_energy(...)={energy}', extra={'client': self.id_})
        return energy

    def train(self, ridx: int):
        """Run a local training process"""
        self.trainer.train(ridx)

    def evaluate(self):
        self.trainer.evaluate()

    def aggregate(self, state_dicts: List[Dict]):
        """Aggregate models using predefined policy"""
        state_dicts.append(self.model.state_dict())
        agg_state = self.aggregator.aggregate(state_dicts)
        self.model.load_state_dict(agg_state)

    def select_peers(self, k: float = None):
        """Select peers from neighboring nodes using a predefined policy"""
        self.metadata.peers = self.selector.select_peers(self.metadata.neighbors, k)

    def activate(self):
        self.metadata.active = self.activator.activate()

    @staticmethod
    def distance(client1: 'Client', client2: 'Client') -> float:
        """Compute distance between two clients using their lon - lat coordinates"""
        radius = 6371.0  # Radius of the Earth in kilometers

        lat1, lon1 = client1.metadata.location
        lat2, lon2 = client2.metadata.location

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
