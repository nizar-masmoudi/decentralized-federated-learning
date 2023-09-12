import itertools
import logging
import math
from typing import List, Dict

import torch.optim
from torch.utils.data import Dataset
from client.dataset.sampling import DataChunk

from client.activation.activators import Activator
from client.aggregation import Aggregator
from client.selection import PeerSelector
from client.training import Trainer
from client.loggers import SystemLogger, WandbLogger
from client.configuration import ClientConfig
import random

logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class Client:
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

        logger.debug(repr(self.config.transmitter), extra={'client': self.id_})
        logger.debug(repr(self.config.cpu), extra={'client': self.id_})
        logger.debug(repr(self.trainer), extra={'client': self.id_})
        logger.debug(repr(self.datachunk), extra={'client': self.id_})
        logger.debug(repr(self), extra={'client': self.id_})

    def __eq__(self, other: 'Client') -> bool:
        return self.id_ == other.id_

    def __repr__(self) -> str:
        return repr(self.config)

    def __str__(self):
        return f'Client(id={self.id_})'

    def relocate(self):
        (lat_min, lon_min), (lat_max, lon_max) = self.config.geo_limits
        self.config.location = (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))
        logger.info(f'Client.relocate()={self.config.location}', extra={'client': self.id_})

    def lookup(self, clients: List['Client'], max_dist: float) -> List['Client']:
        neighbors = [client for client in clients if (client != self) and Client.distance(self, client) < max_dist]
        self.config.neighbors = neighbors
        logger.info('{}=[{}]'.format(f'Client.lookup({max_dist=})', ', '.join(str(neighbor) for neighbor in self.config.neighbors)), extra={'client': self.id_})
        return neighbors

    def learning_slope(self):
        return abs(self.config.loss_history[1] - self.config.loss_history[0])

    def computation_energy(self):
        local_epochs = self.trainer.args.local_epochs
        kappa = self.config.cpu.kappa
        flops = self.model.flops
        ds_size = self.datachunk.size
        cpu_freq = self.config.cpu.frequency
        fpc = self.config.cpu.flops_per_cycle
        energy = local_epochs*kappa*flops*ds_size*(cpu_freq**2)/fpc
        logger.debug(f'Client.computation_energy()={energy:.3f}', extra={'client': self.id_})
        return energy

    def communication_energy(self, peer: 'Client', object_size: float):
        power = Client.dbm_to_mw(self.config.transmitter.power)  # Convert dBm to mW
        bandwidth = self.config.transmitter.bandwidth
        psd = Client.dbm_to_mw(self.config.transmitter.psd)  # Convert dBm to mW
        distance = Client.distance(self, peer) * 1e3  # Convert km to meters
        channel_gain = Client.channel_gain(self.config.transmitter.transmit_gain, self.config.transmitter.receive_gain,
                                           self.config.transmitter.signal_frequency, distance)
        logger.debug(f'Client.channel_gain(peer={peer})={channel_gain:.3f}', extra={'client': self.id_})
        channel_gain = Client.dbm_to_mw(channel_gain)  # Convert dBm to mW
        r = bandwidth * math.log2(1 + (power * channel_gain) / (psd * bandwidth))
        logger.debug(f'Client.transmission_rate(peer={peer})={r:.0e}', extra={'client': self.id_})
        t = object_size / r
        energy = t * Client.dbm_to_mw(power)
        logger.debug(f'Client.communication_energy(peer={peer})={energy:.3e}', extra={'client': self.id_})
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
        if self.activator.__class__.__name__ == 'RandomActivator':
            self.config.active = self.activator.activate()
        elif self.activator.__class__.__name__ == 'EfficientActivator':
            self.config.active = self.activator.activate(self, self.config.neighbors)
        logger.info(f'Client.config.active={self.config.active}', extra={'client': self.id_})

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

    @staticmethod
    def dbm_to_mw(value: float) -> float:
        return 10 ** ((value - 30) / 10)

    @staticmethod
    def channel_gain(transmit_gain: float, receive_gain: float, signal_frequency: float, distance: float) -> float:
        return transmit_gain + receive_gain + 20 * math.log10(3e8 / (4 * math.pi * signal_frequency)) - 20 * math.log10(distance)