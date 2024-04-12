import itertools
import logging
import math
import random
from typing import List
from typing import Tuple

from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from torch.utils.data import Dataset, DataLoader

from client.activation.activator import Activator
from client.aggregation.aggregator import Aggregator
from client.components import CPU, Transmitter
from client.dataset.utils import train_valid_split
from client.loggers import ConsoleLogger, JSONLogger
from client.selection.selector import PeerSelector

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Client:
    """
    Client class.
    """
    inc = itertools.count(start=1)

    def __init__(
            self,
            *,
            geo_limits: Tuple[Tuple, Tuple],
            model: LightningModule,
            datachunk: Dataset,
            testset: Dataset,
            local_epochs: int,
            batch_size: int = 32,
            valid_split: float = .1,
            cpu: CPU = None,
            transmitter: Transmitter = None,
            activator: Activator,
            aggregator: Aggregator,
            selector: PeerSelector,
            json_logger: JSONLogger = None
    ):
        """
        Initialize a client.
        :param geo_limits: Geographical limits within which clients move, represented as a tuple of tuple,
        (lat_min, lon_min), (lat_max, lon_max).
        :param model: An implemented PyTorch Lightning Module.
        :param datachunk: A PyTorch Dataset. A portion of it will later be reserved for validation.
        :param testset: A PyTorch Dataset used to test the client's local model.
        :param local_epochs: Local training epochs per round.
        :param batch_size: Dataloaders' batch size.
        :param valid_split: Validation split will be taken from train_ds.
        :param cpu: Client's CPU component. If not specified, a CPU with random specifications will be generated.
        :param transmitter: Client's Transmitter component. If not specified, a Transmitter with random specifications
        will be generated.
        :param activator: Client activator.
        :param aggregator: Model aggregator.
        :param selector: Peer selector.
        :param json_logger: JSONLogger instance.
        """
        self.id_ = next(Client.inc)

        self.model = model
        self.datachunk = datachunk
        self.testset = testset

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.valid_split = valid_split

        self.cpu = cpu or CPU()
        self.transmitter = transmitter or Transmitter()
        logger.debug(repr(self.cpu), extra={'id': self.id_})
        logger.debug(repr(self.transmitter), extra={'id': self.id_})

        self.activator = activator
        self.aggregator = aggregator
        self.selector = selector

        self.geo_limits = geo_limits
        self.lookup_dist = 2
        self.location = (0., 0.)
        self.neighbors: List['Client'] = []
        self.peers: List['Client'] = []
        self.is_active = False

        self.json_logger = json_logger

        logger.debug(repr(self), extra={'id': self.id_})
        self.json_logger.add_client(self)

    def __eq__(self, other: 'Client') -> bool:
        return self.id_ == other.id_

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'id={self.id_}, '
                f'location={self.location}, '
                f'is_active={self.is_active}, '
                f'neighbors={list(map(lambda e: str(e), self.neighbors))}, '
                f'peers={list(map(lambda e: str(e), self.peers))})')

    def __str__(self):
        return f'Client{self.id_}'

    def relocate(self):
        """
        Update client location within specified geo_limits.
        """
        (lat_min, lon_min), (lat_max, lon_max) = self.geo_limits
        self.location = (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))

        logger.info('Client relocated to {}, {}'.format(*self.location), extra={'id': self.id_})
        self.json_logger.log_location(self)

    def lookup(self, clients: List['Client']):
        """
        Lookup clients within communication distance.
        :param clients: List of all clients.
        :return: List of neighbors.
        """
        neighbors = [client for client in clients
                     if (client != self) and (Client.distance(self, client) < self.lookup_dist)]
        self.neighbors = neighbors
        nbrs_str = ', '.join(str(neighbor) for neighbor in self.neighbors)

        if self.neighbors:
            logger.info('Client found {} clients nearby: {}'.format(len(self.neighbors), nbrs_str),
                        extra={'id': self.id_})
        else:
            logger.info('Client found 0 clients nearby', extra={'id': self.id_})
        self.json_logger.log_neighbors(self)

    def knowledge_gain(self, neighbor: 'Client') -> float:
        """
        Compute knowledge gain with neighbor.
        :param neighbor: Neighbor to compute KG with.
        :return: Knowledge gain.
        """
        ln = self.model.loss_history[-1]
        lk = neighbor.model.loss_history[-1]
        delta = lk - ln
        return (1 - math.exp(-2 * delta)) * (delta > 0)

    def communication_energy(self, peer: 'Client' = None, distance: float = None) -> float:
        """
        Calculate communication energy consumed by client when communicating with specified peer.
        :param peer: Peer communication with client (used only when distance is None).
        :param distance: Distance between client and peer (used only when peer is None).
        :return: Communication energy
        """
        if peer is None and distance is None:
            raise ValueError('peer and distance cannot be both None.')

        distance = Client.distance(self, peer) if peer else distance
        channel_gain = (self.transmitter.transmit_gain + self.transmitter.receive_gain +
                        20 * math.log10(3e8 / (4 * math.pi * self.transmitter.signal_frequency)) -
                        20 * math.log10(distance))
        rate = (self.transmitter.bandwidth *
                math.log2(1 + (Client.dbm_to_mw(self.transmitter.power) * Client.dbm_to_mw(channel_gain)) /
                          (Client.dbm_to_mw(self.transmitter.psd) * self.transmitter.bandwidth)))
        time = self.model.size_bytes / rate
        energy = time * Client.dbm_to_mw(self.transmitter.power)
        logger.debug('Peer = {}: Distance = {:.3f} - Channel gain = {:.3f} dB - '
                     'Transmission rate = {}/s - '
                     'Communication energy = {:.3f} µW'.format(peer, distance, channel_gain, Client.format_size(rate),
                                                               energy * 1e3),
                     extra={'id': self.id_})
        return energy

    def train(self):
        """
        Run a local training round.
        """
        trainer = Trainer(max_epochs=self.local_epochs, enable_checkpointing=False, logger=self.json_logger,
                          log_every_n_steps=1, enable_model_summary=False, enable_progress_bar=False)
        logger.info('Local training process started', extra={'id': self.id_})
        train_ds, valid_ds = train_valid_split(self.datachunk, .1)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size)
        valid_dl = DataLoader(valid_ds, batch_size=self.batch_size)
        trainer.fit(self.model, train_dl, valid_dl)
        logger.info('Local training process ended', extra={'id': self.id_})

    def test(self):
        """
        Run a local test round.
        """
        trainer = Trainer(max_epochs=self.local_epochs, enable_checkpointing=False, logger=self.json_logger,
                          log_every_n_steps=1, enable_model_summary=False, enable_progress_bar=False)
        logger.info('Model testing process started', extra={'id': self.id_})
        test_dl = DataLoader(self.testset, batch_size=self.batch_size)
        trainer.test(self.model, test_dl, verbose=False)
        logger.info('Model testing process ended', extra={'id': self.id_})

    def aggregate(self, models: List[LightningModule]):
        """
        Aggregate local model with shared models.
        :param models: Models shared by selected neighbors.
        """
        logger.info('Model aggregation process started with peers: {}'.format(
            ', '.join([f'Client{model.id_}' for model in models])), extra={'id': self.id_})

        state_dicts = [self.model.state_dict()] + [model.state_dict() for model in models]
        agg_state = self.aggregator.aggregate(state_dicts)
        self.model.load_state_dict(agg_state)

        logger.info('Model aggregation process ended', extra={'id': self.id_})

    def activate(self):
        """
        Activate client to participate in the next round.
        """
        logger.info('Client activation process started', extra={'id': self.id_})
        is_active = self.activator.activate(self)
        self.is_active = is_active

        logger.info('Client has been activated' if is_active else 'Client has been deactivated', extra={'id': self.id_})
        self.json_logger.log_activity(self)

    def select_peers(self):
        """
        Amongst neighbors, select peers to share local model with.
        """
        logger.info('Peer selection process started', extra={'id': self.id_})
        peers = self.selector.select(self)
        peers_str = ', '.join(str(peer) for peer in peers)
        self.peers = peers

        logger.info('Client selected {} peers: {}'.format(len(peers), peers_str), extra={'id': self.id_})
        self.json_logger.log_peers(self)

    # Utilities
    @staticmethod
    def distance(client1: 'Client', client2: 'Client') -> float:
        # Radius of the Earth in kilometers
        radius = 6371.0
        # Get locations
        lat1, lon1 = client1.location
        lat2, lon2 = client2.location
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
    def format_size(bits: float):
        units = ['bits', 'KB', 'MB', 'GB']
        unit_size = 1e3
        unit_index = 0
        size = bits
        while size >= unit_size and unit_index < len(units) - 1:
            size /= unit_size
            unit_index += 1
        return '{:.2f} {}'.format(size, units[unit_index])

    @staticmethod
    def dbm_to_mw(dbm: float):
        return 10 ** ((dbm - 30) / 10)

    @staticmethod
    def fill_gaps(input_list: list, mask: list):
        p = []
        i = 0
        last_item = input_list[0]
        for m in mask:
            if m:
                p += input_list[i:i+3]
                last_item = input_list[i+2]
                i += 3
            else:
                p += [last_item]*3
        return p
