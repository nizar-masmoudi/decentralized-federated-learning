import itertools
import logging
import math
from enum import IntEnum
from functools import partial
from typing import List, Dict, Callable

import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset

from client.activator import Activator
from client.aggregator import Aggregator
from client.dataset.utils import DataChunk
from client.selector import PeerSelector
from client.trainer import Trainer
from client.loggers.console import ConsoleLogger
from client.loggers.wandb import WandbLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Client:
    inc = itertools.count(start=1)

    class AggregationPolicy(IntEnum):
        FEDAVG = 0
        MIXING = 1

    class ActivationPolicy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    class SelectionPolicy(IntEnum):
        RANDOM = 0
        FULL = 1
        EFFICIENT = 2

    def __init__(self,
                 # Trainer args
                 train_ds: DataChunk,
                 test_ds: Dataset,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int,
                 loss_fn: Callable,
                 n_epochs: int,
                 # Aggregator args
                 aggregation_policy: AggregationPolicy,
                 # Selector args
                 selection_policy: SelectionPolicy,
                 # Activator args
                 activation_policy: ActivationPolicy,
                 wandb_logger: WandbLogger,
                 ) -> None:
        self.client_id = next(Client.inc)  # Auto-increment ID
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.location = (-1, -1)
        self.is_active = False
        self.neighbors = []
        self.peers = []
        self.wandb_logger = wandb_logger

        # Modules
        self.trainer = Trainer(
            client_id=self.client_id,
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            model=self.model,
            optimizer=self.optimizer,
            batch_size=self.batch_size,
            loss_fn=self.loss_fn,
            n_epochs=self.n_epochs,
            wandb_logger=self.wandb_logger
        )
        self.aggregator = Aggregator(client_id=self.client_id, policy=aggregation_policy)
        self.selector = PeerSelector(client_id=self.client_id, policy=selection_policy)
        self.activator = Activator(client_id=self.client_id, policy=activation_policy)

    def __eq__(self, other: 'Client') -> bool:
        return self.client_id == other.client_id

    def __repr__(self) -> str:
        return f'Client{self.client_id}'

    def lookup(self, clients: List['Client'], max_dist: float) -> List['Client']:
        """Lookup clients within a certain distance (communication reach).

    Args:
        clients (Sequence['Client']): Sequence of all clients within the network.
        max_dist (float): Maximum distance for communication reach.
        
    Returns:
        Sequence['Client']: Sequence of clients within communication reach (neighbors).
    """
        neighbors = [client for client in clients if (client != self) and Client.distance(self, client) < max_dist]
        self.neighbors = neighbors
        logger.info('Detected neighbors: {}'.format(', '.join(str(neighbor) for neighbor in self.neighbors)), extra={'client': self.client_id})
        return neighbors

    def train(self):
        """Run a local training process.
    """
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()

    def aggregate(self, state_dicts: List[Dict]):
        """Aggregate models using predefined policy.

    Args:
        state_dicts (Sequence[OrderedDict]): State dicts of models to aggregate with.
    """
        state_dicts.append(self.model.state_dict())
        agg_state = self.aggregator.aggregate(state_dicts)
        self.model.load_state_dict(agg_state)

    def select_peers(self, k: float = None):
        """Select peers from neighboring nodes using a predefined policy.

    Args: k (float, optional): Number of peers to select. This parameter is relevant only when using Random Selection
    policy. Defaults to None.
        """
        self.peers = self.selector.select_peers(self.neighbors, k)

    def activate(self):
        self.is_active = self.activator.activate()

    @staticmethod
    def distance(client1: 'Client', client2: 'Client') -> float:
        """Compute distance between two clients using their lon - lat coordinates.

    Args:
        client1 (Client): Client 1.
        client2 (Client): Client 2.

    Returns:
        float: Distance between both clients.
    """
        radius = 6371.0  # Radius of the Earth in kilometers

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
