import itertools
import logging
import math
from typing import Sequence

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from client.activator import Activator
from client.aggregator import Aggregator
from client.dataset.sampler import DataSample
from client.selector import PeerSelector
from client.trainer import Trainer
from collections import OrderedDict

logger = logging.getLogger(__name__)

class Client:
  inc = itertools.count(start = 1)
  
  def __init__(self, config: dict) -> None:
    self._id = next(Client.inc) # Auto-increment ID
    self.config = config['client']
    self.location = None
    self.neighbors = None
    self.peers = None
    
    # Setup model
    Model = getattr(__import__('client.models', fromlist = [self.config['model']]), self.config['model'])
    self.model = Model()
    logger.debug(f'{Model.__name__} model initialized', extra = {'client': self._id})
    
    # Setup datasets
    self.train_set = DataSample(MNIST(root = 'data', train = True, transform = ToTensor(), download = True), 20000)
    self.test_set = MNIST(root = 'data', train = False, transform = ToTensor(), download = True)
    logger.debug(f'{self.test_set.__class__.__name__} dataset initialized', extra = {'client': self._id})
    
    
    # Modules
    self.trainer = Trainer(
      id = self._id,
      train_set = self.train_set,
      test_set = self.test_set,
      model = self.model,
      config = config['client']['trainer']
    )
    self.aggregator = Aggregator(self.config['aggregator']['policy']['name'])
    self.selector = PeerSelector(self.config['selector']['policy']['name'])
    self.activator = Activator(self.config['activator']['policy']['name'])
    
  def __eq__(self, other: 'Client') -> bool:
    return self._id == other._id
  
  def __repr__(self) -> str:
    return f'Client{self._id}'
    
  def lookup(self, clients: Sequence['Client'], max_dist: float) -> Sequence['Client']:
    """Lookup clients within a certain distance (communication reach).

    Args:
        clients (Sequence['Client']): Sequence of all clients within the network.
        max_dist (float): Maximum distance for communication reach.
        
    Returns:
        Sequence['Client']: Sequence of clients within communication reach (neighbors).
    """
    neighbors = [client for client in clients if (client != self) and Client.distance(self, client) < max_dist]
    self.neighbors = neighbors
    return neighbors
    
  def train(self):
    self.trainer.train()
  
  def aggregate(self, state_dicts: Sequence[OrderedDict]):
    state_dicts.append(self.model.state_dict())
    agg_state = self.aggregator(state_dicts)
    self.model.load_state_dict(agg_state)
  
  def select_peers(self, peers: Sequence[tuple], k: float = None):
    self.peers = self.selector.select_peers(peers, k)
  
  def activate(self):
    self.activator.activate()
    
  @staticmethod
  def distance(client1: 'Client', client2: 'Client') -> float:
    """Compute distance between two clients using their lon - lat coordinates.

    Args:
        client1 (Client): Client 1.
        client2 (Client): Client 2.

    Returns:
        float: Distance between both clients.
    """
    R = 6371.0 # Radius of the Earth in kilometers
    
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
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance
  