import itertools

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from client.dataset.sampler import DataSample

from client.trainer import Trainer
import logging

logger = logging.getLogger(__name__)

class Client:
  inc = itertools.count(start = 1)
  
  def __init__(self, config: dict) -> None:
    self._id = next(Client.inc) # Auto-increment ID
    self.config = config['client']
    
    # Setup model
    Model = getattr(__import__('client.models', fromlist = [self.config['model']]), self.config['model'])
    self.model = Model()
    logger.debug(f'{Model.__name__} model initialized', extra = {'client': self._id})
    
    # Setup datasets
    self.train_set = DataSample(MNIST(root = 'data', train = True, transform = ToTensor(), download = True), 10000)
    self.test_set = MNIST(root = 'data', train = False, transform = ToTensor(), download = True)
    logger.debug(f'Dataset initialized with {len(self.train_set)} train images and {len(self.test_set)} test images', extra = {'client': self._id})
    
    self.location = None
    
    # Modules
    self.trainer = Trainer(
      id = self._id,
      train_set = self.train_set,
      test_set = self.test_set,
      model = self.model,
      config = config['client']['trainer']
    )
    

  def train(self):
    self.trainer.train()
  
  def aggregate(self):
    raise NotImplementedError
  
  def select_peers(self):
    raise NotImplementedError
  
  def activate(self):
    raise NotImplementedError