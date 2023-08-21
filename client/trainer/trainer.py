import torch

from client.dataset.loaders import DeviceDataLoader
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from client.dataset.sampler import DataSample
import logging

logger = logging.getLogger(__name__)

class Trainer:
  def __init__(self, id: int, train_set: DataSample, test_set: Dataset, model: nn.Module, config: dict) -> None:
    self._id = id
    self.train_set = train_set
    self.test_set = test_set
    self.model = model
    self.config = config
    
    # Setup optimizer
    Optimizer = getattr(__import__('torch.optim', fromlist = [self.config['optimizer']['class']]), self.config['optimizer']['class'])
    self.optimizer = Optimizer(self.model.parameters(), **{k: v for k, v in self.config['optimizer'].items() if k != 'class'})
    logger.debug(f'{Optimizer.__name__} optimizer initialized with params ' + ', '.join([f'{k} = {v}' for k, v in self.config['optimizer'].items() if k != 'class']), extra = {'client': self._id})
    
    # Setup Loss function
    Loss = getattr(__import__('torch.nn', fromlist = [self.config['loss_fn']]), self.config['loss_fn'])
    self.loss_fn = Loss()
    logger.debug(f'{Loss.__name__} function initialized', extra = {'client': self._id})
    
    # Setup device
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.debug('CUDA available. Device set to CUDA' if torch.cuda.is_available() else 'CUDA not available. Device set to CPU', extra = {'client': self._id})
    
    # Split data for training and validation
    self.train_set, self.valid_set = Trainer.train_valid_split(self.train_set, valid_split = .1)
    logger.debug(f'Training and validation splits setup with lengths {len(self.train_set)} and {len(self.valid_set)} respectively', extra = {'client': self._id})
    
    # Prepare dataloaders
    self.train_dl = DeviceDataLoader(DataLoader(self.train_set, batch_size = self.config['batch_size'], shuffle = True), self.device)
    self.valid_dl = DeviceDataLoader(DataLoader(self.valid_set, batch_size = self.config['batch_size'], shuffle = True), self.device)
    self.test_dl = DeviceDataLoader(DataLoader(self.test_set, batch_size = self.config['batch_size']), self.device)
    
  def loss_batch(self, batch: tuple):
    inputs, labels = batch
    self.optimizer.zero_grad()
    outputs = self.model(inputs)
    loss = self.loss_fn(outputs, labels)
    loss.backward()
    self.optimizer.step()
    return loss.item()/len(batch) # Average loss
  
  def train(self):
    for epoch in range(1, self.config['n_epochs'] + 1):
      # Training (1 epoch)
      running_tloss = 0.
      self.model.train()
      for batch in self.train_dl:
        loss = self.loss_batch(batch)
        running_tloss += loss
      avg_tloss = running_tloss/len(self.train_dl)
      # Validation
      self.model.eval()
      avg_vloss = self.validate(self.valid_dl)
      # Gather and report
      logger.info('Epoch [{:>2}/{:>2}] - Training loss = {:.3f} - Validation loss = {:.3f}'.format(epoch, self.config['n_epochs'], avg_tloss, avg_vloss), extra = {'client': self._id})
  
  def validate(self, valid_dl: DeviceDataLoader):
    running_vloss = 0.
    with torch.no_grad():
      for batch in valid_dl:
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        running_vloss += loss
      avg_vloss = running_vloss/len(valid_dl)
    return avg_vloss
  
  def test(self):
    raise NotImplementedError
  
  @staticmethod
  def train_valid_split(dataset: Dataset, valid_split: float):
    valid_split = int((1 - valid_split)*len(dataset))
    train_split = len(dataset) - valid_split
    return random_split(dataset, [valid_split, train_split])