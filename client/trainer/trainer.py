import torch

from client.dataset.loaders import DeviceDataLoader
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from client.dataset.utils import DataChunk
import logging
from torch.optim import Optimizer
from typing import Callable
from client.loggers import ConsoleLogger, WandbLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 client_id: int,
                 train_ds: DataChunk,
                 test_ds: Dataset,
                 model: nn.Module,
                 optimizer: Optimizer,
                 batch_size: int,
                 loss_fn: Callable,
                 n_epochs: int,
                 wandb_logger: WandbLogger
                 ) -> None:
        self.client_id = client_id
        self.model = model
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.wandb_logger = wandb_logger

        # Setup device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        logger.debug('CUDA available. Device set to CUDA.' if torch.cuda.is_available() else 'CUDA not available. Device set to CPU', extra={'client': self.client_id})
        # Split data for training and validation
        self.train_ds, self.valid_ds = Trainer.train_valid_split(self.train_ds, valid_split=.1)
        logger.debug(f'Length of training subset: {len(self.train_ds)}', extra={'client': self.client_id})
        logger.debug(f'Length of validation subset: {len(self.valid_ds)}', extra={'client': self.client_id})

        # Prepare dataloaders
        self.train_dl = DeviceDataLoader(DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True), self.device)
        self.valid_dl = DeviceDataLoader(DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=True), self.device)
        self.test_dl = DeviceDataLoader(DataLoader(self.test_ds, batch_size=self.batch_size), self.device)

        # W&B - Watch model
        # self.wandb_logger.watch(model=self.model, loss_fn=self.loss_fn, client_id=self.client_id)

    def loss_batch(self, batch: tuple):
        inputs, labels = batch
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item() / len(batch)  # Average loss

    def train(self):
        for epoch in range(1, self.n_epochs + 1):
            # Training (1 epoch)
            running_tloss = 0.
            self.model.train()
            for batch in self.train_dl:
                loss = self.loss_batch(batch)
                running_tloss += loss
            avg_tloss = running_tloss / len(self.train_dl)
            # Validation
            self.model.eval()
            avg_vloss = self.validate()
            # Gather and report
            logger.info('Epoch [{:>2}/{:>2}] - Training loss = {:.3f} - Validation loss = {:.3f}'.format(epoch, self.n_epochs, avg_tloss, avg_vloss), extra={'client': self.client_id})
            self.wandb_logger.log_metrics(metric_dict={'Training loss': {f'Client {self.client_id}': avg_tloss}, 'Validation loss': {f'Client {self.client_id}': avg_vloss}}, epoch=epoch)

    def validate(self):
        running_vloss = 0.
        with torch.no_grad():
            for batch in self.valid_dl:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_vloss += loss
            avg_vloss = running_vloss / len(self.valid_dl)
        return avg_vloss

    def evaluate(self):
        running_sloss = 0.
        ground_truth = torch.Tensor([])
        predictions = torch.Tensor([])
        with torch.no_grad():
            for batch in self.test_dl:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_sloss += loss
                ground_truth = torch.cat((ground_truth, labels))
                predictions = torch.cat((predictions, outputs))
        avg_sloss = running_sloss/(len(self.train_dl)*self.batch_size)
        logger.info('Evaluation loss = {:.3f}'.format(avg_sloss), extra={'client': self.client_id})
        self.wandb_logger.pr_curve(ground_truth, predictions)
        return avg_sloss

    @staticmethod
    def train_valid_split(dataset: Dataset, valid_split: float):
        valid_split = int((1 - valid_split) * len(dataset))
        train_split = len(dataset) - valid_split
        return random_split(dataset, [valid_split, train_split])
