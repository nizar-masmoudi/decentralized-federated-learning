import torch

from client.dataset.loaders import DeviceDataLoader
from torch.utils.data import DataLoader, Dataset, random_split
import logging
from client.loggers import ConsoleLogger, WandbLogger
from client.configs import TrainerConfig
from client.dataset.sampling import DataChunk

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, id_: int, model: torch.nn.Module, dataset: DataChunk, testset: torch.utils.data.Dataset, cfg: TrainerConfig, wandb_logger: WandbLogger) -> None:
        self.id_ = id_
        self.model = model
        self.dataset = dataset
        self.testset = testset
        self.cfg = cfg
        self.cfg.setup_optim(model)
        self.wandb_logger = wandb_logger

        # Split data for training and validation
        self.train_ds, self.valid_ds = Trainer.train_valid_split(self.dataset, valid_split=self.cfg.validation_split)

        # Prepare dataloaders
        self.train_dl = DeviceDataLoader(DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True), self.cfg.device)
        self.valid_dl = DeviceDataLoader(DataLoader(self.valid_ds, batch_size=self.cfg.batch_size, shuffle=True), self.cfg.device)
        self.test_dl = DeviceDataLoader(DataLoader(self.testset, batch_size=self.cfg.batch_size, shuffle=True), self.cfg.device)

        # W&B - Watch model
        # self.wandb_logger.watch(model=self.model, loss_fn=self.loss_fn, id_=self.id_)

    def loss_batch(self, batch: tuple):
        inputs, labels = batch
        self.cfg.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.cfg.loss_fn(outputs, labels)
        loss.backward()
        self.cfg.optimizer.step()
        return loss.item() / len(batch)  # Average loss

    def train(self, ridx: int):
        for epoch in range(1, self.cfg.local_epochs + 1):
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
            logger.info('Epoch [{:>2}/{:>2}] - Training loss = {:.3f} - Validation loss = {:.3f} - Total epochs = {}'.format(epoch, self.cfg.local_epochs, avg_tloss, avg_vloss, epoch + (ridx*self.cfg.local_epochs)), extra={'client': self.id_})
            self.wandb_logger.log_metrics(metric_dict={'tloss': {f'c{self.id_}': avg_tloss}, 'vloss': {f'c{self.id_}': avg_vloss}}, epoch=epoch, ridx=ridx)

    def validate(self):
        running_vloss = 0.
        with torch.no_grad():
            for batch in self.valid_dl:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.cfg.loss_fn(outputs, labels)
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
                loss = self.cfg.loss_fn(outputs, labels)
                running_sloss += loss
                ground_truth = torch.cat((ground_truth, labels))
                predictions = torch.cat((predictions, outputs))
        avg_sloss = running_sloss/(len(self.train_dl)*self.cfg.batch_size)
        logger.info('Evaluation loss = {:.3f}'.format(avg_sloss), extra={'client': self.id_})
        self.wandb_logger.pr_curve(ground_truth, predictions)
        return avg_sloss

    @staticmethod
    def train_valid_split(dataset: Dataset, valid_split: float):
        valid_split = int((1 - valid_split) * len(dataset))
        train_split = len(dataset) - valid_split
        return random_split(dataset, [valid_split, train_split])
