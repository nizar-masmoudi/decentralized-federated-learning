import torch

from client.dataset.loaders import DeviceDataLoader
from torch.utils.data import DataLoader, random_split
import logging
from client.loggers import SystemLogger, WandbLogger
from client.training.arguments import NetworkTrainerArguments, SelectionTrainerArguments
import itertools
from client.dataset.sampling import DataChunk

logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)


class NetworkTrainer:
    inc = itertools.count(start=1)

    def __init__(self, args: NetworkTrainerArguments, wandb_logger: WandbLogger) -> None:
        self.id_ = next(NetworkTrainer.inc)  # Auto-increment ID
        self.args = args
        self.wandb_logger = wandb_logger

        # W&B - Watch model
        # self.wandb_logger.watch(model=self.model, loss_fn=self.loss_fn, id_=self.id_)

    def __repr__(self):
        return repr(self.args)

    def loss_batch(self, model: torch.nn.Module, batch: tuple):
        inputs, labels = batch
        self.args.optimizer.zero_grad()
        outputs = model(inputs)
        loss = self.args.loss_fn(outputs, labels)
        loss.backward()
        self.args.optimizer.step()
        return loss.item() / len(batch)  # Average loss

    def train(self, model: torch.nn.Module, datachunk: DataChunk, round_: int):
        if self.args.optimizer is None:
            self.args.init_optim(model)

        # Split data for training and validation
        train_ds, valid_ds = NetworkTrainer.train_valid_split(datachunk, valid_split=self.args.valid_split)

        # Prepare dataloaders
        train_dl = DeviceDataLoader(DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True), self.args.device)
        valid_dl = DeviceDataLoader(DataLoader(valid_ds, batch_size=self.args.batch_size, shuffle=True), self.args.device)

        for epoch in range(1, self.args.local_epochs + 1):
            # Training (1 epoch)
            running_tloss = 0.
            model.train()
            for batch in train_dl:
                loss = self.loss_batch(model, batch)
                running_tloss += loss
            avg_tloss = running_tloss / len(train_dl)
            # Validation
            model.eval()
            avg_vloss = self.validate(model, valid_dl)
            # Gather and report
            global_epoch = epoch + (round_*self.args.local_epochs)
            logger.info('Epoch [{:>2}/{:>2}] - Training loss = {:.3f} - Validation loss = {:.3f} - Total epochs = {}'.format(epoch, self.args.local_epochs, avg_tloss, avg_vloss, global_epoch), extra={'client': self.id_})
            self.wandb_logger.log_metrics(metric_dict={'tloss': {f'c{self.id_}': avg_tloss}, 'vloss': {f'c{self.id_}': avg_vloss}}, epoch=epoch, round_=round_)

    def validate(self, model: torch.nn.Module, valid_dl: DeviceDataLoader):
        running_vloss = 0.
        with torch.no_grad():
            for batch in valid_dl:
                inputs, labels = batch
                outputs = model(inputs)
                loss = self.args.loss_fn(outputs, labels)
                running_vloss += loss
            avg_vloss = running_vloss / len(valid_dl)
        return avg_vloss

    def evaluate(self, model: torch.nn.Module, testset: torch.utils.data.Dataset):
        test_dl = DeviceDataLoader(DataLoader(testset, batch_size=self.args.batch_size, shuffle=True), self.args.device)

        running_sloss = 0.
        ground_truth = torch.Tensor([])
        predictions = torch.Tensor([])
        with torch.no_grad():
            for batch in test_dl:
                inputs, labels = batch
                outputs = model(inputs)
                loss = self.args.loss_fn(outputs, labels)
                running_sloss += loss
                ground_truth = torch.cat((ground_truth, labels))
                predictions = torch.cat((predictions, outputs))
        avg_sloss = running_sloss/(len(test_dl)*self.args.batch_size)
        logger.info('Evaluation loss = {:.3f}'.format(avg_sloss), extra={'client': self.id_})
        # self.wandb_logger.pr_curve(ground_truth, predictions)
        return avg_sloss

    @staticmethod
    def train_valid_split(datachunk: DataChunk, valid_split: float):
        valid_split = int((1 - valid_split) * len(datachunk))
        train_split = len(datachunk) - valid_split
        return random_split(datachunk, [valid_split, train_split])


class SelectionTrainer:
    inc = itertools.count(start=1)

    def __init__(self, args: SelectionTrainerArguments):
        self.id_ = next(SelectionTrainer.inc)  # Auto-increment ID
        self.args = args

    def __repr__(self):
        return repr(self.args)

    def train(self, model: torch.nn.Module, energy: torch.Tensor, gain: torch.Tensor):
        pass
