import copy
import itertools
import logging
import math

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy
from torchtnt.utils.flops import FlopTensorDispatchMode
from torchtnt.utils.module_summary import get_module_summary

from client.loggers import ConsoleLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class CIFAR10Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256 * 4 * 4, 1024)
        self.relu7 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.relu8 = nn.ReLU()
        self.linear3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu7(x)
        x = self.linear2(x)
        x = self.relu8(x)
        x = self.linear3(x)
        return x

    def count_flops(self):
        input_ = torch.randn(1, 3, 32, 32)
        with FlopTensorDispatchMode(self) as ftdm:
            # count forward flops
            output = self(input_).mean()
            ff_dict = copy.deepcopy(ftdm.flop_counts)
            flops_forward = ff_dict['']['convolution.default']
            flops_forward += ff_dict['']['addmm.default']

            # reset count before counting backward flops
            ftdm.reset()
            output.backward()
            fb_dict = copy.deepcopy(ftdm.flop_counts)
            flops_backward = fb_dict['']['convolution_backward.default']
            flops_backward += fb_dict['']['mm.default']
        return flops_forward, flops_backward


class LightningCIFAR10(LightningModule):
    inc = itertools.count(start=1)

    def __init__(self):
        super().__init__()
        self.id_ = next(LightningCIFAR10.inc)

        self.model = CIFAR10Model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.accuracy = Accuracy(task='multiclass', num_classes=10)

        self.training_step_outputs = {'loss': [], 'acc': []}
        self.validation_step_outputs = {'loss': [], 'acc': []}
        self.test_step_outputs = {'loss': [], 'acc': [], 'cm': []}
        self.current_round = 1

        self.loss_history = (math.inf, math.inf)

        summary = get_module_summary(self)
        self.size_bytes = summary.size_bytes
        self.flops = sum(self.model.count_flops())

    def __repr__(self):
        summary = get_module_summary(self)
        flops_forward, flops_backward = self.model.count_flops()
        return (f'{self.__class__.__name__}('
                f'total_parameters={summary.num_parameters}, '
                f'trainable_parameters={summary.num_trainable_parameters}, '
                f'model_size={summary.size_bytes}, '
                f'flops_forward={flops_forward}, '
                f'flops_backward={flops_backward}, '
                f'loss_fn={self.loss_fn.__class__.__name__}(), '
                f'optimizer={self.optimizer.__class__.__name__}())')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.float:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.training_step_outputs['loss'].append(loss)
        self.training_step_outputs['acc'].append(acc)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.float:
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.validation_step_outputs['loss'].append(loss)
        self.validation_step_outputs['acc'].append(acc)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.float:
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.test_step_outputs['loss'].append(loss)
        self.test_step_outputs['acc'].append(acc)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    # Hooks
    def on_train_epoch_end(self) -> None:
        # Gather
        toutputs = self.training_step_outputs
        voutputs = self.validation_step_outputs
        epoch_tloss = torch.tensor(toutputs['loss']).mean()
        epoch_tacc = torch.tensor(toutputs['acc']).mean()
        epoch_vloss = torch.tensor(voutputs['loss']).mean()
        epoch_vacc = torch.tensor(voutputs['acc']).mean()
        self.training_step_outputs = {'loss': [], 'acc': []}
        self.validation_step_outputs = {'loss': [], 'acc': []}
        # Report
        self.logger.log_metrics({
            f'{self.id_}/train/loss': epoch_tloss.item(),
            f'{self.id_}/train/accuracy': epoch_tacc.item(),
            f'{self.id_}/valid/loss': epoch_vloss.item(),
            f'{self.id_}/valid/accuracy': epoch_vacc.item(),
        })
        logger.info(
            'Epoch [{:>2}/{:>2}] - '.format(self.current_epoch + 1, self.trainer.max_epochs) +
            'Training loss = {:.3f} - Training accuracy = {:.3f} - '.format(epoch_tloss, epoch_tacc) +
            'Validation loss = {:.3f} - Validation accuracy = {:.3f}'.format(epoch_vloss, epoch_vacc),
            extra={'id': self.id_}
        )

    def on_train_end(self) -> None:
        self.current_round += 1

    def on_test_end(self) -> None:
        # Gather
        outputs = self.test_step_outputs
        avg_loss = torch.tensor(outputs['loss']).mean()
        avg_acc = torch.tensor(outputs['acc']).mean()
        self.test_step_outputs = {'loss': [], 'acc': [], 'cm': []}
        # Report
        self.logger.log_metrics({
            f'{self.id_}/test/loss': avg_loss.item(),
            f'{self.id_}/test/accuracy': avg_acc.item(),
        })
        logger.info('Test loss = {:.3f} - Test accuracy = {:.3f}'.format(avg_loss, avg_acc), extra={'id': self.id_})
        # Update slope
        self.loss_history = (self.loss_history[1], avg_loss)
