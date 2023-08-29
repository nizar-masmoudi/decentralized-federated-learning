from typing import Callable

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv

load_dotenv()


class WandbLogger:
    def __init__(self, project: str, name: str, group: str = None, config: dict = None):
        self.project = project
        self.name = name
        self.group = group
        self.config = config
        self.run = wandb.init(project=self.project, name=self.name, group=self.group, config=self.config)
        # Defining metrics
        self.run.define_metric('epoch')
        self.run.define_metric('tloss', step_metric='epoch', summary='min', goal='minimize')
        self.run.define_metric('vloss', step_metric='epoch', summary='min', goal='minimize')

    def log_metrics(self, metric_dict: dict, epoch: int):
        self.run.log({**metric_dict, 'epoch': epoch})

    def watch(self, model: nn.Module, loss_fn: Callable, client_id: int):
        self.run.watch(model, loss_fn, log='all', idx=client_id)

    def pr_curve(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        self.run.log({'pr': wandb.plot.pr_curve(ground_truth, prediction)})

    def confusion_matrix(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        self.run.log({'cm': wandb.plot.confusion_matrix(y_true=ground_truth, probs=prediction)})
