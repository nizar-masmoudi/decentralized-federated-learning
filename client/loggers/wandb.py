from typing import Callable, Sequence

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from itertools import permutations
import pandas as pd

load_dotenv()


class WandbLogger:
    def __init__(self, project: str, name: str, group: str = None, config: dict = None):
        self.project = project
        self.name = name
        self.group = group
        self.config = config
        self.run = wandb.init(project=self.project, name=self.name, group=self.group, config=self.config)
        # Defining metrics
        # self.run.define_metric('epoch')
        # self.run.define_metric('tloss', step_metric='epoch', summary='min', goal='minimize')
        # self.run.define_metric('vloss', step_metric='epoch', summary='min', goal='minimize')

    def log_metrics(self, metric_dict: dict, epoch: int, round_: int):
        self.run.log({**metric_dict, 'epoch': epoch, 'round': round_})

    def watch(self, model: nn.Module, loss_fn: Callable, client_id: int):
        self.run.watch(model, loss_fn, log='all', idx=client_id)

    def pr_curve(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        self.run.log({'pr': wandb.plot.pr_curve(ground_truth, prediction)})

    def confusion_matrix(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        self.run.log({'cm': wandb.plot.confusion_matrix(y_true=ground_truth, probs=prediction)})

    def distance_matrix(self, clients: Sequence, distance_fn: Callable):
        dists = [[.0 for _ in range(len(clients))] for _ in range(len(clients))]
        for ci, cj in permutations(clients, 2):
            dists[clients.index(ci)][clients.index(cj)] = distance_fn(ci, cj)
        dist_matrix = pd.DataFrame(dists, columns=[str(client) for client in clients]).round(3)
        dist_matrix.insert(loc=0, column='', value=pd.Series(f'**{client}**' for client in clients))
        self.run.log({'dists': wandb.Table(dataframe=dist_matrix)})

    def finish(self):
        self.run.finish()
