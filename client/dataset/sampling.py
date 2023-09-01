import random

from torch.utils.data import Dataset, Subset
import torch
from torch.utils.data import WeightedRandomSampler


class DataChunk(Subset):
    def __init__(self, dataset: Dataset, size: int, equal: bool = True):
        unique_targets = dataset.targets.unique()
        class_weights = torch.ones(size=(len(unique_targets),)) if equal else torch.rand(size=(len(unique_targets),))
        sample_weights = dataset.targets.type(torch.float16).apply_(lambda t: class_weights[int(t)])
        indices = list(WeightedRandomSampler(sample_weights, size, replacement=False))
        super().__init__(dataset, indices)
