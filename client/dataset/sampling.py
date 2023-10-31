import random
from collections import Counter

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import WeightedRandomSampler


class DataChunk(Subset):
    MIN_SIZE = 800
    MAX_SIZE = 1200

    def __init__(self, dataset: Dataset, size: int = None, iid: bool = True):
        self.size = size or random.randint(DataChunk.MIN_SIZE, DataChunk.MAX_SIZE)
        self.iid = iid

        if hasattr(dataset, 'targets'):
            dataset.targets = torch.tensor(dataset.targets) if not torch.is_tensor(dataset.targets) else dataset.targets
            unique_targets = dataset.targets.unique()
            if self.iid:
                class_weights = torch.ones(size=(len(unique_targets),))
            else:
                class_weights = torch.rand(size=(len(unique_targets),))

            sample_weights = dataset.targets.type(torch.float16).apply_(lambda t: class_weights[int(t)])
            indices = list(WeightedRandomSampler(sample_weights, self.size, replacement=False))
            super().__init__(dataset, indices)
        else:
            raise AttributeError("dataset has no attributes 'targets'")

    def __repr__(self):
        return f'DataChunk(size={self.size}, iid={self.iid})'

    def class_dist(self):
        if hasattr(self.dataset, 'targets'):
            class_distribution = Counter(self.dataset.targets[i].item() for i in self.indices).values()
            return list(class_distribution)
        else:
            raise AttributeError("Dataset has no attributes 'targets'")

    def to_dict(self):
        return {
            'name': self.dataset.__class__.__name__,
            'iid': self.iid,
            'distribution': self.class_dist()
        }
