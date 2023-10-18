from torch.utils.data import Dataset, Subset
import torch
from torch.utils.data import WeightedRandomSampler
import random
from collections import Counter


class DataChunk(Subset):
    MIN_SIZE = 4000
    MAX_SIZE = 8000

    def __init__(self, dataset: Dataset, size: int = None, balanced_sampling: bool = True):
        self.size = size or random.randint(DataChunk.MIN_SIZE, DataChunk.MAX_SIZE)
        self.balanced_sampling = balanced_sampling

        if hasattr(dataset, 'targets'):
            unique_targets = getattr(dataset, 'targets').unique()
            if self.balanced_sampling:
                class_weights = torch.ones(size=(len(unique_targets),))
            else:
                class_weights = torch.rand(size=(len(unique_targets),))

            sample_weights = getattr(dataset, 'targets').type(torch.float16).apply_(lambda t: class_weights[int(t)])
            indices = list(WeightedRandomSampler(sample_weights, self.size, replacement=False))
            super().__init__(dataset, indices)
        else:
            raise AttributeError("Dataset has no attributes 'targets'")

    def __repr__(self):
        return f'DataChunk(size={self.size}, balanced_sampling={self.balanced_sampling})'

    def class_dist(self):
        if hasattr(self.dataset, 'targets'):
            class_distribution = Counter(self.dataset.targets[i].item() for i in self.indices).values()
            return list(class_distribution)
        else:
            raise AttributeError("Dataset has no attributes 'targets'")

    def to_dict(self):
        return {
            'name': self.dataset.__class__.__name__,
            'iid': self.balanced_sampling,
            'distribution': self.class_dist()
        }
