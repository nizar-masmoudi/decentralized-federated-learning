from torch.utils.data import Dataset, Subset
import torch
from torch.utils.data import WeightedRandomSampler


class DataChunk(Subset):
    def __init__(self, dataset: Dataset, size: int, eq_dist: bool = True):
        self.size = size
        self.eq_dist = eq_dist

        if hasattr(dataset, 'targets'):
            unique_targets = getattr(dataset, 'targets').unique()
            class_weights = torch.ones(size=(len(unique_targets),)) if eq_dist else torch.rand(size=(len(unique_targets),))
            sample_weights = getattr(dataset, 'targets').type(torch.float16).apply_(lambda t: class_weights[int(t)])
            indices = list(WeightedRandomSampler(sample_weights, size, replacement=False))
            super().__init__(dataset, indices)
        else:
            raise AttributeError("Dataset has no attributes 'targets'")

    def __repr__(self):
        return f'{self.__class__.__name__}(size={self.size}, eq_dist={self.eq_dist})'
