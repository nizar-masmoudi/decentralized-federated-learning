from torch.utils.data import Dataset, Subset
import torch
from torch.utils.data import WeightedRandomSampler


class DataChunk(Subset):
    def __init__(self, dataset: Dataset, size: int, balanced_sampling: bool = True):
        self.size = size
        self.balanced_sampling = balanced_sampling

        if hasattr(dataset, 'targets'):
            unique_targets = getattr(dataset, 'targets').unique()
            if self.balanced_sampling:
                class_weights = torch.ones(size=(len(unique_targets),))
            else:
                class_weights = torch.rand(size=(len(unique_targets),))
            sample_weights = getattr(dataset, 'targets').type(torch.float16).apply_(lambda t: class_weights[int(t)])
            indices = list(WeightedRandomSampler(sample_weights, size, replacement=False))
            super().__init__(dataset, indices)
        else:
            raise AttributeError("Dataset has no attributes 'targets'")

    def __repr__(self):
        return f'DataChunk(size={self.size}, balanced_sampling={self.balanced_sampling})'
