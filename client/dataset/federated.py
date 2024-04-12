import numpy as np
import torch
from torch.utils.data import Dataset


class DataChunkGenerator:
    def __init__(self, dataset: Dataset, size: int, alpha: float = 1000):
        """
        Data chunk generator class used to sample data chunks following Dirichlet distribution.
        :param dataset: Dataset to sample from.
        :type size: Number of datachunks to sample.
        :param alpha: Dirichlet parameter.
        """
        self.dataset = dataset
        self.size = size
        self.alpha = alpha
        self.chunkz = [[] for _ in range(size)]

        if hasattr(dataset, 'targets'):
            targets = dataset.targets.numpy() if torch.is_tensor(dataset.targets) else np.array(dataset.targets)
            for target in np.unique(targets):
                indices = np.argwhere(targets == target)

                weights = np.random.dirichlet(np.ones(self.size) * self.alpha, 1).flatten()
                weights = weights / weights.sum()
                counts = (weights * len(indices)).round().astype(np.int16)
                target_chunkz = np.split(indices, np.cumsum(counts)[:-1])

                assert len(target_chunkz) == self.size, f'{len(target_chunkz)} != {self.size}'

                for i in range(size):
                    self.chunkz[i] += target_chunkz[i].flatten().tolist()

        else:
            raise AttributeError("dataset has no attributes 'targets'")

    def __iter__(self):
        yield from self.chunkz
