import random

from torch.utils.data import Dataset, Subset

class DataSample(Subset):
  def __init__(self, dataset: Dataset, size: int) -> None:
    indices = random.sample(range(0, len(dataset)), size)
    super().__init__(dataset, indices)