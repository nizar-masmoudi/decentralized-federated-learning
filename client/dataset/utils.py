import random
from typing import List, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class DataChunk(Subset):
  def __init__(self, dataset: Dataset, size: int) -> None:
    indices = random.sample(range(0, len(dataset)), size)
    super().__init__(dataset, indices)

def to_device(data: Union[torch.Tensor, List[torch.Tensor]], device: str):
  if isinstance(data, (list, tuple)):
      return [to_device(x, device) for x in data]
  elif isinstance(data, dict):
    return {k: to_device(v, device) for k, v in data.items()}
  return data.to(device, non_blocking = True)

class DeviceDataLoader:
  def __init__(self, dl: DataLoader, device: str):
      self.dl = dl
      self.device = device
  def __iter__(self):
      for b in self.dl:
          yield to_device(b, self.device)
  def __len__(self):
      return (len(self.dl))