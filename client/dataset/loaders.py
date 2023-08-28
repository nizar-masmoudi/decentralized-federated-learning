from typing import List, Union

import torch
from torch.utils.data import DataLoader


def to_device(data: Union[torch.Tensor, List[torch.Tensor]], device: torch.device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl: DataLoader, device: torch.device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
