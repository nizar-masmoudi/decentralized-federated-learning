import torch
from torch.utils.data import TensorDataset

from client.dataset.utils import train_valid_split


def test_train_valid_split():
    size = 1000
    dataset = TensorDataset(torch.rand(size))
    train_ds, valid_ds = train_valid_split(dataset, .1)
    assert len(train_ds) == 900 and len(valid_ds) == 100
