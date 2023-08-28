import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2 = nn.Dropout2d(.5)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(.5)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
