import torch
from client.models import ConvNet
from functools import partial

convnet = ConvNet()

optim = partial(torch.optim.SGD, lr = .1, momentum = 0.9)
print(optim.__class__.__name__)
optim = optim(convnet.parameters())

print(optim)