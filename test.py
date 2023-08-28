import torch
from client.models import ConvNet
from functools import partial

convnet = ConvNet()
print(type(convnet.state_dict()))

print(torch.device('cpu').__class__)

optim = partial(torch.optim.SGD, lr=.1, momentum=0.9)
print(optim.__class__.__name__)
optim = optim(convnet.parameters())

print(optim)
