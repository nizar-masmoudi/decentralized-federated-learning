from client.training.arguments import TrainerArguments
import torch

args = TrainerArguments(
    batch_size=32,
    loss_fn=torch.nn.CrossEntropyLoss(),
    local_epochs=3,
    valid_split=.1,
    opt_class=torch.optim.SGD,
    opt_params=dict(lr=.01, momentum=.9)
)
print(repr(args))
