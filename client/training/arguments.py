import dataclasses
import torch


@dataclasses.dataclass
class TrainerArguments:
    batch_size: int
    loss_fn: callable
    local_epochs: int
    valid_split: float
    opt_class: type
    optimizer: torch.optim.Optimizer = dataclasses.field(default=None, init=False)
    opt_params: dict = dataclasses.field(default_factory=dict)
    device: torch.device = dataclasses.field(
        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), init=False)

    def init_optim(self, model: torch.nn.Module):
        self.optimizer = self.opt_class(model.parameters(), **self.opt_params)
