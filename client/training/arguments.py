import dataclasses
import torch


@dataclasses.dataclass
class NetworkTrainerArguments:
    batch_size: int
    loss_fn: callable
    local_epochs: int
    valid_split: float
    opt_class: type
    optimizer: torch.optim.Optimizer = dataclasses.field(default=None, init=False)
    opt_params: dict = dataclasses.field(default_factory=dict)
    device: torch.device = dataclasses.field(
        default=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        init=False,
    )

    def __repr__(self):
        return (f"NetworkTrainerArguments(batch_size={self.batch_size}, loss_fn={self.loss_fn.__class__.__name__}(), "
                f"local_epochs={self.local_epochs}, valid_split={self.valid_split}, "
                f"optimizer={self.opt_class.__name__}(" +
                ", ".join(
                    [f"{param}={value}" for param, value in self.opt_params.items()]
                ) +
                f"), device='{self.device.type}')"
                )

    def init_optim(self, model: torch.nn.Module):
        self.optimizer = self.opt_class(model.parameters(), **self.opt_params)


@dataclasses.dataclass
class SelectionTrainerArguments:
    opt_class: type
    max_epochs: int = dataclasses.field(default=1, init=False)
    alpha: int = dataclasses.field(default=1)
    theta: int = dataclasses.field(default=1)
    optimizer: torch.optim.Optimizer = dataclasses.field(default=None, init=False)
    opt_params: dict = dataclasses.field(default_factory=dict)
    device: torch.device = dataclasses.field(
        default=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        init=False,
    )

    def __repr__(self):
        return (f"SelectionTrainerArguments(alpha={self.alpha}, theta={self.theta}, "
                f"max_epochs={self.max_epochs}, "
                f"optimizer={self.opt_class.__name__}(" +
                ", ".join(
                    [f"{param}={value}" for param, value in self.opt_params.items()]
                ) +
                f"), device='{self.device.type}')"
                )

    def init_optim(self, model: torch.nn.Module):
        self.optimizer = self.opt_class(model.parameters(), **self.opt_params)
