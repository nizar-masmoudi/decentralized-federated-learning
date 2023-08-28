import wandb
from dotenv import load_dotenv

load_dotenv()


class WandbLogger:
    def __init__(self, project: str, name: str, group: str = None, config: dict = None):
        self.project = project
        self.name = name
        self.group = group
        self.config = config
        self.run = wandb.init(project=self.project, name=self.name, group=self.group, config=self.config)

    def log_metrics(self, metric_dict: dict, epoch: int, client_id: int):
        self.run.log({**metric_dict, 'Epoch': epoch, 'Client': client_id})
