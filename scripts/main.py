import argparse
import logging
from client import Client
from client.models import ConvNet
from client.loggers import SystemLogger, WandbLogger
from client.dataset.sampling import DataChunk
from client.configuration import ClientConfig, Transmitter, CPU
from client.training import NetworkTrainer
from client.training.arguments import NetworkTrainerArguments
from client.aggregation import Aggregator
from client.selection import EfficientPeerSelector
from client.activation import RandomActivator, EfficientActivator
import torch
from torchvision.datasets import MNIST
import math
from torchvision.transforms import ToTensor
import os
import os.path as osp

# Setup console logger
logging.setLoggerClass(SystemLogger)
logger = logging.getLogger(__name__)

# Initialize W&B
wandb_logger = WandbLogger(
    project='decentralized-federated-learning',
    name='vanilla',
)


def main():
    # TODO: Write description
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--clients', type=int, default=5, help='Number of clients.')
    parser.add_argument('-r', '--rounds', type=int, default=10, help='Number of rounds.')
    args = parser.parse_args()

    # Initialize clients
    clients = []
    for id_ in range(1, args.clients+1):
        # Initialize modules
        trainer = NetworkTrainer(
            args=NetworkTrainerArguments(
                batch_size=32,
                loss_fn=torch.nn.CrossEntropyLoss(),
                local_epochs=3,
                valid_split=.1,
                opt_class=torch.optim.SGD,
                opt_params=dict(lr=.01, momentum=.9)
            ),
            wandb_logger=wandb_logger
        )
        aggregator = Aggregator(policy=Aggregator.AggregationPolicy.FEDAVG)
        selector = EfficientPeerSelector()
        activator = EfficientActivator(threshold=1, alpha=.5)
        # Initialize model & datasets
        model = ConvNet()
        datachunk = DataChunk(
            MNIST(root=osp.join(os.environ['BASE_PATH'], 'data'), train=True, transform=ToTensor(), download=True),
            size=1024,
            eq_dist=True
        )
        testset = MNIST(root=osp.join(os.environ['BASE_PATH'], 'data'), train=False, transform=ToTensor(), download=True)
        # Initialize client configuration
        config = ClientConfig(
            geo_limits=((36.897092, 10.152086), (36.870453, 10.219636)),
            transmitter=Transmitter(),
            cpu=CPU()
        )
        # Initialize client
        client = Client(model, datachunk, testset, trainer, aggregator, activator, selector, config, wandb_logger)
        clients.append(client)

    for client in clients:
        client.lookup(clients, math.inf)
        client.activate()

    wandb_logger.finish()


if __name__ == '__main__':
    main()
