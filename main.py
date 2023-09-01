import argparse
from client import Client
import logging
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from client.models import ConvNet
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from client.loggers import ConsoleLogger, WandbLogger
from client.dataset.sampling import DataChunk
import torch

# Setup console logger
logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)

# Configuration
IID = True
GEO_LIMITS = ((36.897092, 10.152086), (36.870453, 10.219636))
LOOKUP_DISTANCE = 99999
MODEL = ConvNet
OPTIMIZER = SGD
OPT_PARAMS = dict(lr=.01, momentum=.9)
BATCH_SIZE = 1024
LOSS = CrossEntropyLoss
EPOCHS = 3
AGGREGATION_POLICY = Client.AggregationPolicy.FEDAVG
SELECTION_POLICY = Client.SelectionPolicy.FULL
ACTIVATION_POLICY = Client.ActivationPolicy.FULL

# Initialize W&B
wandb_logger = WandbLogger(
    project='decentralized-federated-learning',
    name='vanilla',
    config={
        'Model': MODEL.__name__,
        'Optimizer': {
            'class': OPTIMIZER.__name__,
            **OPT_PARAMS,
        },
        'Batch Size': BATCH_SIZE,
        'Criterion': LOSS.__name__,
        'Epochs per Round': EPOCHS,
        'Client Activation': {
            'policy': ACTIVATION_POLICY.name,
        },
        'Aggregation': {
            'policy': AGGREGATION_POLICY.name,
        },
        'Peer Selection': {
            'policy': SELECTION_POLICY.name,
            'lookup distance': LOOKUP_DISTANCE,
        },
    }
)


def main():
    # TODO: Write description
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--clients', type=int, default=1, help='Number of clients.')
    parser.add_argument('-r', '--rounds', type=int, default=10, help='Number of rounds.')
    args = parser.parse_args()

    # Initialize clients
    clients = []
    for _ in range(args.clients):
        train_ds = DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), size=1024, equal=IID)
        test_ds = MNIST(root='data', train=False, transform=ToTensor(), download=True)
        model = MODEL()
        optimizer = OPTIMIZER(model.parameters(), **OPT_PARAMS)
        batch_size = BATCH_SIZE
        loss_fn = LOSS()
        n_epochs = EPOCHS
        aggregation_policy = AGGREGATION_POLICY
        selection_policy = SELECTION_POLICY
        activation_policy = ACTIVATION_POLICY

        clients.append(
            Client(
                geo_limits=GEO_LIMITS,
                train_ds=train_ds,
                test_ds=test_ds,
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                loss_fn=loss_fn,
                n_epochs=n_epochs,
                aggregation_policy=aggregation_policy,
                selection_policy=selection_policy,
                activation_policy=activation_policy,
                wandb_logger=wandb_logger
            )
        )

    client = clients[0]
    train_dl = client.trainer.train_dl
    imgs, targets = next(iter(train_dl))
    print(torch.unique(targets, return_counts=True))
    # imgs, targets = next(iter(train_dl))
    # print(imgs.shape, targets.shape)

    # for ridx in range(args.rounds):
    #     logger.info(f'Round [{ridx+1}/{args.rounds}] started')
    #     # Simulate UAV movement
    #     for client in clients:
    #         client.relocate()
    #     # Client activation
    #     for client in clients:
    #         client.activate()
    #     # Local training
    #     for client in clients:
    #         if client.is_active:
    #             client.train(ridx)
    #     # Peer selection
    #     for client in clients:
    #         client.lookup(clients, max_dist=LOOKUP_DISTANCE)
    #         client.select_peers()
    #     # Aggregation
    #     for client in clients:
    #         client.aggregate([peer.model.state_dict() for peer in client.peers])

    wandb_logger.finish()


if __name__ == '__main__':
    main()
