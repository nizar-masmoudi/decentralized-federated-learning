import argparse
from client import Client
import logging
from client.dataset.utils import DataChunk
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from client.models import ConvNet
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from client.loggers import ConsoleLogger, WandbLogger

# Setup console logger
logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)

# Configuration
MODEL = ConvNet
OPTIMIZER = SGD
OPT_PARAMS = dict(lr=.01, momentum=.9)
BATCH_SIZE = 32
LOSS = CrossEntropyLoss
EPOCHS = 3
AGGREGATION_POLICY = Client.AggregationPolicy.FEDAVG
SELECTION_POLICY = Client.SelectionPolicy.FULL
ACTIVATION_POLICY = Client.ActivationPolicy.FULL

# Initialize W&B
wandb_logger = WandbLogger(
    project='decentralized-federated-learning',
    name='test',
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
        },
    }
)


def main():
    # TODO: Write description
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--clients', type=int, default=4, help='Number of clients.')
    args = parser.parse_args()

    # Initialize clients
    clients = []
    for _ in range(args.clients):
        train_ds = DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), 10000)
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

    clients[0].location = (36.89891408403747, 10.171681937598443)
    clients[1].location = (36.88078621070918, 10.212364771421965)
    clients[2].location = (36.837255863182236, 10.198052311203753)
    clients[3].location = (36.84182578721515, 10.311322519219702)

    # Report client information

    for client in clients:
        client.train()

    # dists = [[.0 for _ in range(len(clients))] for _ in range(len(clients))]
    # for ci, cj in itertools.permutations(clients, 2):
    #   dists[clients.index(ci)][clients.index(cj)] = Client.distance(ci, cj)
    # print(np.array(dists))

    # for client in clients:
    #     client.lookup(clients, max_dist=10)
    #
    # for r in range(1, 5):
    #     logger.info(f'Round {r} started')
    #     for client in clients:
    #         client.activate()
    #
    #     for client in clients:
    #         if client.is_active:
    #             client.train()
    #
    #     for client in clients:
    #         client.select_peers()
    #
    #     for client in clients:
    #         client.aggregate([peer.model.state_dict() for peer in client.peers])


if __name__ == '__main__':
    main()
