from client.models import LightningConvNet
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import warnings
import argparse
import logging
from client.logger import ConsoleLogger
from client import Client
from client.dataset.sampling import DataChunk
from client.activation import FullActivator, RandActivator
from client.aggregation import FedAvg
from client.selection import EfficientPeerSelector, FullPeerSelector
import json
from functools import reduce


# Disabling unnecessary warning and logs
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')
logging.getLogger('fsspec').setLevel(logging.CRITICAL)
logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# Setup loggers
logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clients', type=int, default=3, help='Set the number of clients in the network')
    parser.add_argument('--rounds', type=int, default=1, help='Set the number of federated learning rounds')
    args = parser.parse_args()

    # Load dataset
    dataset = MNIST(root='data', train=True, transform=ToTensor(), download=True)
    test_ds = MNIST(root='data', train=False, transform=ToTensor(), download=True)

    clients = []
    for _ in range(args.clients):
        activator = RandActivator(p=.7)
        aggregator = FedAvg()
        selector = FullPeerSelector()

        client = Client(
            geo_limits=((36.897092, 10.152086), (36.870453, 10.219636)),
            model=LightningConvNet(),
            train_ds=DataChunk(dataset, balanced_sampling=False),
            test_ds=test_ds,
            local_epochs=3,
            batch_size=16,
            activator=activator,
            aggregator=aggregator,
            selector=selector,
        )
        clients.append(client)

    for round_ in range(args.rounds):
        logger.info('Round [{:>2}/{:>2}] started'.format(round_ + 1, args.rounds))
        # Client activation
        for client in clients:
            client.lookup(clients)
            client.activate()
        active_clients = [client for client in clients if client.is_active]
        # Local model training and evaluation
        for client in active_clients:
            client.train()
            client.test()
        # Peer selection
        for client in active_clients:
            client.lookup(active_clients)
            client.select_peers()
        # Aggregation
        for receiver in active_clients:
            models = []
            for sender in active_clients:
                if receiver in sender.peers:
                    models.append(sender.model)
            receiver.aggregate(models)
        # Relocate clients
        for client in clients:
            client.relocate()
        logger.info('Round [{:>2}/{:>2}] ended'.format(round_ + 1, args.rounds))
        # Update config and reset
        for client in clients:
            client.update_dict()
            client.neighbors = []
            client.peers = []

    with open('dashboard/data.json', 'w') as file:
        json.dump(reduce(lambda a, b: {**a, **b}, [client.simulation_dict for client in clients]), file)


if __name__ == '__main__':
    main()
