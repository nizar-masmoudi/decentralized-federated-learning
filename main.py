import argparse
import copy
import importlib
import logging
import warnings

from torchvision.transforms import ToTensor

from client import Client
from client.activation import FullActivator
from client.aggregation import FedAvg
from client.dataset.sampling import DataChunk
from client.loggers import ConsoleLogger, JSONLogger
from client.selection import NonePeerSelector

# Disabling unnecessary warnings and logs
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')
logging.getLogger('fsspec').setLevel(logging.CRITICAL)
logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# Setup loggers
logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)

json_logger = JSONLogger('decentralized-federated-learning', 'no-communication')


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clients', type=int, help='Set the number of clients in the network')
    parser.add_argument('--rounds', type=int, help='Set the number of federated learning rounds')
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10'], help='Dataset to use for training')
    args = parser.parse_args()

    # Load Dataset
    Dataset = getattr(importlib.import_module('torchvision.datasets'), args.dataset)
    dataset = Dataset(root='data', train=True, transform=ToTensor(), download=True)
    test_ds = DataChunk(Dataset(root='data', train=False, transform=ToTensor(), download=True), 1000)
    # Load LightningModel
    Model = getattr(importlib.import_module('client.models'), f'Lightning{args.dataset}')
    model = Model()

    clients = []
    for _ in range(args.clients):
        activator = FullActivator()
        selector = NonePeerSelector()
        aggregator = FedAvg()

        if json_logger.config == {}:
            json_logger.config['activator'] = {
                'policy': activator.__class__.__name__,
                **{k: v for k, v in vars(activator).items() if k != 'id_'}
            }
            json_logger.config['selector'] = {
                'policy': selector.__class__.__name__,
                **{k: v for k, v in vars(selector).items() if k != 'id_'}
            }
            json_logger.config['aggregator'] = {
                'policy': aggregator.__class__.__name__,
                **{k: v for k, v in vars(aggregator).items() if k != 'id_'}
            }
            json_logger.config['local_epochs'] = 3
            json_logger.config['geo_limits'] = ((36.897092, 10.152086), (36.870453, 10.219636))

        local_model = copy.deepcopy(model)
        local_model.id_ = _ + 1

        client = Client(
            geo_limits=((36.897092, 10.152086), (36.870453, 10.219636)),
            model=local_model,
            train_ds=DataChunk(
                dataset,
                size=len(dataset)//args.clients,
                iid=True
            ),
            test_ds=test_ds,
            local_epochs=3,
            batch_size=64,
            activator=activator,
            aggregator=aggregator,
            selector=selector,
            json_logger=json_logger
        )
        clients.append(client)

    for round_ in range(args.rounds):
        logger.info('Round [{:>2}/{:>2}] started'.format(round_ + 1, args.rounds))
        # Relocate clients
        for client in clients:
            client.relocate()
        # Lookup neighborhood
        for client in clients:
            client.lookup(clients)
        # Client activation
        for client in clients:
            client.activate()
        active_clients = [client for client in clients if client.is_active]
        # Local model training and evaluation
        for client in active_clients:
            client.train()
            client.test()
        # Peer selection
        for client in active_clients:
            client.select_peers()
        # Aggregation
        for receiver in active_clients:
            models = []
            for sender in active_clients:
                if receiver in sender.peers:
                    models.append(sender.model)
            receiver.aggregate(models)

        # Reset
        for client in clients:
            client.neighbors = []
            client.peers = []
        logger.info('Round [{:>2}/{:>2}] ended'.format(round_ + 1, args.rounds))

    json_logger.save()


if __name__ == '__main__':
    main()
