import argparse
from client import Client
import logging
from client.dataset.utils import DataChunk
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from client.models import ConvNet
from functools import partial
import torch


class Filter(logging.Filter):
    def filter(self, record):
        record.source = f'{record.name}.{record.funcName}'
        if hasattr(record, 'client'):
            setattr(record, 'client', f'Client {record.client}')
        else:
            setattr(record, 'client', 'Global')
        return True


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)-5s | %(source)-40s | %(client)-10s | %(message)s'))
handler.addFilter(Filter())

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

logger = logging.getLogger(__name__)


def main():
    # TODO: Write description
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--nodes', type=int, default=3, help='Number of nodes.')
    args = parser.parse_args()

    client1 = Client(
        train_ds=DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), 10000),
        test_ds=MNIST(root='data', train=False, transform=ToTensor(), download=True),
        model=ConvNet(),
        optimizer=partial(torch.optim.SGD, lr=.01, momentum=0.9),
        batch_size=32,
        loss_fn=torch.nn.CrossEntropyLoss(),
        n_epochs=3,
        aggregation_policy=Client.AggregationPolicy.FEDAVG,
        selection_policy=Client.SelectionPolicy.FULL,
        activation_policy=Client.ActivationPolicy.FULL
    )

    client2 = Client(
        train_ds=DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), 10000),
        test_ds=MNIST(root='data', train=False, transform=ToTensor(), download=True),
        model=ConvNet(),
        optimizer=partial(torch.optim.SGD, lr=.01, momentum=0.9),
        batch_size=32,
        loss_fn=torch.nn.CrossEntropyLoss(),
        n_epochs=3,
        aggregation_policy=Client.AggregationPolicy.FEDAVG,
        selection_policy=Client.SelectionPolicy.FULL,
        activation_policy=Client.ActivationPolicy.FULL
    )

    client3 = Client(
        train_ds=DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), 10000),
        test_ds=MNIST(root='data', train=False, transform=ToTensor(), download=True),
        model=ConvNet(),
        optimizer=partial(torch.optim.SGD, lr=.01, momentum=0.9),
        batch_size=32,
        loss_fn=torch.nn.CrossEntropyLoss(),
        n_epochs=3,
        aggregation_policy=Client.AggregationPolicy.FEDAVG,
        selection_policy=Client.SelectionPolicy.FULL,
        activation_policy=Client.ActivationPolicy.FULL
    )

    client4 = Client(
        train_ds=DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), 10000),
        test_ds=MNIST(root='data', train=False, transform=ToTensor(), download=True),
        model=ConvNet(),
        optimizer=partial(torch.optim.SGD, lr=.01, momentum=0.9),
        batch_size=32,
        loss_fn=torch.nn.CrossEntropyLoss(),
        n_epochs=3,
        aggregation_policy=Client.AggregationPolicy.FEDAVG,
        selection_policy=Client.SelectionPolicy.FULL,
        activation_policy=Client.ActivationPolicy.FULL
    )

    client1.location = (36.89891408403747, 10.171681937598443)
    client2.location = (36.88078621070918, 10.212364771421965)
    client3.location = (36.837255863182236, 10.198052311203753)
    client4.location = (36.84182578721515, 10.311322519219702)

    clients = [client1, client2, client3, client4]

    # dists = [[.0 for _ in range(len(clients))] for _ in range(len(clients))]
    # for ci, cj in itertools.permutations(clients, 2):
    #   dists[clients.index(ci)][clients.index(cj)] = Client.distance(ci, cj)
    # print(np.array(dists))

    for client in clients:
        client.lookup(clients, max_dist=10)

    for r in range(1, 5):
        logger.info(f'Round {r} started')
        for client in clients:
            client.activate()

        for client in clients:
            if client.is_active:
                client.train()

        for client in clients:
            client.select_peers()

        for client in clients:
            client.aggregate([peer.model.state_dict() for peer in client.peers])


if __name__ == '__main__':
    main()
