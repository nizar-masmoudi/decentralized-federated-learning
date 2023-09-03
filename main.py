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
from client.configs import TrainerConfig, NodeConfig, TransmissionConfig, ComputationConfig
from client.aggregator import Aggregator
from client.selector import PeerSelector
from client.activator import ClientActivator

# Setup console logger
logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)

# Initialize W&B
wandb_logger = WandbLogger(
    project='decentralized-federated-learning',
    name='vanilla',
    config={
        # 'Model': MODEL.__name__,
        # 'Optimizer': {
        #     'class': OPTIMIZER.__name__,
        #     **OPT_PARAMS,
        # },
        # 'Batch Size': BATCH_SIZE,
        # 'Criterion': LOSS.__name__,
        # 'Epochs per Round': EPOCHS,
        # 'Client Activation': {
        #     'policy': ACTIVATION_POLICY.name,
        # },
        # 'Aggregation': {
        #     'policy': AGGREGATION_POLICY.name,
        # },
        # 'Peer Selection': {
        #     'policy': SELECTION_POLICY.name,
        #     'lookup distance': LOOKUP_DISTANCE,
        # },
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
        # Configuration
        metadata = NodeConfig(geo_limits=((36.897092, 10.152086), (36.870453, 10.219636)))
        dataset = DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), size=1024, equal=True)
        testset = MNIST(root='data', train=False, transform=ToTensor(), download=True)
        model = ConvNet()
        trainer_cfg = TrainerConfig(
            opt_class=torch.optim.SGD,
            opt_params=dict(lr=.01, momentum=.9),
            batch_size=64,
            loss_fn=torch.nn.CrossEntropyLoss(),
            local_epochs=3,
            validation_split=.1,
        )
        comp_cfg = ComputationConfig(cpu_cycles=2, computation_capacity=2)
        trans_cfg = TransmissionConfig(transmission_power=5, bandwidth=50e10)
        aggregation_policy = Aggregator.Policy.FEDAVG
        selection_policy = PeerSelector.Policy.FULL
        activation_policy = ClientActivator.Policy.FULL

        # Initializing clients
        clients.append(
            Client(
                metadata=metadata,
                dataset=dataset,
                testset=testset,
                model=model,
                trainer_cfg=trainer_cfg,
                aggregation_policy=aggregation_policy,
                selection_policy=selection_policy,
                activation_policy=activation_policy,
                comp_cfg=comp_cfg,
                trans_cfg=trans_cfg,
                wandb_logger=wandb_logger
            )
        )

    # client = clients[0]
    # train_dl = client.trainer.train_dl
    # imgs, targets = next(iter(train_dl))
    # print(torch.unique(targets, return_counts=True))
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
