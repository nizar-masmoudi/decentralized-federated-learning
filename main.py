import argparse
from client import Client
import logging
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from client.models import ConvNet
from client.loggers import ConsoleLogger, WandbLogger
from client.dataset.sampling import DataChunk
import torch
from client.configs import TrainerConfig, Metadata, TransmissionConfig, ComputationConfig
from client.aggregation import Aggregator
from client.selection import PeerSelector
from client.activation import ClientActivator

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
    parser.add_argument('-c', '--clients', type=int, default=3, help='Number of clients.')
    parser.add_argument('-r', '--rounds', type=int, default=10, help='Number of rounds.')
    args = parser.parse_args()

    # Initialize clients
    clients = []
    for id_ in range(1, args.clients + 1):
        # Configuration
        metadata = Metadata(geo_limits=((36.897092, 10.152086), (36.870453, 10.219636)))
        logger.debug(metadata, extra={'client': id_})
        dataset = DataChunk(MNIST(root='data', train=True, transform=ToTensor(), download=True), size=1024, iid=True)
        logger.debug(repr(dataset), extra={'client': id_})
        testset = MNIST(root='data', train=False, transform=ToTensor(), download=True)
        model = ConvNet()
        logger.debug(repr(model), extra={'client': id_})
        trainer_cfg = TrainerConfig(
            opt_class=torch.optim.SGD,
            opt_params=dict(lr=.01, momentum=.9),
            batch_size=64,
            loss_fn=torch.nn.CrossEntropyLoss(),
            local_epochs=3,
            validation_split=.1,
        )
        logger.debug(trainer_cfg, extra={'client': id_})
        comp_cfg = ComputationConfig(cpu_cycles=2, cpu_frequency=2)
        logger.debug(comp_cfg, extra={'client': id_})
        trans_cfg = TransmissionConfig(transmission_power=5, bandwidth=50e10)
        logger.debug(trans_cfg, extra={'client': id_})
        aggregation_policy = Aggregator.Policy.FEDAVG
        logger.debug(repr(aggregation_policy), extra={'client': id_})
        selection_policy = PeerSelector.Policy.FULL
        logger.debug(repr(selection_policy), extra={'client': id_})
        activation_policy = ClientActivator.Policy.FULL
        logger.debug(repr(activation_policy), extra={'client': id_})

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

    for client in clients:
        client.computation_energy()
        for other in clients:
            if other != client:
                client.communication_energy(other, 1)
        # client.lookup(clients, max_dist=3)
        # active = EfficientActivation.activate(client, client.metadata.neighbors, .5)

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
