import json
import os
import os.path as osp
import uuid
from abc import ABC

import numpy as np
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

import client as cl


class JSONLogger(Logger, ABC):
    """
    JSON logger used to log client metrics and information during rounds of federated learning.
    """
    def __init__(
            self,
            project: str,
            name: str = None,
            config: dict = None,
            version: str = None,
            save_dir: str = None
    ) -> None:
        super().__init__()
        self._id = str(uuid.uuid4())
        self._project = project
        self._name = name or self._id
        self._version = version or '1.0.0'
        self._save_dir = save_dir or 'logs/'
        self._config = config or {}
        self._json = {
            'project': self._project,
            'name': self._name,
            'version': self.version,
            'config': self._config,
            'clients': []
        }
        # Bug in new version
        self.experiment = None

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def config(self):
        return self._config

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        pass

    @rank_zero_only
    def add_client(self, client: 'cl.Client'):
        """
        Add client metadata.
        :param client: client to add.
        """
        targets = np.array([target for _, target in client.datachunk])

        self._json['clients'].append({
            'id': client.id_,
            'components': {
                'cpu': client.cpu.to_dict(),
                'transmitter': client.transmitter.to_dict(),
            },
            'dataset': {
                'name': client.datachunk.__class__.__name__,
                'distribution': [np.count_nonzero(targets == t) for t in range(10)]
            },
        })

    @rank_zero_only
    def log_location(self, client: 'cl.Client'):
        """
        Log client location.
        :param client: client to log.
        """
        # Lookup client
        obj = next((item for item in self._json['clients'] if item['id'] == client.id_), None)
        assert obj is not None

        if 'locations' not in obj.keys():
            obj['locations'] = [client.location]
        else:
            obj['locations'].append(client.location)

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log client metrics.
        :param metrics: metrics to log.
        :param step: log step.
        """
        id_ = int(list(metrics.keys())[0].split('/')[0])
        metrics = {k[k.find('/') + 1:]: v for k, v in metrics.items()}
        # Lookup client
        client = next((item for item in self._json['clients'] if item['id'] == id_), None)
        assert client is not None

        for metric, value in metrics.items():
            if metric not in client.keys():
                client[metric] = [value]
            else:
                client[metric].append(value)

    @rank_zero_only
    def log_activity(self, client: 'cl.Client'):
        """
        Log client activity.
        :param client: client to log.
        """
        # Lookup client
        obj = next((item for item in self._json['clients'] if item['id'] == client.id_), None)
        assert obj is not None

        if 'activity' not in obj.keys():
            obj['activity'] = [client.is_active]
        else:
            obj['activity'].append(client.is_active)

    @rank_zero_only
    def log_neighbors(self, client: 'cl.Client'):
        """
        Log client neighbors.
        :param client: client to log.
        """
        # Lookup client
        obj = next((item for item in self._json['clients'] if item['id'] == client.id_), None)
        assert obj is not None

        if 'neighbors' not in obj.keys():
            obj['neighbors'] = [[{
                'id': neighbor.id_,
                'energy': client.communication_energy(neighbor),
                'distance': cl.Client.distance(client, neighbor),
            } for neighbor in client.neighbors]]
        else:
            obj['neighbors'].append([{
                'id': neighbor.id_,
                'energy': client.communication_energy(neighbor),
                'distance': cl.Client.distance(client, neighbor),
            } for neighbor in client.neighbors])

    @rank_zero_only
    def log_peers(self, client: 'cl.Client'):
        """
        Log client peers.
        :param client: client to log.
        """
        # Lookup client
        obj = next((item for item in self._json['clients'] if item['id'] == client.id_), None)
        assert obj is not None

        if 'peers' not in obj.keys():
            obj['peers'] = [[{
                'id': peer.id_,
                'energy': client.communication_energy(peer),
                'distance': cl.Client.distance(client, peer),
            } for peer in client.peers]]
        else:
            obj['peers'].append([{
                'id': peer.id_,
                'energy': client.communication_energy(peer),
                'distance': cl.Client.distance(client, peer),
            } for peer in client.peers])

    @rank_zero_only
    def save(self):
        """
        Save JSON file.
        """
        if not osp.exists(self._save_dir):
            os.makedirs(self._save_dir)
        with open(osp.join(self._save_dir, f'{self._name}.json'), 'w') as file:
            json.dump(self._json, file)

    @rank_zero_only
    def finalize(self, status):
        pass
