import torch

from client.aggregation import FedAvg


class TestFedAvg:
    state_dicts = [
        {'layer1': torch.tensor([2., 7., 9., 8., 6.]), 'layer2': torch.tensor([1., 4., 8., 9., 4.])},
        {'layer1': torch.tensor([7., 1., 7., 9., 3.]), 'layer2': torch.tensor([3., 9., 9., 3., 8.])},
        {'layer1': torch.tensor([7., 3., 1., 2., 4.]), 'layer2': torch.tensor([2., 9., 9., 5., 2.])},
        {'layer1': torch.tensor([4., 4., 7., 6., 7.]), 'layer2': torch.tensor([8., 9., 4., 6., 6.])},
    ]
    agg_state = {'layer1': torch.tensor([5., 3.75, 6., 6.25, 5.]), 'layer2': torch.tensor([3.5, 7.75, 7.5, 5.75, 5.])}

    def test_aggregate(self):
        aggregator = FedAvg()
        result = aggregator.aggregate(self.state_dicts)

        for key in self.agg_state:
            assert torch.equal(result[key], self.agg_state[key])
