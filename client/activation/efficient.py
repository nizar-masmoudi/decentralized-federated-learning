from client.activation.activator import Activator
import numpy as np
import client as cl
from client.dataset.sampling import DataChunk


class EfficientActivator(Activator):
    def __init__(self, alpha: float, threshold: float):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def activate(self, client: 'cl.Client', *args) -> bool:
        # Computation energy consumption
        energy = client.computation_energy()
        max_energy = (client.local_epochs * client.cpu.kappa * client.model.flops * DataChunk.MAX_SIZE *
                      (client.cpu.MAX_FREQ ** 2) / client.cpu.fpc)
        min_energy = 0
        sc_energy = (energy - min_energy) / (max_energy - min_energy)

        # Learning slope
        slope = client.compute_lslope()
        sc_slope = min(slope, 1)

        # Activation cost
        cost = self.alpha * sc_energy + (1 - self.alpha) * (1 - sc_slope)

        return bool(self.threshold > cost)
