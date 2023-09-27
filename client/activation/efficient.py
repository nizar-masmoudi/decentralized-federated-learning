from client.activation.activator import Activator
import client as cl


class EfficientActivator(Activator):
    def __init__(self, alpha: float, threshold: float):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def activate(self, client: 'cl.Client') -> bool:
        # Computation energy consumption
        energy = client.computation_energy()
        energies = [neighbor.computation_energy() for neighbor in client.neighbors] + [energy]
        if max(energies) == min(energies):
            scaled_energy = 0.
        else:
            scaled_energy = (energy - min(energies)) / (max(energies) - min(energies))
        # Learning slope
        slope = client.compute_lslope()
        slopes = [neighbor.compute_lslope() for neighbor in client.neighbors] + [slope]
        if max(slopes) == min(slopes):
            scaled_slope = 1.
        else:
            scaled_slope = (energy - min(slopes)) / (max(slopes) - min(slopes))
        # Activation cost
        cost = self.alpha * scaled_energy + (1 - self.alpha) * (1 - scaled_slope)
        return self.threshold > cost
