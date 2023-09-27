from client.activation.activator import Activator
import client as cl


class ConvergenceBasedActivator(Activator):
    def __init__(self, tolerance: int, min_delta: float):
        super().__init__()
        self.tolerance = tolerance
        self.min_delta = min_delta

    def activate(self, client: 'cl.Client') -> bool:
        raise NotImplementedError
