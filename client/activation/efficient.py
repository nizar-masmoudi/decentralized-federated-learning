from statistics import mean
import logging
from client.loggers import ConsoleLogger

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger(__name__)


class EfficientActivation:
    @staticmethod
    def activate(client: any, neighbors: list, threshold: float, alpha: float = .5) -> bool:
        energy = client.computation_energy() / mean([neighbor.computation_energy() for neighbor in neighbors] + [client.computation_energy()])
        slope = client.metadata.learning_slope / mean([neighbor.metadata.learning_slope for neighbor in neighbors] + [client.metadata.learning_slope])
        cost = alpha*energy - (1 - alpha)*slope
        logger.debug(f'EfficientActivation.activate(...)={cost < threshold}', extra={'client': client.id_})
        return cost < threshold
