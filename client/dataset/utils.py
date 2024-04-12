from torch.utils.data import Dataset, random_split


def train_valid_split(dataset: Dataset, validation_split: float = .1):
    """
    Split dataset into train and validation splits.
    :param dataset: Dataset to split.
    :param validation_split: Validation split
    :return: Train and validation splits
    """
    validation_size = int(validation_split * len(dataset))
    train_size = len(dataset) - validation_size
    return random_split(dataset, [train_size, validation_size])
