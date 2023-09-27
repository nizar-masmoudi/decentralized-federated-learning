from torch.utils.data import Dataset, random_split


def train_valid_split(dataset: Dataset, valid_split: float = .1):
    """
    Split dataset into train and vlidation splits.
    :param dataset: Dataset to split.
    :param valid_split: Validation split
    :return: Train and validation splits
    """
    valid_split = int((1 - valid_split) * len(dataset))
    train_split = len(dataset) - valid_split
    return random_split(dataset, [train_split, valid_split])
