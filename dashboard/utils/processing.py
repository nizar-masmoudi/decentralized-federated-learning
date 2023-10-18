import numpy as np


def process_data(data: dict):
    local_epochs = data['config']['local_epochs']
    for client in data['clients']:
        for key in ['train/loss', 'valid/loss', 'train/accuracy', 'valid/accuracy']:
            values = np.array(client[key])
            activity = np.array(client['activity']).astype(np.float64)
            mask = np.repeat(activity, local_epochs)
            indices = np.where(mask == 1)[0]
            mask[indices] = values
            prev = np.arange(len(mask))
            prev[mask == 0] = 0
            prev = np.maximum.accumulate(prev)
            values = mask[prev]
            values[values == 0] = values[values != 0][0]
            client[key] = values.tolist()

        for key in ['test/loss', 'test/accuracy']:
            values = np.array(client[key])
            activity = np.array(client['activity']).astype(np.float64)
            mask = activity
            indices = np.where(mask == 1)[0]
            mask[indices] = values
            prev = np.arange(len(mask))
            prev[mask == 0] = 0
            prev = np.maximum.accumulate(prev)
            values = mask[prev]
            values[values == 0] = values[values != 0][0]
            client[key] = values.tolist()

    return data
