import numpy as np


def process_data(data: dict):
    local_epochs = int(len(data['1']['tloss']) / sum(data['1']['activity']))
    for id_ in data:
        for key in ['tloss', 'vloss', 'tacc', 'vacc']:
            values = np.array(data[id_][key])
            activity = np.array(data[id_]['activity']).astype(np.float64)
            mask = np.repeat(activity, local_epochs)
            indices = np.where(mask == 1)[0]
            mask[indices] = values
            prev = np.arange(len(mask))
            prev[mask == 0] = 0
            prev = np.maximum.accumulate(prev)
            values = mask[prev]
            values[values == 0] = values[values != 0][0]
            data[id_][key] = values.tolist()

        for key in ['sloss', 'sacc']:
            values = np.array(data[id_][key])
            activity = np.array(data[id_]['activity']).astype(np.float64)
            mask = activity
            indices = np.where(mask == 1)[0]
            mask[indices] = values
            prev = np.arange(len(mask))
            prev[mask == 0] = 0
            prev = np.maximum.accumulate(prev)
            values = mask[prev]
            values[values == 0] = values[values != 0][0]
            data[id_][key] = values.tolist()

    return data
