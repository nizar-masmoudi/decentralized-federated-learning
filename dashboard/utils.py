import math


def format_size(value: float, unit: str = 'bit'):
    units = {
        'bit': ['bits', 'KB', 'MB', 'GB'],
        'hertz': ['Hz', 'KHz', 'MHz', 'GHz']
    }

    unit_size = 1e3
    unit_index = 0
    size = value
    while size >= unit_size and unit_index < len(units[unit]) - 1:
        size /= unit_size
        unit_index += 1
    return '{} {}'.format(int(size), units[unit][unit_index])


def geo_distance(location1: tuple, location2: tuple) -> float:
    # Radius of the Earth in kilometers
    radius = 6371.0
    # Get locations
    lat1, lon1 = location1
    lat2, lon2 = location2
    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c
    return distance
