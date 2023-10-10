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