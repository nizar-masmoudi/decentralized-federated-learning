import logging


class Filter(logging.Filter):
    def filter(self, record):
        # Merge name and funcName
        setattr(record, 'source', f'{record.name}.{record.funcName}')
        # Add client id
        if hasattr(record, 'client'):
            setattr(record, 'client', f'Client {record.client}')
        else:
            setattr(record, 'client', 'Global')
        return True


class Formatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt='%(levelname)-5s | %(source)-40s | %(client)-10s | %(message)s')


class ConsoleLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        # Add console handler with custom formatter and custom filter
        handler = logging.StreamHandler()
        handler.setFormatter(Formatter())
        handler.addFilter(Filter())
        self.addHandler(handler)
        self.setLevel(logging.DEBUG)
