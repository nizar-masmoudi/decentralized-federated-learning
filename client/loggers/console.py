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
        # Add console handler
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(Formatter())
        streamhandler.addFilter(Filter())
        # Add file handler
        filehandler = logging.FileHandler('logs/access_log.log')
        filehandler.setFormatter(Formatter())
        filehandler.addFilter(Filter())

        self.addHandler(streamhandler)
        self.addHandler(filehandler)
        self.setLevel(logging.DEBUG)
