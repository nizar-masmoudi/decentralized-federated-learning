import logging
import sys


class Filter(logging.Filter):
    def filter(self, record):
        # Merge name and funcName
        setattr(record, 'source', f'{record.name}.{record.funcName}')
        # Add client id
        if hasattr(record, 'id'):
            setattr(record, 'id', record.id)
        else:
            setattr(record, 'id', 'GLB')
        return True


class Formatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt='[%(id)3s] [%(levelname)5s] [%(source)-40s] %(message)s')


class ConsoleLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        # Add console handler
        streamhandler = logging.StreamHandler(stream=sys.stdout)
        streamhandler.setFormatter(Formatter())
        streamhandler.addFilter(Filter())

        self.addHandler(streamhandler)
        self.setLevel(logging.INFO)

        self.propagate = False
