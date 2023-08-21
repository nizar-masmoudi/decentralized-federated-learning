import argparse
import yaml
from client import Client
import logging

class Filter(logging.Filter):
  def filter(self, record):
    record.source = f'{record.name}.{record.funcName}'
    if not hasattr(record, 'client'):
      setattr(record, 'client', -1)
    return True
  
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)-5s | Client %(client)-2s | %(source)-35s | %(message)s'))
handler.addFilter(Filter())

logging.basicConfig(level = logging.DEBUG, handlers = [handler])

def main():
  # TODO: Write description
  parser = argparse.ArgumentParser(description = '', formatter_class = argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-c', '--config', type = argparse.FileType('r'), default = 'config.yml', help = 'YAML Config file.')
  args = parser.parse_args()
  
  
  config = yaml.safe_load(args.config)
  client = Client(config)
  client.train()

  
if __name__ == '__main__':
  main()