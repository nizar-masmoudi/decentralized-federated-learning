import argparse
import yaml
from client import Client
import logging
  
logging.basicConfig(level = logging.DEBUG, format = '%(levelname)-5s | %(name)-25s | %(funcName)-10s | %(message)s')

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