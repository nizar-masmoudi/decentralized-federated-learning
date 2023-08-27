import argparse
import yaml
from client import Client
import logging
import itertools

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
  parser.add_argument('-n', '--nodes', type = int, default = 3, help = 'Number of nodes.')
  args = parser.parse_args()
  
  
  config = yaml.safe_load(args.config)
      
  client1 = Client(config)
  client2 = Client(config)
  client3 = Client(config)
  client4 = Client(config)
  
  client1.location = (36.89891408403747, 10.171681937598443)
  client2.location = (36.88078621070918, 10.212364771421965)
  client3.location = (36.837255863182236, 10.198052311203753)
  client4.location = (36.84182578721515, 10.311322519219702)
  
  clients = [client1, client2, client3, client4]
  
  for ci, cj in itertools.combinations(clients, 2):
    print(f'Distance between {ci} and {cj} = {Client.distance(ci, cj)}')
  
  print(client4.lookup([client1, client2, client3, client4], max_dist = 5))
  

  
if __name__ == '__main__':
  main()