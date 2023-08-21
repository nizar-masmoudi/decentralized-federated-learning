# decentralized-federated-learning

## Project structure
```
.
├── client/
│   ├── models/               # Contains different model architectures for testing 
│   ├── trainer/
│   │   └── trainer.py        # Contains local training script 
│   ├── aggregator/
│   │   └── aggregator.py     # Contains model aggregation policies
│   ├── selector/
│   │   └── selector.py       # Contains peer selection policies
│   ├── activator/
│   │   └── activator.py      # Contains client activation policies
│   └── client.py             # Contains Client class
├── [data/]
├── config.yml                # Contains simulation configration
├── main.py                   # Main script to run simulation
└── [.env]
└── .env.template
```

