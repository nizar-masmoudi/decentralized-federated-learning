# decentralized-federated-learning

## Project structure
```
.
├── client/
│   ├── models/               # Contains different model architectures for testing 
│   ├── trainer/
│   │   └── networktrainer.py        # Contains local training script 
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
├── [.env]
└── .env.template
```
## Resources
- [A Safe Deep Reinforcement Learning Approach for Energy Efficient Federated Learning in Wireless Communication Networks](https://arxiv.org/pdf/2308.10664.pdf)
- [Federated Learning for Energy-balanced Client Selection in Mobile Edge Computing](file:///C:/Users/nizar/Downloads/Federated_Learning_for_Energy-balanced_Client_Selection_in_Mobile_Edge_Computing.pdf)

## TODO
- [ ] Implement energy consumption calculator
- [ ] Implement mixing aggregation policy
- [ ] Implement efficient client activation policy
- [ ] Implement efficient peer selection policy
- [x] Explore IID and Non-IID DataChunks
- [ ] Look into adding a config.yml file

