# OCD-FL: A Novel Communication-Efficient Peer Selection-based Decentralized Federated Learning

## Abstract

The conjunction of edge intelligence and the ever-growing Internet-of-Things (IoT) network heralds a new era of 
collaborative machine learning, with federated learning (FL) emerging as the most prominent paradigm. With the growing 
interest in these learning schemes, researchers started addressing some of their most fundamental limitations. Indeed, 
conventional FL with a central aggregator presents a single point of failure and a network bottleneck. To bypass this 
issue, decentralized FL where nodes collaborate in a peer-to-peer network has been proposed. Despite the latter’s 
efficiency, communication costs and data heterogeneity remain key challenges in decentralized FL. In this context, 
we propose a novel scheme, called opportunistic communication-efficient decentralized federated learning, a.k.a., 
OCD-FL, consisting of a systematic FL peer selection for collaboration, aiming to achieve maximum FL knowledge gain 
while reducing energy consumption. Experimental results demonstrate the capability of OCD-FL to achieve similar or 
better performances than the fully collaborative FL, while significantly reducing consumed energy by at least 30% and 
up to 80%.

## Repository structure

```
.
├── client/
│   ├── activation/         # Client activation policies
│   ├── aggregation/        # Model aggregation policies
│   ├── selection/          # Peer selection policies
│   ├── dataset/            # Dataset distribution policy amongst clients
│   ├── models/             # Optimization models used
│   ├── loggers/            # Loggers used for debugging and governance
│   ├── client.py           # Client class
│   └── components.py       # Client compoenents i.e. CPU and Transmitter
├── dashboard/
│   ├── assets/             # Stylesheets
│   ├── components/         # Dash components
│   ├── figures/            # Plotly figures
│   ├── pages/              # Dash pages
│   ├── utils/              # Utilities
│   └── app.py              # Script to launch server
├── [data]                  # Suppressed by .gitignore. Contains CIFAR10 and MNIST data
└── main.py                 # Script to launch simulation
```

## Repository overview

This repository contains a client module which contains all operations and policies a federated client runs, from 
activation to aggregation. At the end of a simulation, `JSONLogger` creates a `.json` file inside a `logs/` directory. 
The latter is then used by the dashboard module to generate a visualised summary.

## Execution

It is preferable to use an IDE (ideally PyCharm) as it makes life so much easier.
Create a virtual environment and install required libraries using the following command line,
````commandline
pip install -r requirements. txt
````

To start a federated learning simulation, run the main script as follows,
````commandline
python main.py --clients 10 --rounds 20 --dataset MNIST
````
You can set your own configuration by either changing the script arguments' values or manually editing the script file.

To visualize the dashboard, start the server as follows,
````commandline
python app.py
````

## Licence
TL;DR: This is an open-source research project so feel free to use it as you wish.