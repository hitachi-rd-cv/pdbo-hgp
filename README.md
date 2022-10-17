# Personalized Decentralized Bilevel Optimization over Stochastic and Directed Networks

This is the official implementation of the experiments in the following paper:

> Naoyuki Terashita and Satoshi Hara  
> [Personalized Decentralized Bilevel Optimization over Stochastic and Directed Networks](https://arxiv.org/abs/2210.02129)  
> *arXiv:2210.02129 (under review)*, 2022

## Environments

Our experimental results were made in a NVIDIA Docker (`2.11.0` with Docker engine of `19.03.12`) container built
from [`docker/Dockerfile`](./docker/Dockerfile).
The container can be obtained by the following steps:

~~~
$ git clone https://github.com/hitachi-rd-cv/pdbo-hgp.git
$ docker build ./pdbo-hgp/docker/ --tag pdbo-hgp
$ nvidia-docker run -it -u user -v $PWD/pdbo-hgp:/home/user/pdbo-hgp -w /home/user/pdbo-hgp pdbo-hgp /bin/bash
~~~

## Experiments

### Distributed EMNIST Classification

1. PDBO-DA, PDBO-MTL, and PDBO-DA&MTL on the fully-connected and static undirected communication networks

~~~
$ python main.py MakeAccuracyTableHyperSGDOnFedEmSetting config.paper.personalization_fedem_hgp --local-scheduler
> ...
+-----------------------------+---------+-----------------------+
|            Method           | Average | Bottom 10% percentile |
+-----------------------------+---------+-----------------------+
|    PDBO-DA (centralized)    |  82.92  |         74.79         |
|    PDBO-MTL (centralized)   |  83.85  |         76.47         |
|  PDBO-DA&MTL (centralized)  |  83.90  |         76.24         |
|   PDBO-DA (decentralized)   |  83.02  |         75.49         |
|   PDBO-MTL (decentralized)  |  83.92  |         76.54         |
| PDBO-DA&MTL (decentralized) |  83.96  |         77.31         |
+-----------------------------+---------+-----------------------+
~~~

2. Baselines on the fully-connected and static undirected communication networks

~~~
$ python main.py MakeAccuracyTableBaselineOnFedEmSetting config.paper.personalization_fedem_baseline --local-scheduler
> ...
+----------------------------------------+---------+-----------------------+
|                 Method                 | Average | Bottom 10% percentile |
+----------------------------------------+---------+-----------------------+
|                 FedAvg                 |  82.24  |         73.76         |
|        FedAvg + local adaption         |  83.03  |         75.14         |
|                 Local                  |  74.67  |         63.87         |
|              Clustered FL              |  82.32  |         73.83         |
|                FedProx                 |  69.61  |         58.16         |
|                 FedEM                  |  83.89  |         75.89         |
|         FedEM (Decentralized)          |  83.82  |         75.89         |
|         FedAvg (Decentralized)         |  82.32  |         74.11         |
+----------------------------------------+---------+-----------------------+
~~~

3. PDBO-DA, PDBO-MTL, and PDBO-DA&MTL on the stochastic undirected and stochastic directed communication networks

~~~
$ python main.py MakeAccuracyTableHyperSGD config.paper.personalization_sgp_hgp --local-scheduler
> ...
+--------------------------+---------+-----------------------+
|          Method          | Average | Bottom 10% percentile |
+--------------------------+---------+-----------------------+
|   PDBO-DA (undirected)   |  80.89  |         73.18         |
|  PDBO-MTL (undirected)   |  81.60  |         73.83         |
| PDBO-DA&MTL (undirected) |  83.05  |         76.29         |
|    PDBO-DA (directed)    |  80.78  |         72.90         |
|   PDBO-MTL (directed)    |  81.62  |         75.00         |
|  PDBO-DA&MTL (directed)  |  82.24  |         74.51         |
+--------------------------+---------+-----------------------+
~~~

4. Baselines on the stochastic undirected and stochastic directed communication networks

~~~
$ python main.py MakeAccuracyTableHyperSGD config.paper.personalization_sgp_baseline --local-scheduler
> ...
+------------------+---------+-----------------------+
|      Method      | Average | Bottom 10% percentile |
+------------------+---------+-----------------------+
|      Local       |  74.67  |         63.87         |
| SGP (undirected) |  79.69  |         71.63         |
|  SGP (directed)  |  79.73  |         72.54         |
+------------------+---------+-----------------------+
~~~

### (Appendix) Comparison of α and β

~~~
$ python main.py PlotZipComputeHyperGradErrorOfSteps config.paper.compare_alpha_beta --local-scheduler
~~~

![vr_error.png](./vr_error.png)

### Notes
- Random generation of static undirected communication networks may vary across the different runs on the same node.
  `processed/BuildGraph_2f4e54164782770979ba9078ca7ef7f2.pt` ensures the identical network communication network as in
  the
  paper experiments.
- Implementations of dataset generation and distributed learning on the fully-connected and static undirected
  communication networks are based on [omarfoq/FedEM](https://github.com/omarfoq/FedEM).

---
If you have questions, please contact Naoyuki
Terashita ([naoyuki.terashita.sk@hitachi.com](mailto:naoyuki.terashita.sk@hitachi.com)).
