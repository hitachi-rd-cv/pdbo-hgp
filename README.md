# Decentralized Hyper-Gradient Computation over Time-Varying Directed Networks

This is the official implementation of the experiments in the following paper:

> Naoyuki Terashita and Satoshi Hara  
> [**Decentralized Hyper-Gradient Computation over Time-Varying Directed Networks**](https://arxiv.org/abs/2210.02129)  
> *arXiv:2210.02129 (under review)*, 2023

## Environments

Our experimental results were made in a NVIDIA Docker container built
from [`docker/Dockerfile`](./docker/Dockerfile).
The container can be obtained by the following steps:

~~~
docker build ./docker/ --tag pdbo-hgp
nvidia-docker run -it -u user -v $PWD/pdbo-hgp:/home/user/pdbo-hgp -w /home/user/pdbo-hgp pdbo-hgp /bin/bash
~~~

## Experiments

### Estimation Error of Hyper-Gradient (Section 5.1 and Appendix A.1)

~~~
# Full-batch g_i
python main.py PlotZipComputeHyperGradErrorOfSteps config.paper.error_synth_fullbatch --local-scheduler
# Mini-batch g_i 
python main.py PlotZipComputeHyperGradErrorOfSteps config.paper.error_synth_minibatch --local-scheduler
~~~

### Decentralized Influence Estimation of Training Instances (Section 5.2 and Appendix A.2)

~~~
# Logistic Regression with full-batch g_i
python main.py CompareApproxActualDiffByMostInfluentialPerturbs config.paper.infl_toy --local-scheduler
# CNN with full-batch g_i
python main.py CompareApproxActualDiffByMostInfluentialPerturbs config.paper.infl_emnist_digits_fullbatch --local-scheduler
# CNN with mini-batch g_i 
python main.py CompareApproxActualDiffByMostInfluentialPerturbs config.paper.infl_emnist_digits_minibatch --local-scheduler
~~~

### Decentralized Personalization for Deep Learning (Section 5.3 and Appendix A.3)

~~~
# Experiments on the fully-connected and static undirected communication networks
## HGP-PL and HGP-MTL 
python main.py MakeAccuracyTableHyperSGDOnFedEmSetting config.paper.personalization_fedem_hgp --local-scheduler
## Baselines with hyperparameter tuning 
python main.py MakeAccuracyTableBaselineOnFedEmSetting config.paper.personalization_fedem_baseline --local-scheduler

# Experiments on the random undirected and random directed communication networks
## HGP-PL and HGP-MTL
python main.py MakeAccuracyTableHyperSGD config.paper.personalization_sgp_hgp --local-scheduler
## Baselines
python main.py MakeAccuracyTableHyperSGD config.paper.personalization_sgp_baseline --local-scheduler
~~~

## Previous versions of paper and codes

- 5 Oct 2022 (v1): _Personalized Decentralized Bilevel Optimization over Stochastic and Directed
  Networks_ ([**Paper**](https://arxiv.org/abs/2210.02129v1), [**Codes**](https://github.com/hitachi-rd-cv/pdbo-hgp/tree/v1))
- 31 Jan 2023 (v2): _Personalized Decentralized Bilevel Optimization over Random Directed
  Networks_ ([**Paper**](https://arxiv.org/abs/2210.02129v2), [**Codes**](https://github.com/hitachi-rd-cv/pdbo-hgp/tree/v2))

---
If you have questions, please contact Naoyuki
Terashita ([naoyuki.terashita.sk@hitachi.com](mailto:naoyuki.terashita.sk@hitachi.com)).
