# Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs (ϵ-MATS)

[Tianyuan Jin](https://tianyuanjin.github.io/) · [Hao-Lun Hsu](https://hlhsu.github.io/) · [William Chang](https://williamc.me/) · [Pan Xu](https://panxulab.github.io/)



Official implementation of the paper "Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs (ϵ-MATS)" which combines the MATS exploration with probability ε and greedy exploitation with probability 1 − ε.


## Installation instructions


### Dependencies
- python==3.6
- scipy >=1.2.1
- matplotlib >= 3.0.2
- pandas >= 0.25.3
- numpy >= 1.17.0



## Example

```python
# Enter the anaconda virtual environment
source activate epsilon_mats
# Train on Bernoulli0101 using random exploration on 10 agents
python main.py --algo rd --env_name bernoulli --iter 2000 --seed 0 --n_agents 10

# Train on Poisson0101 using mats (including different epsilon) on 20 agents
python main.py --algo all --env_name poisson --iter 2000 --seed 0 --n_agents 20
```





