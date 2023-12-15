# Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs

This repo contains code for studying multi-agent multi-armed bandit (MAMAB) problem in our work Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs. We evaluate our proposed ϵ-MATS on several benchmark MAMAB problem.


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





