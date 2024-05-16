# Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs (ϵ-MATS)

### <p align="center">[AAAI 2024 Oral]</p>

<p align="center">
  <a href="https://tianyuanjin.github.io/">Tianyuan Jin</a><sup>*</sup> ·
  <a href="https://hlhsu.github.io/">Hao-Lun Hsu</a><sup>†</sup> ·
  <a href="https://williamc.me/">William Chang</a><sup>‡</sup> ·
  <a href="https://panxulab.github.io/">Pan Xu</a><sup>†</sup>
</p>
<p align="center">
<sup>*</sup> National University of Singapore ·
<sup>†</sup> Duke University ·
<sup>‡</sup> University of California, Los Angles
</p>



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
## Citation
```
@inproceedings{Jin2024MATS,
  title={Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs},
  author={Jin, Tianyuan and Hsu, Hao-Lun and Chang, William and Xu, Pan},
  booktitle={Annual AAAI Conference on Artificial Intelligence (AAAI)},
  volume={38},
  number={11},
  pages={12956--12964},
  year={2024}
}
```




