from environments import Bernoulli0101Chain, Poisson0101Chain
from posteriors import BetaPosterior, GaussianPosterior
from thompson_sampling import MultiAgentThompsonSampling, Random_policy
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

import argparse

from tqdm import tqdm
import time
import csv




def training(n_iter, policy, env):
    total_regret = 0

    regrets = []

    for i in tqdm(range(n_iter)):
        # Do step with MATS
       
        joint_arm = policy.pull(i)
        

        local_rewards = env.execute(joint_arm)


        if args.algo != 'rd':
            policy.update(joint_arm, local_rewards)

        # Logging
        regret = env.regret(joint_arm)

        total_regret += regret
        regrets.append(total_regret)

        

    return regrets



def chain_experiment(args):
    n_iter = args.iter
    n_agents = args.n_agents

    print('start env')
    if args.env_name == 'bernoulli':
        env = Bernoulli0101Chain(n_agents, args.seed)
    elif args.env_name == 'poisson':
        env = PoissonChain(n_agents, args.seed)
    print('create env')



    # Create priors
    if args.priors == 'beta':
        priors = [[BetaPosterior(0.5, 0.5) for _ in range(arms.shape[0])] for arms in env.groups]
    elif args.priors == 'gaussian':
        priors = [[GaussianPosterior(std=1) for _ in range(arms.shape[0])] for arms in env.groups]



    if args.algo == 'all':

        for args.seed in range(0, 20):
           
            epsilons =  [0.005, 0.01, 0.1, 0.5, 0.8, 1.0]
            print('random seed: ', args.seed)
            dict = {}

            for epsilon in epsilons:
                if epsilon == 1.0:
                    args.algo = 'mats'
                else:
                    args.algo = 'ep_mats' 



                priors = [[GaussianPosterior(std=1) for _ in range(arms.shape[0])] for arms in env.groups]
                print('epsilon: ', epsilon)


                policy = MultiAgentThompsonSampling(env.groups, priors, epsilon, args.seed, args.algo, n_agents)
                print('create policy')
            
                start_time = time.time()

                dict[str(epsilon)] = training(n_iter, policy, env)
                

 
     
    elif args.algo == 'rd':
        dict = {}
        for args.seed in range(0, 20):
            policy = Random_policy(env.groups, n_agents, args.seed)
            dict[str(args.seed)] = training(n_iter, policy, env)
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', type=str, default='all')
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--priors', type=str, default='gaussian')
    parser.add_argument('--env_name', type=str, default='bernoulli')
    parser.add_argument('--n_agents', type=int, default=10)



    args = parser.parse_args()

    chain_experiment(args)




