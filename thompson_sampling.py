from coordination_graph import variable_elimination

import numpy as np
import pandas as pd
import scipy as sp
import random
import copy


class Random_policy():
    def __init__(self, groups, n_agents, seed):

        self._groups = groups
        self._n_agents = n_agents
        self.agents = [f'A{i}' for i in range(n_agents)]
        np.random.seed(seed=seed)
        random.seed(seed)

    def pull(self, iter):
        """
        Returns
        -------
        pd.Series
            A joint arm with the agents' names as columns
        """
        # Sample
        data = np.random.randint(0, 2, self._n_agents)

        df = pd.Series(data, index=self.agents)

        return df



class ThompsonSampling():
    """
    Traditional Thompson sampling mechanism.

    Methods
    -------
    sample()
        Sample a single value for each the mean posteriors.
    pull()
        Pull an arm according to the probability matching mechanism of Thompson sampling.
    update(arm, reward)
        Update an arm's mean posterior with a given reward.
    """

    def __init__(self, arms, prior):
        """
        Parameters
        ----------
        arms : pd.DataFrame
            arms with entries labeled with the associated agent
        priors : list of objects with superclass 'posteriors.Posterior'
            prior for each arm (should be in the same order as arms)
        """
        self._arms = arms
        self._posteriors = prior  # Mean posteriors

        # print('self arms: ', self._arms)



    def sample(self, mean = False):
        """
        Returns
        -------
        list of float
            A sample from every mean's posterior.
        """
        theta = self._arms.copy()

        if mean == False:
            theta['mu'] = [post.sample() for post in self._posteriors]


        else:
            theta['mu'] = [post.mean for post in self._posteriors]





        return theta

    def pull(self, iter):
        """
        Returns
        -------
        pd.Series
            A joint arm with the agents' names as columns
        """
        # Sample
        means = self.sample()



        # Maximize
        max_operator = lambda x: x.loc[x[self.name] == x[self.name].max()]
        a_max = means.loc[means['mu'] == means['mu'].max()]
        a_max.drop(columns='mu', inplace=True)


        return a_max




    def update(self, arm, reward):
        """
        Parameters
        ----------
        arm : pd.Series
            arm with entries labeled with the associated agent
        reward : float
            reward received for executing the arm
        """



        index = np.where((self._arms == arm).all(axis=1))[0][0]
        self._posteriors[index].update(reward)

        

class MultiAgentThompsonSampling():
    """
    Multi-agent Thompson sampling (MATS) mechanism (epsilon exploration)

    Methods
    -------
    sample()
        Sample  from the mean posteriors.
    pull()
        Pull a joint arm according to the probability matching mechanism of MATS.
    update(arm, reward)
        Update an arm's mean posterior with a given reward.
    """

    def __init__(self, groups, priors, epsilon, seed, algo, n_agents):
        """
        Parameters
        ----------
        groups : list of pd.DataFrame
            A data frame for each local group. The data frame consists of every possible local joint arm (rows) jointly over the agents (columns) within the group.
        priors : list of list of objects with superclass 'posteriors.Posterior'
            Each group has a list of priors, i.e., one for the mean of every local joint action.
        """
        # Create local Thompson sampler per group
        self._groups = groups

        self.n_agents = n_agents



        self._groups_samplers = [ThompsonSampling(local_arms, local_priors) for local_arms, local_priors in zip(groups, priors)]

        self.epsilon = epsilon
        self.algo = algo




        np.random.seed(seed=seed)
        random.seed(seed)

        self._n_agents = n_agents
        self.agents = [f'A{i}' for i in range(n_agents)]



    def sample(self, iter):
        """
        Returns
        -------
        list of list of float
            For every group, a sample from every mean's posterior.
        """
        theta = []



        # # Sample per group
        for e, sampler in enumerate(self._groups_samplers):
            if random.random() <= self.epsilon or iter == 0:
                theta_e = sampler.sample(False)

            else:
                theta_e = sampler.sample(True)

            theta_e.rename(columns={'mu': f'mu{e}'}, inplace=True)
            theta.append(theta_e)


        return theta

    def pull(self, iter):
        """
        Returns
        -------
        pd.Series
            A joint arm with the agents' names as columns

        """
        group_means = self.sample(iter)
        a_max = variable_elimination(group_means)
        return a_max

    def update(self, joint_arm, local_rewards):
        """
        Parameters
        ----------
        joint_arm : pd.Series
            arm with entries labeled with the associated agent
        local_rewards : list of float
            For each group, the reward received for executing the local arm
        ----------
        """
        for local_arms, local_sampler, local_reward in zip(self._groups, self._groups_samplers, local_rewards):
          
            local_sampler.update(joint_arm[local_arms.columns], local_reward)


         



