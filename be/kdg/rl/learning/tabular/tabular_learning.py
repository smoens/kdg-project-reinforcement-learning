import random
from abc import abstractmethod

import numpy as np
from numpy import ndarray

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class TabularLearner(LearningStrategy):
    """
    A tabular learner implements a tabular method such as Q-Learning, N-step Q-Learning, ...
    """
    π: ndarray          # policy
    v_values: ndarray   # quantification of policy by v-values (based on states)
    q_values: ndarray   # quantification of policy by q-values (based on states and actions)

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        # TODO set t_max to 99
        super().__init__(environment, λ, γ, t_max)
        # learning rate
        self.α = α

        # policy table
        self.π = np.full((self.env.n_actions, self.env.state_size), fill_value=1 / self.env.n_actions)

        # state value table
        self.v_values = np.zeros((self.env.state_size,))

        # state-action table
        self.q_values = np.zeros((self.env.state_size, self.env.n_actions))

    def next_action(self, s: int):
        # TODO implement next_action function - Algorithm 7
        exploitation_tradeoff = random.uniform(0, 1)
        if exploitation_tradeoff > self.ε:
            action = np.argmax(self.π[:, s])
        else:
            action = self.env.action_space.sample()  # just a random next action
        print(f'ε: {self.ε}; Exploitation tradeoff: {exploitation_tradeoff}; N'
              f'ext: {action}; Prop: {round(self.π[action, s],3)}')
        print(f'Policy for state {s} = {self.π[:, s]}')
        return action

    @abstractmethod
    def learn(self, episode: Episode):
        # implementatie uit subklassen komt voor deze statements:
        self.evaluate()
        self.improve()
        super().learn(episode)

    @abstractmethod
    def evaluate(self):
        # hier moet je niets voorzien, maar wel in de subklassen
        pass

    def improve(self):
        # TODO implement improve function - Algorithm 8
        # van q-waarden beste actie eruit halen
        for s in range(self.env.state_size):
            a = np.argmax(self.q_values[s, :])
            for i in range(self.env.n_actions):
                if a == i:
                     self.π[i, s] = 1 - self.ε + self.ε/self.env.n_actions
                else:
                     self.π[i, s] = self.ε/self.env.n_actions
        self.ε = self.ε_min + (self.ε_max - self.ε_min) * np.exp(-self.λ * self.τ)
        #print(f'{self.π}\n\n')


    def start_episode(self):
        self.t = 0
