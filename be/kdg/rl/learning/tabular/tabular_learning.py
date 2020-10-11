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
    π: ndarray
    v_values: ndarray
    q_values: ndarray

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
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
        # HIER AANVULLEN
        pass

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
        # HIER AANVULLEN
        pass

    def start_episode(self):
        self.t = 0
