from abc import ABC, abstractmethod

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
import numpy as np

class LearningStrategy(ABC):
    """
    Implementations of this class represent a Learning Method
    Deze klasse is ONVOLLEDIG
    """
    env: Environment

    def __init__(self, environment: Environment, λ, γ, t_max) -> None:
        self.env = environment
        self.λ = λ  # exponential decay rate used for exploration/exploitation (given)
        self.γ = γ  # discount rate for exploration (given)
        self.ε_max = 1.0  # Exploration probability at start (given)
        self.ε_min = 0.0005  # Minimum exploration probability (given)

        self.ε = self.ε_max  # (decaying) probability of selecting random action according to ε-soft policy
        self.t_max = t_max  # upper limit voor episode
        self.t = 0  # episode time step
        self.τ = 0  # overall time step

    @abstractmethod
    def next_action(self, state):
        pass

    @abstractmethod
    def learn(self, episode: Episode):
        # implementatie uit subklassen komt voor deze statements:
        self.t += 1
        self.τ += 1

    @abstractmethod
    def start_episode(self):
        """ Implements all necessary initialization that needs to be done at the start of new Episode """
        pass

    def decay(self):
        # Reduce epsilon ε, because we need less and less exploration as time progresses
        self.ε = self.ε_min + (self.ε_max - self.ε_min) * np.exp(-self.λ * self.τ)

    def done(self):
        return self.t > self.t_max
