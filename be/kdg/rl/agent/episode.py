from collections import deque
    # https://docs.python.org/3/library/collections.html#collections.deque
    # https://pythontic.com/containers/deque/index
import numpy as np

from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment
from numpy.random import choice

class Episode:
    """
    Een verzameling van Percepts vormt een Episode. Per stap/tijdstip t wordt een Percept toegevoegd.
    Deze klasse zal je moeten aanvullen met nieuwe functionaliteiten
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._percepts: [Percept] = deque()
        self.Gt = 0  # return Gt (discounted sum of rewards)

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Haal n laatste percepts op uit Episode """
        return list(self._percepts)[-n:]

    def compute_returns(self, t, λ) -> int:  # oorspronkelijke functie compute_returns(self) -> None
        """ Bereken voor elke Percept uit de Episode zijn discounted return Gt"""
        if t < (self.size - 1):
            p = self._percepts[t+1]
            self.Gt += np.exp(λ, t) * p.reward
            t += 1
            self.compute_returns()
        else:
            return self.Gt  #is deze return niet nodig als we het voortschrijdend gemiddelde van de discounted rewards willen berekenen?
            print(f'The discounted sum of rewards at timestamp {t}: {self.Gt}')

    def sample(self, batch_size: int):
        """ Sample een willekeurige batch uit deze Episode """
        sample = choice(self._percepts, batch_size)
        return sample

    @property
    def size(self):
        return len(self._percepts)
