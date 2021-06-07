from collections import deque
    # https://docs.python.org/3/library/collections.html#collections.deque
from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment


class Episode:
    """
    Een verzameling van Percepts vormt een Episode. Per stap/tijdstip t wordt een Percept toegevoegd.
    Deze klasse zal je moeten aanvullen met nieuwe functionaliteiten
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._percepts: [Percept] = deque()

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Haal n laatste percepts op uit Episode """
        return self._percepts[n]

    def compute_returns(self) -> None:
        """ Bereken voor elke Percept uit de Episode zijn discounted return Gt"""
        # TODO HIER AANVULLEN
        pass

    def sample(self, batch_size: int):
        """ Sample een willekeurige batch uit deze Episode """
        # TODO HIER AANVULLEN
        pass

    @property
    def size(self):
        return len(self._percepts)
