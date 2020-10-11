import numpy as np

from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.agent.percept import Percept


class MarkovDecisionProcess:
    """
    Abstractie van Environment maken a.d.h.v. een wiskundig model voor de omgeving (Environment)
    Deze klasse is ONVOLLEDIG. Je hebt ze niet direct nodig, maar de implementatie ervan kan wel meer inzicht geven.
    """

    def __init__(self, environment: Environment) -> None:
        self.env = environment  # (given)
        self.n_actions = self.env.n_actions  # (given)
        self.n_states = self.env.state_size  # (given)

        # state-action-next state reward model (learned)
        self._reward_model = np.zeros((self.n_states, self.n_actions))

        # how often state s and action a occurred (learned)
        self.n_sa = np.zeros((self.n_states, self.n_actions))

        # how often has state t followed a state s after action a (learned)
        self.n_tsa = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Markov Decision Process transition model (learned)
        self.P = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Update count
        self.n = 0

    def update(self, percept: Percept) -> None:
        self.n += 1
        self.update_reward(percept)
        self.update_counts(percept)
        self.update_transition_model(percept)

    def update_reward(self, p: Percept) -> None:
        # aan te vullen
        pass

    def update_counts(self, percept: Percept) -> None:
        # aan te vullen
        pass

    def update_transition_model(self, percept: Percept) -> None:
        # aan te vullen
        pass

    def p(self, tsa) -> float:
        return self.P[tsa]

    def reward(self, state: int, action: int) -> float:
        return self._reward_model[state, action]
