import numpy as np
import time

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.5, t_max=199) -> None: #α=0.7
        # TODO move params to utils file for easier playing with values
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)

    def learn(self, episode: Episode):
        # update rule Algorithm 4
        percepts = episode.percepts(1)[0]  # only get the last percept
        s = percepts.state
        a = percepts.action
        r = percepts.reward
        s2 = percepts.next_state
        done = percepts.done
        self.q_values[s, a] = self.q_values[s, a] + self.α *\
                        (r + self.γ * (np.max(self.q_values[s2, :]) - self.q_values[s, a]))
        self.total_rewards += r
        # if r == 1:
        #     print(f'State: {s} - Action: {a} - Reward: {r}')
        #     print("\n$$$$$$$$$$$$$$$$$$$")
        #     print("==== JOEPIIEEE ====")
        #     print("$$$$$$$$$$$$$$$$$$$")
        #     print(f'Total rewards: $$$ {self.total_rewards} $$$\n\n')
        #     time.sleep(0.6)
        # elif done:
        #     print("==================== DEAD ====================")
        #     print(f'You fell in the hole after {(self.t+1)} timesteps')
        #     print(f'Total rewards: $$$ {self.total_rewards} $$$')
        #     time.sleep(0.4)

        # compute return
        episode.compute_returns(t=self.t, λ=self.λ)

        super().learn(episode)

    def evaluate(self):
        # TODO Algorithm 4
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s, :])
        #print("\n=Policy evaluation: v_values are updated=")


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        if episode.size >= self.n:
            for p in episode.percepts(self.n):
                s = p.state
                a = p.action
                r = p.reward
                s2 = p.next_state
                done = p.done
                self.q_values[s, a] = self.q_values[s, a] + self.α * \
                                      (self.q_values[s, a] - (r + self.γ * (np.max(self.q_values[s2, :]))))
            super().learn(episode)

    def evaluate(self):
        # TODO implement evaluate function Algorithm 4
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s, :])
        pass


class MonteCarloLearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        for p in episode.percepts(episode.size):
            s = p.state
            a = p.action
            r = p.reward
            s2 = p.next_state
            done = p.done
            self.q_values[s, a] = self.q_values[s, a] + self.α * \
                                  (self.q_values[s, a] - (r + self.γ * (np.max(self.q_values[s2, :]))))
        super().learn(episode)

    def evaluate(self):
        # TODO implement evaluate function Algorithm 4
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s, :])
        pass
