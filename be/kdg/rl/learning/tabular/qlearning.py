import numpy as np
import time

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        # TODO move params to config file for easier playing with values
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        # TODO set t_max to 99
        total_rewards = 0

    def learn(self, episode: Episode):
        # TODO implement learn function - update rule Algorithm 4
        percepts = episode.percepts(1) #only get the last percept
        s = percepts.state
        a = percepts.action
        r = percepts.reward
        s2 = percepts.next_state
        done = percepts.done
        self.q_values[s, a] = self.q_values[s, a] + self.α *\
                        (r + self.γ * (np.max(self.q_values[s2, :]) - self.q_values[s, a]))
        total_rewards =+ r
        if r == 1:
            print(f'State: {s} - Action: {a} - Reward: {r}')
            print("\n$$$$$$$$$$$$$$$$$$$")
            print("==== JOEPIIEEE ====")
            print("$$$$$$$$$$$$$$$$$$$")
            print(f'Total rewards: $$$ {total_rewards} $$$\n\n')
            time.sleep(0.4)
        elif done:
            print("==================== DEAD ====================")
            print(f'You fell in the hole after {self.t} timesteps')
            print(f'Total rewards: $$$ {total_rewards} $$$')
            time.sleep(0.4)

        super().learn(episode)

    def evaluate(self):
        # TODO implement evaluate function Algorithm 4
        print("\n===== policy evaluation ====")
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s, :])
        print("v_values are updated")


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
