from abc import abstractmethod

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import pandas as pd
import os

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner
from be.kdg.rl.utils.visuals import QValuesVisual, PolicyVisual, ReturnVisual
from be.kdg.rl.utils import config

class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=config.n_episodes-1):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.episode_count = 0
        self.stats = pd.DataFrame({
            "episode_nr": np.arange(1, n_episodes + 2, 1),
            "total_reward": np.empty(n_episodes + 1, dtype=int),
            "avg_reward": np.empty(n_episodes + 1)
        })

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def done(self):
        return self.episode_count > self.n_episodes

    def init_plotting(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.grid()
        xdata, ydata = [], []

        return fig, ax, xdata, ydata, line


class TabularAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=config.n_episodes-1) -> None:
        super().__init__(environment, learning_strategy, n_episodes)
        # TODO set n_episodes to 10000

    def train(self) -> None:
        super(TabularAgent, self).train()

        # as longs as the agents hasn't reached the maximum number of episodes
        while not self.done:

            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state = self.env.reset()
            # reset the learning strategy
            self.learning_strategy.start_episode()

            # Added episode count for easier tracking of episodes
            print(f'\n\nEpisode {self.episode_count + 1}')
            #print(f'=============================')

            # while the episode isn't finished by length
            while not self.learning_strategy.done():

                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action: the next_state and the corresponding reward
                observation = self.env.step(action)[:-1]
                # render environment
                #self.env.render()
                # create Percept from s,a,r,s' and add to Episode
                percept = Percept((state, action) + observation)
                episode.add(percept)

                # learn from one or more Percepts in the Episode
                self.learning_strategy.learn(episode)

                # update state
                state = percept.next_state

                # break if episode is over
                if percept.done:
                    self.results()
                    break

            # end episode
            self.episode_count += 1

        self.stats.to_pickle(
            os.path.join(
                config.params.get("dirs").get("output"),
                config.params.get("experiment").get(config.current_experiment).get("environment"),
                config.current_experiment,
                "results.pkl"))
        self.stats.to_csv(
            os.path.join(
                config.params.get("dirs").get("output"),
                config.params.get("experiment").get(config.current_experiment).get("environment"),
                config.current_experiment,
                "results.csv"))
        self.env.close()

    def results(self):
        self.stats.at[self.episode_count, 'total_reward'] = self.learning_strategy.total_rewards
        self.stats.at[self.episode_count, 'avg_reward'] = \
            np.round(self.learning_strategy.total_rewards / (self.episode_count + 1) * 100, 1)

        if self.episode_count == 0 or (self.episode_count + 1) % config.output_freq == 0:
            # print(f'Total rewards after {self.episode_count + 1} episodes: '
            #         f'$$$ {self.stats.total_reward[self.episode_count]} '
            #         f'({self.stats.avg_reward[self.episode_count]}%) $$$'
            # )
            # print(self.learning_strategy.π)
            ReturnVisual.plot(self.stats[:self.episode_count], self.episode_count)
            QValuesVisual.plot(self.learning_strategy.q_values, self.episode_count)
            PolicyVisual.plot(self.learning_strategy.π, self.episode_count)
            # plt.show()

