# https://github.com/NikhilDange1/Cartpole-DQN-agent/blob/master/DQNagent.py
#

from tensorflow.keras import Model
    # https://keras.io/api/models/model/
import numpy as np
from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    Deze klasse is ONVOLLEDIG.
    """
    q1: Model   #neural network that will be trained
    q2: Model   #used to build training set for q1

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn
        # TODO HIER AANVULLEN
        self.c = 5 # update interval
        self.q1 = 0 # create (dense?) neural network
        self.q2 = 0 # create (dense?) neural network

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        # TODO add function from Algorithm 10
        return self.env.action_space.sample()  # just a random next action

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        if episode.size >= self.batch_size:
            percepts = episode.sample(self.batch_size)
            self.learn_from_batch(percepts)
        super().learn(episode)
        pass

    def start_episode(self):
        # TODO implement start_episode
        self.t = 0


    def learn_from_batch(self, percepts):
        count = 0
        training_data = self.build_training_set(percepts)
        self.train_network(training_data)

        count += 1
        θ1 = self.q1.get_weights()
        θ2 = self.q2.get_weights()
        if count % self.c:
            θ2 = θ1


    def build_training_set(self, percepts):
        training_data = []
        for p in percepts:
            print(p)
        return 0

    def train_network(self, D):
        pass
