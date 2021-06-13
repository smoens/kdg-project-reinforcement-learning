from tensorflow.keras import Model
import numpy as np
import random
from collections import deque
from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.model import model1
from be.kdg.rl.utils import config

class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    Deze klasse is ONVOLLEDIG.
    """
    q1: Model   #neural network that will be trained; used to predict best action
    q2: Model   #used to build training set for q1

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn

        self.c = config.update_interval
        self.q1 = model1.create_model("model1", self.env.state_size, self.env.n_actions)
        self.q2 = model1.create_model("model1", self.env.state_size, self.env.n_actions)
        self.max_timesteps = 0  #save longest balancing

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        exploitation_tradeoff = random.uniform(0, 1)
        if exploitation_tradeoff > self.ε:
            action = np.argmax(self.q1.predict(state))
        else:
            action = self.env.action_space.sample()  # just a random next action
        return action

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        if episode.size >= self.batch_size:
            percepts = episode.sample(self.batch_size)
            self.learn_from_batch(percepts)
        super().learn(episode)

    def start_episode(self):
        self.t = 0

    def learn_from_batch(self, percepts):
        count = 0
        training_data = self.build_training_set(percepts)  # transform percepts so they can be used by a neural network
        self.train_network(training_data)

        count += 1
        if count % self.c:
            self.q2.set_weights(self.q1.get_weights())

    def build_training_set(self, percepts):
        training_data = deque()
        for p in percepts:  # random sample of percepts
            #print(f'p: {p}')
            s = p.state
            a = p.action
            r = p.reward
            s2 = p.next_state
            done = p.done

            q_values = self.q1.predict(np.reshape(s, [1, self.env.state_size]))
            if self.ddqn:
                optimal_a = np.argmax(self.q1.predict(np.reshape(s2, [1, self.env.state_size])))
                optimal_q = self.q2.predict(np.reshape(s2, [1, self.env.state_size]))[optimal_a]
            else:
                optimal_q = np.max(self.q2.predict(np.reshape(s2, [1, self.env.state_size]))) #Q2 wordt gebruikt om een training set te bouwen voor Q1

            if done:
                q_values[0][a] = r
            else:
                q_values[0][a] = r + self.γ * optimal_q
            training_data.append((s, q_values[0][a]))  # koppeling van huidige state aan toekomstige informatie (predictie 2de netwerk)
        return training_data

    def train_network(self, training_data):  # train the network q1
        for s, q in training_data:
            s_reshape = np.reshape(s, (1,self.env.state_size))
            q_reshape = np.asarray([[q]])
            self.q1.fit(s_reshape, q_reshape, batch_size=self.batch_size, verbose=0)   # training of Q1
