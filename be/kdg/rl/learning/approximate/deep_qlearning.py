from tensorflow.keras import Model
    # https://keras.io/api/models/model/
import numpy as np
from collections import deque
from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.model import model1


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
        # TODO HIER AANVULLEN
        self.c = 5  # update interval
        self.q1 = model1.create_model("model1", self.env.state_size, self.env.n_actions)
        self.q2 = model1.create_model("model1", self.env.state_size, self.env.n_actions)
        self.q_values = np.zeros((self.env.state_size, self.env.n_actions))

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        # TODO add function from Algorithm 10
        return self.env.action_space.sample()  # just a random next action

    def learn(self, episode: Episode):
        # TODO = CHECK
        """ Sample batch from Episode and train NN on sample"""
        if episode.size >= self.batch_size:
            percepts = episode.sample(self.batch_size)
            self.learn_from_batch(percepts)
        super().learn(episode)

    def start_episode(self):
        self.t = 0


    def learn_from_batch(self, percepts):
        # TODO = CHECK
        count = 0
        training_data = self.build_training_set(percepts)  #transform percepts so they can be used by a neural network
        self.train_network(training_data)

        count += 1
        if count % self.c:
            self.q2.set_weights(self.q1.get_weights())


    def build_training_set(self, percepts):
        # TODO = CHECK
        #   Q2 wordt gebruikt om een training set te bouwen voor Q1
        #   Q1 wordt getraind
        [training_data] = deque()
        for p in percepts:  # random sample of percepts
            s = p.state
            a = p.action
            r = p.reward
            s2 = p.next_state
            done = p.done

            optimal_q = self.q1.predict(s2)
            if done:
                self.q_values[s, a] = r
            else:
                self.q_values[s, a] = r + self.γ * np.max(optimal_q)
            training_data.append(s, self.q_values[s, a])
        return training_data

    def train_network(self, training_data):  # train the network q1
        # TODO = CHECK
        for (s, q) in training_data:
            self.q1.fit(s, q, batch_size=self.batch_size)
