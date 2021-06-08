from tensorflow.keras import Model

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    Deze klasse is ONVOLLEDIG.
    """
    q1: Model
    q2: Model

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn
        # TODO HIER AANVULLEN

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        pass

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        super().learn(episode)
        pass

    def start_episode(self):
        # TODO implement start_episode
        pass


    def learn_from_batch(self, P):
        count = 0
        D = self.build_training_set()
        pass

    def build_training_set(self, P):
        return 0
        pass

    def train_network(self, D):
        pass
