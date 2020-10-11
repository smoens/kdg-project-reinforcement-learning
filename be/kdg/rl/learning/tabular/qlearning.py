from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment) -> None:
        TabularLearner.__init__(self, environment)

    def learn(self, episode: Episode):
        # HIER AANVULLEN

        super().learn(episode)

    def evaluate(self):
        # HIER AANVULLEN
        pass


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int) -> None:
        TabularLearner.__init__(self, environment)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        # HIER AANVULLEN

        super().learn(episode)

    def evaluate(self):
        # HIER AANVULLEN
        pass
