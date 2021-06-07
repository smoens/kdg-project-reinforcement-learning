# https://medium.com/swlh/introduction-to-q-learning-with-openai-gym-2d794da10f3d
# https://valohai.com/blog/reinforcement-learning-tutorial-part-1-q-learning/
# https://www.kaggle.com/sarjit07/reinforcement-learning-using-q-table-frozenlake
# https://cwong8.github.io/projects/FrozenLake/ : nice example with visual representatioin of policy
# https://github.com/katjawittfoth/Frozen_Lake

# https://github.com/codeheroku/Introduction-to-Machine-Learning/tree/master/Reinforcement%20Learning
    #YT: https://www.youtube.com/watch?v=3begG_s9lzg
# https://github.com/Procuste34/IntroRL/tree/master/I%20-%20Model-based/code/4%20-%20Gym%20:%20Frozen%20Lake
    #YT: https://www.youtube.com/watch?v=-tvLISFkkK8

# https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/

import numpy as np

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        # TODO set t_max to 99

    def learn(self, episode: Episode):
        # TODO implement learn function - update rule Algorithm 4
        percepts = episode.percepts(episode.size-1)
        s = percepts.state
        a = percepts.action
        r = percepts.reward
        s2 = percepts.next_state
        self.q_values[s, a] = self.q_values[s, a] + self.α *\
                        (r + self.γ * (np.max(self.q_values[s2, :]) - self.q_values[s, a]))

        if r == 1:
            print(f'State: {s} - Action: {a} - Reward: {r}')
            print("==== JOEPIIEEE ====")
            print("$$$$$$$$$$$$$$$$$$$\n\n")
        super().learn(episode)

    def evaluate(self):
        # TODO implement evaluate function Algorithm 4
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s, :])


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        # TODO implement learn function

        super().learn(episode)

    def evaluate(self):
        # TODO implement evaluate function
        pass
