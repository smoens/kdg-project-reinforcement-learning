from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment, FrozenLakeNotSlipperyEnvironment, CartPoleEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning
from be.kdg.rl.learning.approximate.deep_qlearning import DeepQLearning
from be.kdg.rl.agent.episode import Episode
import matplotlib

if __name__ == '__main__':
    matplotlib.use("TkAgg")                 #activate tkinter
    environment = FrozenLakeEnvironment()
    agent: Agent = TabularAgent(environment, Qlearning(environment))
    agent.train()
