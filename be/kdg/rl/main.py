from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment, FrozenLakeNotSlipperyEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning

if __name__ == '__main__':
    environment = FrozenLakeNotSlipperyEnvironment()
    agent: Agent = TabularAgent(environment, MonteCarloLearning(environment, n=4))
    agent.train()
