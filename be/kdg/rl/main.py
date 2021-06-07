from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment
from be.kdg.rl.environment.openai import FrozenLakeNotSlipperyEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning

if __name__ == '__main__':
    environment = FrozenLakeNotSlipperyEnvironment()
    agent: Agent = TabularAgent(environment, Qlearning(environment))
    agent.train()
