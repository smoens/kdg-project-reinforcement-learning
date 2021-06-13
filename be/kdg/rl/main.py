from be.kdg.rl.agent.agent import DQNAgent, TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment, FrozenLakeNotSlipperyEnvironment, CartPoleEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning
from be.kdg.rl.learning.approximate.deep_qlearning import DeepQLearning
from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.utils import config
import matplotlib

if __name__ == '__main__':
    matplotlib.use("TkAgg")                 # activate tkinter
    experiment = config.current_experiment  # choose experiment to run
    config.init()                           # initialize local environment

    learning = config.params.get("experiment").get(experiment).get("learning")
    env_name = config.params.get("experiment").get(experiment).get("environment") + "()"
    agent = config.params.get("experiment").get(experiment).get("agent")
    n = config.params.get("experiment").get(experiment).get("n")
    ddqn = config.params.get("experiment").get(experiment).get("ddqn")

    environment = eval(env_name)
    agent: Agent = eval(agent + "(environment," + learning + "(environment" +
                        ("))" if n is None and ddqn is None else "," + n) +
                        ("))" if ddqn is None else "," + ddqn + "))")
                    )
    agent.train()
