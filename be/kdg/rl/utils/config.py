import os

############# PARAMETERS #############
current_experiment = "experiment30"
n_episodes = 100
output_freq = 10        # define at what frequency of episodes we want to create output
update_interval = 10    # update interval for deep learning

def init():
    init_folders(current_experiment)


params = {
    # directories for storing output graphs and results
    "dirs": {
        "output": os.path.join("./", "output"),
        "qval": os.path.join("images", "qval"),
        "reward": os.path.join("images", "reward")
    },
    "experiment": {
        "default": {
            'description': 'Experiment QLearning with default settings',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        # ## Qlearning experiments ## #
        "experiment01": {
            'description': 'Experiment QLearning with completely random agent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment02": {
            'description': 'Experiment QLearning',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment03": {
            'description': 'Experiment QLearning with lower discount rate',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment04": {
            'description': 'Experiment QLearning with decreased learning rate',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment05": {
            'description': 'Experiment QLearning with very small learning rate',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.2,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment06": {
            'description': 'Experiment QLearning with large learning rate',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.9,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment07": {
            'description': 'Experiment QLearning with very small discount rate',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.1,
            't_max': 99
        },
        "experiment08": {
            'description': 'Experiment QLearning',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'n': None,
            'α': 0.7,
            'λ': 0.001,
            'γ': 0.9,
            't_max': 99
        },
        # ## NStepQLearning experiments ## #
        "experiment10": {
            'description': 'Experiment NStepQLearning',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '3',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment11": {
            'description': 'Experiment NStepQLearning with lower discount',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '3',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment12": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '7',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment13": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '2',
            'α': 0.6,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment14": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '2',
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.6,
            't_max': 99
        },
        "experiment15": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '2',
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.4,
            't_max': 99
        },
        "experiment16": {
            'description': 'Experiment NStepQLearning',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'n': '1',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        # ## Monte Carlo experiments ## #
        "experiment20": {
            'description': 'Experiment MonteCarloLearning',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'MonteCarloLearning',
            'n': None,     #nstep
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        # ## DeepQLearning experiments ## #
        "experiment30": {
            'description': 'Experiment DeepQLearning',
            'environment': 'CartPoleEnvironment',
            'learning': 'DeepQLearning',
            'n': '4',       #batch_size
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
    },
    "models": {
        "model1": {
            "model_filename": "model1.h5",
            "lr": 0.001,
            "num_epochs": 10,
            "validation_steps": 10,
            "metrics": ["acc"],
            "seed": 1354874613,
        }
    },
    "environments": {
        "FrozenLake": "FrozenLakeEnvironment",
        "CartPole": "CartPoleEnvironment"
    },
    "methods": [
        "Qlearning",
        "NStepQLearning",
        "MonteCarloLearning",
        "DeepQLearning",
        "DoubleDeepQLearning"
    ]
}


def init_folders(experiment):
    path_experiment = os.path.join(
        params.get("dirs").get("output"),
        params.get("experiment").get(current_experiment).get("environment"),
        experiment)
    path_qval = params.get("dirs").get("qval")
    path_reward = params.get("dirs").get("reward")
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
        os.makedirs(os.path.join(path_experiment, path_qval))
        os.makedirs(os.path.join(path_experiment, path_reward))
    return
