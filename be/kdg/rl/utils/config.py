############# PARAMETERS #############
params = {
    "experiment": {
        "default": {
            'description': 'Experiment with default settings',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'nstep': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment1": {
            'description': 'Experiment with completely random agent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'nstep': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment2": {
            'description': 'Experiment with lower discount rate',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'nstep': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        }
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