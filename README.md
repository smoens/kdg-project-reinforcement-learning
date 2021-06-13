# 🤖 Reinforcement Learning project - Karel de Grote Hogeschool

This repository contains the code for solving two reinforcement learning problems provided by [OpenAI Gym](https://gym.openai.com/) as part of my Artificial Intelligence course: 
* [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)
* [Cartpole-v1](https://gym.openai.com/envs/CartPole-v1/)

The initial code structure was provided by Wouter De Ketelaere. In addition to the implementation of the functions, I've added a models and utils directory. See below for the complete structure.

## 🧐 What's inside?
The code breaks down in 5 big blocks:
* agent
* environment
* learning: learning strategies the agent can apply
* model: contains models used for neural network learning strategies
* utils: contains helper functions and configuration file for easy tracking of expirements

    .
    ├── **agent**
    ├──── agent.py
    ├──── episode.py
    ├──── percept.py            
    ├── **environment**
    ├──── environment.py
    ├──── markovdecisionprocess.py
    ├──── openai.py            
    ├── **learning**
    ├──── tabular
    ├────── tabular_learning.py
    ├────── qlearning.py        
    ├──── learning_strategy.py        
    ├── **model**
    ├──── model1.py
    ├── **utils**
    ├──── config.py
    ├──── visuals.py        
    ├── main.py

## 🧊 Frozen Lake
Reference: [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)

Applied methods:
* q-learning
* n-step q-learning
* monte carlo

To test the different methods experiments were conducted to evaluate performance of the different methods or different configurations of the same method. All these experiments are described in the config-file. With the QLearning-method, a score of 70% (expressed as average rewards) was achieved (α=0.7, λ=0.0005, γ=0.9).

## ⚖️ Cartpole
Reference: [Cartpole-v1](https://gym.openai.com/envs/CartPole-v1/)

Applied methods:
* deep q-learning (DQN)
* double deep q-learning (DDQN)