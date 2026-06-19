![IT](https://img.shields.io/badge/IT-purple)

![Software](https://img.shields.io/badge/Software-lightblue)

# Inverted Pendulum Control: Reinforcement Learning approach
A reinforcement  learning framework for gymnasium  environments. Allows for flexible reconfiguration of modules like: update policy, state representation, and strategy. Showcased on the [inverted pendulum environment](https://gymnasium.farama.org/environments/classic_control/pendulum/) from Gymnasium.

Authors: Selim Mucharski, Arkadiusz Rozmarynowicz

Development time: from 04.2026 to 06.2026

## Table of contents


## General information
 - Implemented:
    - Q-Learning and SARSA algorithms,
    - epsilon-greedy strategy,
    - Q-table with state discretization.
- Allowed for combining various algorithms, strategies, and state representations in a flexible way.
- Trained models on the inverted pendulum control problem.
- Tested and analyzed parameters' impact on effectiveness.

## Outcome
Having compared Q-Learning's and SARSA's effectiveness in regard to parameters (learning rate, discount factor, epsilon decay), their outcome is rather similar in this environment.

## Technologies used
- Python
- numpy
- pandas

## Setup

Create python3 virtual enviroment:
> python3 -m venv .venv

Install packages:
> pip install -r requirements.txt