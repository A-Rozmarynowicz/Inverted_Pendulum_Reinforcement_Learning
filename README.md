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
Having compared Q-Learning's and SARSA's effectiveness in regard to parameters (learning rate, discount factor, epsilon decay), their outcome is rather similar in this environment. Figure 1 shows how rewards change with respect to episodes passed, for the best parameter configuration:

<p align="center">
learning rate (alpha) = 0.05 </br>
discount factor (gamma) = 0.95 </br>
epsilon decay (ϵd) = 0.9995 </br>
total number of episodes = 8000 </br>
</p>

<p align="center">
<img src="./Images/QL_vs_SARSA_curve.png" alt="./Images/QL_vs_SARSA_curve.png" width="85%"><br>
<em>Figure 1: Q-Learning vs SARSA: <br> The lines in the background are rewards over episodes during training. <br> The dimmer lines in the front are rewards over episodes, smoothed over 50 steps.</em>
</p>

We see that both algorithms performed well.

> For the purpose of the project, a more in-depth analysis of parameters' impact was conducted; however we have decided not to share it here.

## Visualization
Below you can see a GIF showing a pendulum controled by a trained Q-Learning model:



A quick presentation of the training process along with a few more examples can be seen in [this YouTube video](https://www.youtube.com/watch?v=yqmj4zeWN_Q).

## Technologies used
- Python
- numpy
- pandas

## Setup

Create python3 virtual enviroment:
> python3 -m venv .venv

Install packages:
> pip install -r requirements.txt