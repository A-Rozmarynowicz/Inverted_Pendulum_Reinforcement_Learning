import gymnasium as gym
import numpy as np
from RL_Algorithms import RL_Algorithm
from Updaters import *
from Strategies import *
from State_Representations import *

env = gym.make("Pendulum-v1")

class classic_Q_Learning(RL_Algorithm):
    def __init__(self):
        super().__init__()
        self.strategy = Epsilon_Decay(epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.05)
        self.updater = Q_Learning(alpha=0.1, gamma=0.99)

QL = classic_Q_Learning()


obs_cardinality = (12, 12, 12)  # cos(theta), sin(theta), theta_dot
obs_space_high = np.array([1.0, 1.0, 8.0])
obs_space_low = np.array([-1.0, -1.0, -8.0])

n_actions = 9
action_space = np.linspace(-2, 2, n_actions)

q_table = Q_Table(obs_cardinality, obs_space_low, obs_space_high, (n_actions, ))

episodes = 10000

for episode in range(episodes):
    obs, _ = env.reset()
    state = q_table.Observation_To_State(obs)

    total_reward = 0
    done = False

    while not done:

        action_index, next_state, reward, terminated, truncated = QL.Step(q_table, action_space, state, env)

        state = next_state
        total_reward += reward

        done = terminated or truncated

    QL.Episode_Ended()

    if (episode + 1) % 1000 == 0:
        print(f"Epizod {episode+1}, total reward: {total_reward:.2f}, epsilon: {QL.Get_Strategy().Get_Epsilon():.2f}")

print("Trening zakończony! Total reward: {total_reward:.2f}")

env = gym.make("Pendulum-v1", render_mode = "human")

obs, _ = env.reset()
state = q_table.Observation_To_State(obs)
strategy = Greedy()

for _ in range(200):
    action_idx : int = strategy.Get_Action_Index(q_table, state)
    action : np.array = np.array(action_space[action_idx])

    obs, _, terminated, truncated, _ = env.step([action.item()])
    state = q_table.Observation_To_State(obs)

env.close()