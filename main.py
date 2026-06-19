import gymnasium as gym
import numpy as np
from Base_Modules.RL_Algorithms import RL_Algorithm
from Base_Modules.Updaters import *
from Base_Modules.Strategies import *
from Base_Modules.State_Representations import *

env = gym.make("Pendulum-v1")

obs_cardinality = (12, 12, 12)  # cos(theta), sin(theta), theta_dot
obs_space_high = np.array([1.0, 1.0, 8.0])
obs_space_low = np.array([-1.0, -1.0, -8.0])

n_actions = 9
action_space = np.linspace(-2, 2, n_actions)

episodes = 10000

class classic_Q_Learning(RL_Algorithm):
    def __init__(self, ID, alpha : float, gamma : float, eps_decay : float, eps_min : float):
        super().__init__()
        self.strategy = Epsilon_Decay(epsilon = 1.0, epsilon_decay = eps_min, epsilon_min = eps_min)
        self.updater = Q_Learning(alpha=alpha, gamma=gamma)

        # self.updater = SARSA(alpha=0.1, gamma=0.99)
        # self.strategy = Epsilon_Decay(epsilon = 1.0, epsilon_decay = 0.9995, epsilon_min = eps_min)

class classic_SARSA(RL_Algorithm):
    def __init__(self, ID, alpha : float, gamma : float, eps_decay : float, eps_min : float):
        super().__init__()
        self.strategy = Epsilon_Decay(epsilon = 1.0, epsilon_decay = eps_min, epsilon_min = eps_min)
        self.updater = SARSA(alpha=alpha, gamma=gamma)

gammas = [0.99, 0.9]
alphas = [0.1, 0.01]
eps_decays = [0.9995, 0.9999]
eps_mins = [0.05]

algoritms = {}

i = 0
for gamma in gammas:
    for alpha in alphas:
        for eps_decay in eps_decays:
            for eps_min in eps_mins:
                q = classic_Q_Learning(ID=i, alpha=alpha, gamma=gamma, eps_decay=eps_decay, eps_min=eps_min)
                algoritms[i] = q
                i += 1
                s = classic_SARSA(ID=i, alpha=alpha, gamma=gamma, eps_decay=eps_decay, eps_min=eps_min)
                algoritms[i] = s
                i += 1

alg : RL_Algorithm
for alg in algoritms.values():
    q_table = Q_Table(obs_cardinality, obs_space_low, obs_space_high, (n_actions, ))
    alg.Set_State_Representation(q_table)

    for episode in range(episodes):
        obs, _ = env.reset()
        state = q_table.Observation_To_State(obs)

        total_reward = 0
        done = False

        while not done:

            action_index, next_state, reward, terminated, truncated = alg.Step(action_space, state, env)

            state = next_state
            total_reward += reward

            done = terminated or truncated

        alg.Episode_Ended()

        if (episode + 1) % 5000 == 0:
            print(f"Epizod {episode+1}, total reward: {total_reward:.2f}, epsilon: {alg.Get_Strategy().Get_Epsilon():.2f}")

    print(f"Trening zakończony! Total reward: {total_reward:.2f}")

env = gym.make("Pendulum-v1", render_mode = "human")

obs, _ = env.reset()
state =  list(algoritms.values())[-1].Get_State_Representation().Observation_To_State(obs)
strategy = Greedy()
eval_rew = 0

for _ in range(200):
    action_idx : int = strategy.Get_Action_Index(q_table, state)
    action : np.array = np.array(action_space[action_idx])

    obs, rew, terminated, truncated, _ = env.step([action.item()])
    eval_rew += rew
    state = q_table.Observation_To_State(obs)

print(f"Eval rew: {eval_rew}")
env.close()
