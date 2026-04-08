import gymnasium as gym
import numpy as np

env = gym.make("Pendulum-v1")

# =========================
# Dyskretyzacja stanu
# =========================
bins = (12, 12, 12)  # cos(theta), sin(theta), theta_dot

obs_space_high = np.array([1.0, 1.0, 8.0])
obs_space_low = np.array([-1.0, -1.0, -8.0])

def create_bins():
    return [
        np.linspace(obs_space_low[i], obs_space_high[i], bins[i] - 1)
        for i in range(len(bins))
    ]

bin_edges = create_bins()

def discretize(obs):
    return tuple(
        np.digitize(obs[i], bin_edges[i])
        for i in range(len(obs))
    )

# =========================
# Dyskretyzacja akcji
# =========================
n_actions = 9
actions = np.linspace(-2, 2, n_actions)

# =========================
# Q-table
# =========================
q_table = np.zeros(bins + (n_actions,))

# =========================
# Parametry
# =========================
alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_decay = 0.99995
epsilon_min = 0.05

episodes = 15000

# =========================
# Trening
# =========================
for episode in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)

    total_reward = 0
    done = False

    while not done:  # Pendulum ma limit kroków
        # epsilon-greedy
        if np.random.random() < epsilon:
            action_idx = np.random.randint(n_actions)
        else:
            action_idx = np.argmax(q_table[state])

        action = np.array([actions[action_idx]])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)

        # Q update
        best_next = np.max(q_table[next_state])
        q_table[state + (action_idx,)] += alpha * (
            reward + gamma * best_next - q_table[state + (action_idx,)]
        )

        state = next_state
        total_reward += reward

        done = terminated or truncated

    # epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 1000 == 0:
        print(f"Epizod {episode+1}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")

print("Trening zakończony!")

# =========================
# Test
# =========================
env = gym.make("Pendulum-v1", render_mode="human")

obs, _ = env.reset()
state = discretize(obs)

for _ in range(200):
    action_idx = np.argmax(q_table[state])
    action = np.array([actions[action_idx]])

    obs, _, terminated, truncated, _ = env.step(action)
    state = discretize(obs)

env.close()