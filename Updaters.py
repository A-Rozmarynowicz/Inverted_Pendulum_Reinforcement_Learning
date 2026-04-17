import numpy as np
import State_Representations

class Updater:
    def __init__(self, alpha : float, gamma : float):
        self.alpha = alpha
        self.gamma = gamma

    def Update(self, state_representation : State_Representations.State_Representation):
        pass

class Q_Learning(Updater):
    def Update(self, state_representation : State_Representations.State_Representation, current_state : tuple,
                next_state : tuple, action_idx : int, reward : float):

        best_next = np.max(state_representation[next_state])
        state_representation[current_state + (action_idx,)] += self.alpha * (
            reward + self.gamma * best_next - state_representation[current_state + (action_idx,)]
            )

class SARSA(Updater):
    pass