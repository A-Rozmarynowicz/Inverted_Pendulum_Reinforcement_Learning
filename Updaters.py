import numpy as np
from State_Representations import State_Representation
from Strategies import Exploration_Strategy

class Updater:
    def __init__(self, alpha : float, gamma : float):
        self.alpha = alpha
        self.gamma = gamma

    def Update(self, strategy : Exploration_Strategy, state_representation : State_Representation):
        pass

    def Select_Action_Index(self, state_representation : State_Representation, current_state : int, strategy : Exploration_Strategy) -> int:
        return 0

    def Episode_Ended(self) -> None:
        pass

class Q_Learning(Updater):
    def Update(self, strategy : Exploration_Strategy,  state_representation : State_Representation, current_state : tuple,
                next_state : tuple, action_idx : int, reward : float):

        best_next = state_representation.Get_Max_Value(next_state)
        delta : float = self.alpha * (
            reward + self.gamma * best_next - state_representation.Get_Value_From_State(current_state + (action_idx,))
            )
        state_representation.Increase_Value(current_state + (action_idx,), delta)

    def Select_Action_Index(self, state_representation : State_Representation, current_state : int, strategy : Exploration_Strategy) -> int:
        return strategy.Get_Action_Index(state_representation, current_state)

    def __str__(self):
        return f"Q-Learning: gamma={self.gamma}, alpha={self.alpha}"

class SARSA(Updater):
    def __init__(self, alpha, gamma):
        super().__init__(alpha, gamma)
        self.next_action : int = -1

    def Update(self, strategy : Exploration_Strategy, state_representation : State_Representation, current_state : tuple,
                next_state : tuple, action_idx : int, reward : float):

        self.next_action = strategy.Get_Action_Index(state_representation, next_state)

        delta : float = self.alpha * (
            reward
            + self.gamma * state_representation.Get_Value_From_State(next_state + (self.next_action,))
            - state_representation.Get_Value_From_State(current_state + (action_idx,))
            )
        state_representation.Increase_Value(current_state + (action_idx,), delta)

    def Select_Action_Index(self, state_representation : State_Representation, current_state : int, strategy : Exploration_Strategy) -> int:
        if self.next_action == -1:
            return strategy.Get_Action_Index(state_representation, current_state)
        return self.next_action

    def __str__(self):
        return f"SARSA: gamma={self.gamma}, alpha={self.alpha}"

    def Episode_Ended(self):
        self.next_action = -1
