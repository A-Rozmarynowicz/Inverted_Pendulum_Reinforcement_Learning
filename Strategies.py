import numpy as np
from State_Representations import State_Representation

class Exploration_Strategy:
    def __init__(self):
        pass

    def Get_Action_Index(self, state_representation : State_Representation, state : tuple) -> int:
        return 0

    def Episode_Ended(self) -> None:
        return

class Greedy(Exploration_Strategy):
    def Get_Action_Index(self, state_representation, state):
        return np.argmax(state_representation.Get_Value_From_State(state)).item()

class Epsilon_Greedy(Exploration_Strategy):
    def __init__(self, epsilon : float):
        super().__init__()
        self.epsilon = epsilon

    def Get_Action_Index(self, state_representation : State_Representation, state : tuple):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(state_representation.Get_Action_Cardinality()).item()
        else:
            action_idx = np.argmax(state_representation.Get_Value_From_State(state)).item()
        return action_idx

    def Get_Epsilon(self) -> float:
        return self.epsilon

    def __str__(self):
        return f"Epsilon decay: decay"

class Epsilon_Decay(Epsilon_Greedy):
    def __init__(self, epsilon, epsilon_decay : float, epsilon_min : float):
        super().__init__(epsilon)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def Get_Action_Index(self, state_representation, state):
        return super().Get_Action_Index(state_representation, state)

    def Episode_Ended(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def __str__(self):
        return f"Epsilon decay: decay={self.epsilon_decay}, eps_min={self.epsilon_min}"

