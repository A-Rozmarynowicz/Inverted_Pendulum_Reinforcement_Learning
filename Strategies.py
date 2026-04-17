from State_Representations import State_Representation

class Exploration_Strategy:
    def __init__(self):
        pass

    def Get_Action_Index(self, state_representation : State_Representation) -> int:
        return 0

    def Update(self) -> None:
        return

class Epsilon_Greedy(Exploration_Strategy):
    def __init__(self, epsilon : float):
        super().__init__()
        self.epsilon = epsilon

