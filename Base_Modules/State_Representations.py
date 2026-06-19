import numpy as np

class State_Representation:
    def __init__(self, observation_cardinality : tuple, observation_lower_limits : tuple,
                 observation_upper_limits : tuple, action_cardinality : tuple):
        self.observation_cardinality = observation_cardinality
        self.action_cardinality = action_cardinality

    def Observation_To_State(self, observation : tuple) -> tuple:
        return observation

    def Get_Value_From_State(self, state : tuple) -> float:
        return 0.0

    def Increase_Value(self, state : tuple, increase : float) -> None:
        return

    def Get_Max_Value(self, state : tuple) -> float:
        return 0.0

    def Get_Action_Cardinality(self) -> tuple:
        return self.action_cardinality

    def Get_Observation_Cardinality(self) -> tuple:
        return self.observation_cardinality

    def Write_To_File(self, filename : str) -> None:
        return

class Q_Table(State_Representation):
    def __init__(self, observation_cardinality, observation_lower_limits,
                 observation_upper_limits, action_cardinality : tuple):
        super().__init__(observation_cardinality, observation_lower_limits,
                          observation_upper_limits, action_cardinality)
        self.bin_edges = self.Create_Observation_Bins(observation_cardinality, observation_lower_limits, observation_upper_limits)
        self.q_table = np.zeros(observation_cardinality + action_cardinality)

    def Create_Observation_Bins(self, observation_cardinality : tuple, observation_lower_limits : tuple, observation_upper_limits : tuple):
        return [
            np.linspace(observation_lower_limits[i], observation_upper_limits[i], observation_cardinality[i] - 1)
            for i in range(len(observation_cardinality))
        ]

    def Observation_To_State(self, observation : tuple) -> tuple:
        return tuple(
            np.digitize(observation[i], self.bin_edges[i])
            for i in range(len(observation))
        )

    def Get_Value_From_State(self, state : tuple) -> float:
        return self.q_table[state]

    def Increase_Value(self, state : tuple, increase : float) -> None:
        try:
            self.q_table[state] += increase
        except:
            pass

    def Get_Max_Value(self, state : tuple) -> float:
        return np.max(self.q_table[state])
