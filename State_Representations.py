import numpy as np

class State_Representation:
    def __init__(self, observation_cardinality : tuple, observation_lower_limits : tuple, observation_upper_limits : tuple):
        self.bin_edges = self.Create_Bins()

    def Observation_To_State(self, observation : tuple) -> tuple:
        return observation

    def Create_Bins(observation_cardinality : tuple, observation_lower_limits : tuple, observation_upper_limits : tuple):
        return [
            np.linspace(observation_lower_limits[i], observation_upper_limits[i], observation_cardinality[i] - 1)
            for i in range(len(observation_cardinality))
        ]

class Q_Table(State_Representation):
    def Observation_To_State(self, observation : tuple) -> tuple:
        return tuple(
            np.digitize(observation[i], self.bin_edges[i])
            for i in range(len(observation))
        )