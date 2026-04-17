from State_Representations import State_Representation
import Strategies
import Updaters
import numpy as np

class RL_Algorithm:
    def __init__(self):
        self.strategy : Strategies.Exploration_Strategy
        self.updater : Updaters.Updater

    def Step(self, state_representation : State_Representation, action_space : np.array, current_state : tuple, environment):
        action_index : int = self.strategy.Get_Action_Index(state_representation)
        action : np.array = np.array(action_space[action_index])

        next_obs, reward, terminated, truncated, _ = environment.step(action)
        next_state = state_representation.Observation_To_State(next_obs)

        self.Update(current_state, next_state, action_index, reward)

        return action_index, next_state, reward, terminated, truncated

    def Update(self, current_state : tuple, next_state : float, action_index : int, reward : float):
        self.updater.Update(self.state_representation, current_state, next_state, action_index, reward)
        self.strategy.Update()


