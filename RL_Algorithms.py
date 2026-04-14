import State_Representations
import Strategies
import Updaters
import numpy as np

class RL_Algorithm:
    def __init__(self):
        self.strategy : Strategies.Exploration_Strategy
        self.updater : Updaters.Updater
        self.state_representation : State_Representations.State_Representation

    def Step(self, action_space : np.array, environment, current_state : tuple, ):
        action_index : int = self.strategy.Get_Action()
        action : np.array = np.array(action_space[action_index])

        next_obs, reward, terminated, truncated, _ = environment.step(action)
        next_state = self.state_representation.Discretize_Observation(next_obs)

        self.Update(current_state, next_state, action_index, reward)

        return action_index, next_state, reward, terminated, truncated

    def Update(self, current_state : tuple, next_state : float, action_index : int, reward : float):
        self.updater.Update(self.state_representation, current_state, next_state, action_index, reward)


