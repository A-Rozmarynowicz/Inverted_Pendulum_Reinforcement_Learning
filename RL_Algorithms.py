import State_Representations
import Strategies
import Updaters
import numpy as np

class RL_Algorithm:
    def __init__(self):
        self.strategy : Strategies.Exploration_Strategy
        self.updater : Updaters.Updater
        self.state_representation : State_Representations.State_Representation

    def Step(self, action_space : np.array, environment):
        action_index : int = self.strategy.Get_Action()
        action : np.array = np.array(action_space[action_index])

        next_obs, reward, terminated, truncated, _ = environment.step(action)
