from State_Representations import State_Representation
import Strategies
import Updaters
import numpy as np

class RL_Algorithm:
    def __init__(self, ID : int = 0):
        self.strategy : Strategies.Exploration_Strategy
        self.updater : Updaters.Updater
        self.ID = ID
        self.state_representation : State_Representation = None

    def Step(self, action_space : np.array, current_state : tuple, environment):
        action_index : int = self.updater.Select_Action_Index(state_representation=self.state_representation, current_state=current_state, strategy=self.strategy)
        action : np.array = np.array(action_space[action_index])

        next_obs, reward, terminated, truncated, _ = environment.step([action.item()])
        next_state = self.state_representation.Observation_To_State(next_obs)

        self.Update(current_state, next_state, action_index, reward)

        return action_index, next_state, reward, terminated, truncated

    def Update(self, current_state : tuple, next_state : float, action_index : int, reward : float):
        self.updater.Update(strategy=self.strategy,
                            state_representation=self.state_representation,
                            current_state=current_state,
                            next_state=next_state,
                            action_idx=action_index,
                            reward=reward)

    def Episode_Ended(self):
        self.strategy.Episode_Ended()
        self.updater.Episode_Ended()

    def Get_Strategy(self) -> Strategies.Exploration_Strategy:
        return self.strategy

    def Get_Updater(self) -> Updaters.Updater:
        return self.updater

    def Set_State_Representation(self, state_repr : State_Representation) -> None:
        self.state_representation = state_repr

    def Get_State_Representation(self) -> State_Representation:
        return self.state_representation

    def Get_ID(self) -> int:
        return self.ID

    def __str__(self):
        return str(self.updater) + " | " + str(self.strategy)