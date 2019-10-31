from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, TensorFlowState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
    def act(self, ob, create_obstacles=False):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, create_obstacles=create_obstacles)
        return action
    
    def orca_act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = TensorFlowState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
