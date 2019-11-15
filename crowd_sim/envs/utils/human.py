from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        
    def act(self, ob, create_obstacles=False, obstacles=None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, create_obstacles=create_obstacles, obstacles=obstacles)
        return action
