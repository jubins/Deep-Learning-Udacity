import numpy as np
from physics_sim import PhysicsSim
from task import Task
import math

class Land_Task(Task):
    """Simple task where the goal is land the quadcopter flat on the ground plane, elevation Z = 0"""

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        penalty = (1. - abs(math.sin(self.sim.pose[3])))
        penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        penalty *= (1. - abs(math.sin(self.sim.pose[5])))

        delta = abs(self.sim.pose[:3] - self.target_pos)
        r = math.sqrt(np.dot(delta, delta))
        
        if(r > 0.01): decay = math.exp(-1/r) # Give range -1 to 1
        else: decay = 0
        reward = 1. - decay
        reward *= penalty
        return reward
    
    def getPose(self):
        return self.sim.pose
