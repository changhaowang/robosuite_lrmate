import numpy as np
from ubuntu_controller import robot_controller
import gym
import torch
from rlkit.torch.pytorch_util import set_gpu_mode

class RL_Agent(object):
    '''
    RL_Agent learned from the simulation
    '''
    def __init__(self, env_name, policy_file, gpu) -> None:   
        self.controller = robot_controller()
        self.env_name = env_name
        self.policy_file = policy_file
        self.gpu = gpu

        # Load policy
        self.load_policy()
    
    def load_agent(self):
        '''
        Load policy from the trained parameters
        '''
        self.data = torch.load(self.policy_file)
        self.policy = self.data['evaluation/policy']
        print('Policy Loaded.')
        if self.gpu:
            set_gpu_mode(True)
            self.policy.cuda()    
        self.policy.reset()     

    def get_robot_state(self):
        '''
        Get robot state from the ubuntu controller
        '''
        self.controller.receive()
        TCP_pos = self.controller.robot_pose[0:3]
        TCP_rotm = self.controller.robot_pose[3:12].reshape([3,3]).T
        TCP_rotm = TCP_rotm @ self.offset.T
        TCP_euler = R.from_matrix(TCP_rotm).as_euler('xyz')
        TCP_vel = self.controller.robot_vel
        TCP_wrench = self.controller.TCP_wrench
        World_force = TCP_rotm @ TCP_wrench[0:3]
        World_torque = TCP_rotm @ TCP_wrench[3:6]
        state = np.hstack([TCP_pos,TCP_euler,TCP_vel,World_force,World_torque])
        state[0:3] = state[0:3] - self.goal_pose
        # keep in mind, the unit of state in simulation is cm. However, the real robot uses m.
        state[0:3] = state[0:3] * 100
        state[6:9] = state[6:9] * 100
        return state

    def get_object_state(self):
        '''
        Get required object state if needed
        '''
        pass

    def get_total_observations(self):
        '''
        Get the required observations (robot state + object state)
        '''
        pass
    
    def rollout(self, 
                preprocess_obs_for_policy_fn=None, 
                get_action_kwargs=None):
        '''
        Rollout one step using the learned policy
        '''
        if get_action_kwargs is None:
            get_action_kwargs = {}
        if preprocess_obs_for_policy_fn is None:
            preprocess_obs_for_policy_fn = lambda x: x

        self.get_total_observations()
        o_for_agent = preprocess_obs_for_policy_fn(self.obs)
        action, agent_info = self.policy.get_action(o_for_agent, **get_action_kwargs)




if __name__ == "__main__":