import numpy as np
from ubuntu_controller import robot_controller
import torch
from rlkit.torch.pytorch_util import set_gpu_mode
from scipy.spatial.transform import Rotation as R

INIT_JNT_POSE =  np.array([-0.03477543, 0.89188467, -0.57513507, 0.08426756, -0.11839037, 0.08335862])
INIT_EEF_POSE = np.array([0.47, 0, 0.1, 0, 0, 0])

DEFAULT_KP = np.array([25, 25, 90])
DEFAULT_KD = np.array([17, 17, 90])


class RL_Agent(object):
    '''
    RL_Agent learned from the simulation
    '''
    def __init__(self, env_name, controller_type, policy_folder, gpu=True, print_args=False) -> None:   
        '''
            Args: 
                1. env_name: environment name
                2. controller_type: type of the controller
                3. policy_folder: folder location of the policy
                4. gpu: whether use gpu
                5. print_args: whether to print information
        '''
        self.controller = robot_controller()
        self.env_name = env_name
        self.controller_type = controller_type
        self.policy_folder = policy_folder
        self.gpu = gpu
        self.print_args = print_args

        # Load policy
        self.load_policy()

        # Impedance parameters
        self.Mass = np.array([2,2,2])    # to determine
        self.Inertia = 1*np.array([2, 2, 2])   # to determine
        self.ori_offset = (np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])).T @ np.array([[0, 0, 1],[1, 0, 0], [0, 1, 0]])  # Convert the frame to gripper frame
        self.pos_offset = np.array([-0.574, 0 , 1.202])

        # Init Robot Position
        # self.init_robot_pos()
        if print_args:
            print('Environment Initialized.')

    def test_robot_connection(self):
        '''
        Test the connection between the ubuntu computer and the host computer
        '''
        self.controller.receive()
        if self.print_args:
            print('Success')

    def init_robot_pos(self, eef_pose=INIT_EEF_POSE):
        TCP_d_POSE = eef_pose
        input('Begin to move the robot')
        self.send_commend(TCP_d_pos=TCP_d_POSE[0:3], TCP_d_euler=TCP_d_POSE[3:])

    def load_policy(self):
        '''
        Load policy from the trained parameters
        '''
        self.data = torch.load(self.policy_folder)
        self.policy = self.data['evaluation/policy']
        print('Policy Loaded.')
        if self.gpu:
            set_gpu_mode(True)
            self.policy.cuda()    
        self.policy.reset()     

    def get_robot_state(self):
        '''
        Get robot state from the ubuntu controller
            1. Door Env: robot_joint_pos
        '''
        self.controller.receive()
        # Decode Robot Information
        TCP_pos = self.controller.robot_pose[0:3] + self.pos_offset
        TCP_rotm_temp = self.controller.robot_pose[3:12].reshape([3,3]).T
        TCP_rotm = TCP_rotm_temp @ self.ori_offset
        TCP_euler = R.from_matrix(TCP_rotm).as_euler('xyz', degrees=True)
        self.TCP_pose = np.hstack((TCP_pos, TCP_euler))
        self.TCP_vel = self.controller.robot_vel
        # Obtain force/torque information
        TCP_wrench = self.controller.TCP_wrench
        self.World_force = TCP_rotm @ TCP_wrench[0:3]
        self.World_torque = TCP_rotm @ TCP_wrench[3:6]
        robot_joint_pos = self.controller.joint_pos
        if self.env_name == 'Door':
            state = robot_joint_pos
            return state

    def set_object_state(self, object_state):
        '''
        Set required object state
            1. Door Env: door_pos, handle_pos
        '''
        if self.env_name == 'Door':
            self.door_pos = object_state[0:3]
            self.handle_pos = object_state[3:]

    def get_object_state(self):
        '''
        Get required object state if needed.
            1. Door Env: door_pos, handle_pos
        '''
        if self.env_name == 'Door':
            return np.hstack((self.door_pos, self.handle_pos))

    def get_total_observations(self):
        '''
        Get the required observations (robot state + object state)
        '''
        robot_state = self.get_robot_state()
        object_state = self.get_object_state()
        observations = np.hstack((robot_state, object_state))
        return observations
    
    def operation_space_control(self, action):
        '''
        Operational space control for robot
        Args: 
            action [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler + gripper_command
        '''
        # update robot state
        self.get_robot_state()
        TCP_d_pose = action[6:12] + self.TCP_pose
        TCP_d_pos = TCP_d_pose[0:3]
        TCP_d_euler = TCP_d_pose[3:]
        Kp = action[0:6]
        # use critical damping
        Kd = 2 * np.sqrt(Kp)
        self.send_commend(TCP_d_pos, TCP_d_euler, Kp, Kd)

    def send_commend(self, TCP_d_pos, TCP_d_euler, Kp=DEFAULT_KP, Kd=DEFAULT_KD):
        '''
        Send command to the robot.
        Args:
            TCP_d_pos: 3*1 end-effector position
            TCP_d_euler: 3*1 end-effector orientation
            TCP_d_vel: 6*1 end-effector velocity
        '''

        d_rotm = R.from_euler('xyz', TCP_d_euler).as_matrix()
        TCP_d_euler = R.from_matrix(d_rotm).as_euler('xyz')
        UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, Kp[0:3], Kd[0:3], self.Mass[0:3]])
        print(UDP_cmd)
        self.controller.send(UDP_cmd)    

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

        observations = self.get_total_observations()
        o_for_agent = preprocess_obs_for_policy_fn(observations)
        action, _ = self.policy.get_action(o_for_agent, **get_action_kwargs)
        if self.controller_type == 'OSC_POSE':
            self.operation_space_control(action)

    def impedance_adapt(self, action):
        '''
        Sim-to-real transfer of the impedance parameters
            Args: action [12 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler
        '''
        if self.env_name == 'Door':
            Kp = action[0:6]
            Kd = 2 * np.sqrt(Kp)
            pass

if __name__ == "__main__":
    agent = RL_Agent(env_name='Door', controller_type='OSC_POSE', policy_folder='data/DoorLRmateOSC_POSEvariable_kp_2023_02_23_22_50_18_0000--s-0/params.pkl')
    agent.test_robot_connection()
    robot_state = agent.get_robot_state()
    agent.set_object_state([0.4,0.1,0.1, 0.4,0.1,0.1])
    for i in range(10):
        agent.rollout()
        print('robot state: ', robot_state)