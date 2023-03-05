import numpy as np
from ubuntu_controller import robot_controller
import torch
from rlkit.torch.pytorch_util import set_gpu_mode
from scipy.spatial.transform import Rotation as R

INIT_JNT_POSE =  np.array([-0.03477543, 0.89188467, -0.57513507, 0.08426756, -0.11839037, 0.08335862])
INIT_EEF_POSE = np.array([0.47, 0, 0.1, 5, 0, 0])

DEFAULT_KP = np.array([25, 25, 90, 20, 20, 20])
DEFAULT_KD = np.array([17, 17, 90, 20, 20, 20])
DEFAULT_M = np.array([2, 2, 2])
DEFAULT_INTERTIA = np.array([0.2, 0.2, 0.2])

class RL_Agent(object):
    '''
    RL_Agent learned from the simulation
    '''
    def __init__(self, env_name, controller_type, policy_folder, M=DEFAULT_M, Inertia=DEFAULT_INTERTIA, gpu=True, print_args=False) -> None:   
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
        self.Mass = M
        self.Inertia = Inertia
        self.ori_offset = (np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])).T @ np.array([[0, 0, 1],[1, 0, 0], [0, 1, 0]])  # Convert the frame to gripper frame
        self.pos_offset = np.array([-0.574, 0 , 1.202])
        self.kp_limit = np.array([0, 300])

        # Init Robot Position
        self.init_robot_pos()
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
        Get the required observations (door pos, handle pos, robot joint pos)
        '''
        robot_state = self.get_robot_state()
        object_state = self.get_object_state()
        observations = np.hstack((object_state, robot_state))
        return observations
    
    def operation_space_control(self, action):
        '''
        Operational space control for robot
        Args: 
            action [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler (in YZX order) + gripper_command
        '''
        # update robot state
        self.get_robot_state()
        delta_tcp_pos = action[6:9]
        delta_tcp_euler = action[9:12]
        TCP_d_pos = self.TCP_pose[0:3] + delta_tcp_pos
        TCP_d_euler = self.TCP_pose[3:6] + delta_tcp_euler

        Kp = np.clip(action[0:6], self.kp_limit[0], self.kp_limit[1])
        # use critical damping
        Kd = 2 * np.sqrt(Kp)
        self.send_commend(TCP_d_pos, TCP_d_euler, Kp, Kd)

    def send_commend(self, TCP_d_pos, TCP_d_euler, Kp=DEFAULT_KP, Kd=DEFAULT_KD):
        '''
        Send command to the robot.
        Args:
            TCP_d_pos: 3*1 end-effector position
            TCP_d_euler: 3*1 end-effector orientation ('ZYX' in radians)
            TCP_d_vel: 6*1 end-effector velocity
        '''

        UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, Kp, Kd, self.Mass, self.Inertia])
        print(UDP_cmd)
        self.controller.send(UDP_cmd)    

    def action_transform(self, action_sim):
        '''
        Transform the action in simulation to the same frame in the real world
        Args: 
            action_sim [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler (in YZX order) + gripper_command
        Returns:
            action_real [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler (in ZYX order) + gripper_command
        '''
        kp_sim = action_sim[0:6]
        delta_TCP_d_pos_sim = action_sim[6:9]
        delta_TCP_d_euler_sim = action_sim[9:12]
        gripper_action_sim = action_sim[12]

        action_real = np.zeros_like(action_sim)
        kp_real = kp_sim # may need to transform the orientation Kp
        delta_TCP_d_pos_real = delta_TCP_d_pos_sim
        delta_TCP_d_euler_real = (R.from_euler('YZX', delta_TCP_d_euler_sim, degrees=False)).as_euler('YZX', degress=False)
        gripper_action_real = gripper_action_sim

        action_real[0:6] = kp_real
        action_real[6:9] = delta_TCP_d_pos_real
        action_real[9:12] = delta_TCP_d_euler_real
        action_real[12] = gripper_action_real
        return action_real

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
        action_sim, _ = self.policy.get_action(o_for_agent, **get_action_kwargs)
        action_real = self.action_transform(action_sim)
        if self.controller_type == 'OSC_POSE':
            self.operation_space_control(action_real)

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