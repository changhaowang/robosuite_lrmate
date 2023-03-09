import numpy as np
from ubuntu_controller import robot_controller
import torch
import copy
from rlkit.torch.pytorch_util import set_gpu_mode
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T

INIT_JNT_POSE_SIM = np.array([ 0.06569796, 0.77993625, -0.63284764, 1.47314256, -1.63438477, 1.47628377])
INIT_EEF_POSE_SIM = np.array([-0.1009748, -0.14462975, 1.05970618, -0.0390827, -0.08734996, 1.49746034])

DEFAULT_KP = np.array([25, 25, 90, 20, 20, 20])
DEFAULT_KD = np.array([17, 17, 90, 20, 20, 20])
DEFAULT_M = np.array([2, 2, 2])
DEFAULT_INTERTIA = np.array([0.2, 0.2, 0.2])

class RL_Agent(object):
    '''
    Execute policy learned in simulation on the LRmate 200id in MSC Lab
    '''
    def __init__(self, env_name, controller_type, policy_folder, M=DEFAULT_M, Inertia=DEFAULT_INTERTIA, override_impedance_command=False, fix_orientation=False, gpu=True, print_args=False) -> None:   
        '''
            Args: 
                1. env_name: environment name
                2. controller_type: type of the controller
                3. policy_folder: folder location of the policy
                4. M: 3*1, robot mass in Cartesian space
                5. Inertia: 3*1 robot inertia in Cartesian space
                6. override_impedance_command: bool. Whether force to use the default impedance command instead of the learnt command
                7. fix_orientation: bool. Whether to fix the orientation of the robot
                8. gpu: whether use gpu
                9. print_args: whether to print information
        '''
        self.controller = robot_controller()
        self.env_name = env_name
        self.controller_type = controller_type
        self.policy_folder = policy_folder
        self.override_impedance_command = override_impedance_command
        self.fix_orientation = fix_orientation
        self.gpu = gpu
        self.print_args = print_args

        # Load policy
        self.load_policy()

        # Impedance parameters
        self.control_dim = 6
        self.Mass = M
        self.Inertia = Inertia
        self.ori_offset = np.eye(3)
        self.pos_offset = np.array([-0.56, 0 , 1.142]) # real -> simulation
        self.kp_limit = np.array([20, 100])

        # Simulation related parameters
        self.input_max = np.ones((self.control_dim, ))
        self.input_min = -1 * np.ones((self.control_dim, ))
        self.output_max = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.output_min = np.array([-0.05, -0.05, -0.05, -0.05, -0.05, -0.05])

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

    def init_robot_pos(self, eef_pose=INIT_EEF_POSE_SIM, sim_frame=True):
        '''
        Initialize the robot pose
        Args:
            1. eef_pose: 6*1 ndarray. End effector pose (position + euler ('ZYX' in radians))
            2. sim_frame: bool. Whether to use the frame in simulation or the real world
        '''
        eef_pose_real = eef_pose
        if sim_frame:
            eef_pose_real[0:3] = eef_pose[0:3] - self.pos_offset
        if self.print_args:
            print('Initial EEF Pose in world frame: ', eef_pose_real)
            input('Begin to move the robot')
        TCP_d_POSE = eef_pose_real            
        self.send_commend(TCP_d_pos=TCP_d_POSE[0:3], TCP_d_euler=TCP_d_POSE[3:])

    def load_policy(self):
        '''
        Load policy from the trained parameters
        '''
        self.data = torch.load(self.policy_folder)
        self.policy = self.data['evaluation/policy']
        if self.print_args:
            print('Policy Loaded.')
        if self.gpu:
            set_gpu_mode(True)
            self.policy.cuda()
            if self.print_args:
                print('Set gpu mode: True')    
        self.policy.reset()     

    def get_robot_state(self, sim_frame=True):
        '''
        Get robot state from the ubuntu controller.
        Args:
            1. sim_frame: bool. Get the robot state either in the world frame of the simulation or the real world
                Since the real robot and the simulation use different global coordinate, there is a offset.
                We specifically distinguish the observation to be used in real and simulation
                    a. Door Env for sim: robot_eef_pos, robot_eef_quat
                    b. Door Env for real: Tcp_pos, TCP_euler ('ZYX' in radians)
        '''
        self.controller.receive()
        # Decode Robot Information
        TCP_pos_real = self.controller.robot_pose[0:3]
        TCP_rotm_real = self.controller.robot_pose[3:12].reshape([3,3]).T
        TCP_euler_real = R.from_matrix(TCP_rotm_real).as_euler('ZYX', degrees=False)
        TCP_vel_real = self.controller.robot_vel
        TCP_wrench_real = self.controller.TCP_wrench
        robot_joint_pos_real = self.controller.joint_pos
        # Apply transformation to get equvailent robot information in the simulation coordinate
        TCP_pos_sim = TCP_pos_real + self.pos_offset
        TCP_rotm_sim = TCP_rotm_real
        TCP_quat_sim = R.from_matrix(TCP_rotm_sim).as_quat()
        robot_joint_pos_sim = robot_joint_pos_real
        if sim_frame:
            if self.env_name == 'Door':
                state = np.hstack((TCP_pos_sim, TCP_quat_sim))
        else:
            if self.env_name == 'Door':
                state = np.hstack((TCP_pos_real, TCP_euler_real))
        return state

    def get_robot_rotation_matrix(self):
        '''
        Get robot end-effector rotation matrix.
        Returns:
            1. TCP_rotm: 3*3 ndarray. End-effector orientation matrix
        '''
        self.controller.receive()
        # Decode robot information
        TCP_rotm = self.controller.robot_pose[3:12].reshape([3,3]).T
        return TCP_rotm

    def get_TCP_wrench(self):
        '''
        Get robot end-effector wrench
        Returns:
            1. TCP_wrench: 6*1 ndarray. End-effector wrench
        '''
        self.controller.receive()
        # Decode Robot Information
        TCP_wrench_real = self.controller.TCP_wrench
        return TCP_wrench_real
        
    def set_object_state(self, object_state):
        '''
        Set required object state
            1. Door Env: door_pos, handle_pos, hinge_qpos
        '''
        if self.env_name == 'Door':
            self.door_pos_real = object_state[0:3]
            self.handle_pos_real = object_state[3:6]
            self.hinge_qpos_real = object_state[6]

    def update_hinge_qpos(self):
        '''
        Real-time track the object state. To handle the movement of the object.
        '''
        TCP_wrench = self.get_TCP_wrench()
        if np.linalg.norm(TCP_wrench) >= 2:
            self.hinge_qpos_real += 0.01
        

    def get_object_state(self, sim_frame=True):
        '''
        Get required object state if needed.
        Args:
            1. sim_frame: bool. Whether to output object state in simulation or real world frame
        Returns:
            1. object state: ndarray
                1. Door Env: door_pos, handle_pos, door_to_eef_pos, handle_to_eef_pos, hinge_qpos
        '''
        # self.update_hinge_qpos()
        if sim_frame:
            if self.env_name == 'Door':
                door_pos_sim = self.door_pos_real + self.pos_offset
                handle_pos_sim = self.handle_pos_real + self.pos_offset
                hinge_qpos_sim = self.hinge_qpos_real
                robot_eef_pose = self.get_robot_state(sim_frame=True)
                robot_eef_pos = robot_eef_pose[0:3]
                door_to_eef_pos = door_pos_sim - robot_eef_pos
                handle_to_eef_pos = handle_pos_sim - robot_eef_pos
                return np.hstack((door_pos_sim, handle_pos_sim, door_to_eef_pos, handle_to_eef_pos, hinge_qpos_sim))
        else:
            return np.hstack((self.door_pos_real, self.handle_pos_real, self.hinge_qpos_real))
        
    def get_total_observations(self, sim_frame=True):
        '''
        Get the required observations (door pos, handle pos, robot joint pos)
        Args:
            1. sim_frame: bool. Whether to output object state in simulation or real world frame
        Returns:
            1. observations: ndarray. 
                a. Door Env. Containts object state and robot state
        '''
        robot_state = self.get_robot_state(sim_frame=sim_frame)
        object_state = self.get_object_state(sim_frame=sim_frame)
        observations = np.hstack((object_state, robot_state))
        return observations

    def get_goal_orientation(self, delta_tcp_euler):
        '''
        Get goal orientation in euler.
        Args:
            1. delta_tcp_euler: 3*1 ndarray. delta tcp euler in ZYX order in radians
        Returns:
            1. TCP_d_euler: 3*1 ndarray. Desried tcp euler angle in ZYX order in radians
        '''
        delta_rotm = R.from_euler('ZYX', delta_tcp_euler, degrees=False).as_matrix()
        TCP_rotm = self.get_robot_rotation_matrix()
        TCP_d_rotm = delta_rotm @ TCP_rotm
        TCP_d_euler = R.from_matrix(TCP_d_rotm).as_euler('ZYX', degrees=False)
        return TCP_d_euler

    def action_transform(self, action_sim):
        '''
        Transform the action in robosuite to the same frame in the real world
        Args: 
            action_sim [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp axisangle + gripper_command
        Returns:
            action_real [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler (in ZYX order) + gripper_command
        '''
        kp_sim = action_sim[0:6]
        delta_TCP_d_pos_sim = action_sim[6:9]
        delta_TCP_d_axisangle_sim = action_sim[9:12]
        delta_TCP_d_quat_sim = T.axisangle2quat(delta_TCP_d_axisangle_sim)
        gripper_action_sim = action_sim[12]

        action_real = np.zeros_like(action_sim)
        kp_real = kp_sim # may need to transform the orientation Kp
        delta_TCP_d_pos_real = delta_TCP_d_pos_sim
        delta_TCP_d_euler_real = (R.from_quat(delta_TCP_d_quat_sim)).as_euler('ZYX', degrees=False)
        gripper_action_real = gripper_action_sim

        action_real[0:6] = kp_real
        action_real[6:9] = delta_TCP_d_pos_real
        action_real[9:12] = delta_TCP_d_euler_real
        action_real[12] = gripper_action_real
        return action_real

    def scale_action(self, action):
        """
        Function taken from the robosuite.

        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """
        a = copy.deepcopy(action)
        eef_action = a[6:12]
        action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        action_output_transform = (self.output_max + self.output_min) / 2.0
        action_input_transform = (self.input_max + self.input_min) / 2.0
        scaled_eef_action = np.clip(eef_action, self.input_min, self.input_max)
        transformed_eef_action = (scaled_eef_action - action_input_transform) * action_scale + action_output_transform
        a[6:12] = transformed_eef_action
        return a

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

        observations = self.get_total_observations(sim_frame=True)
        o_for_agent = preprocess_obs_for_policy_fn(observations)
        action_sim, _ = self.policy.get_action(o_for_agent, **get_action_kwargs)
        scaled_action_sim = self.scale_action(action_sim)
        action_real = self.action_transform(scaled_action_sim)
        if self.controller_type == 'OSC_POSE':
            self.operation_space_control(action_real)
    
    def operation_space_control(self, action):
        '''
        Operational space control for robot
        Notice:
            1. The implementation of admittance control in simulink is different from the controller in mujoco
        Args: 
            action [13 * 1]: stiffness + (delta) tcp pos + (delta) tcp euler (in ZYX order) + gripper_command
        '''
        # update robot state
        robot_state = self.get_robot_state(sim_frame=False)
        delta_tcp_pos = action[6:9]
        if self.print_args:
            print('Delta positional action: ', delta_tcp_pos)
        delta_tcp_euler = action[9:12]
        TCP_d_pos = robot_state[0:3] + delta_tcp_pos
        if self.fix_orientation:
            TCP_d_euler = robot_state[3:6]
        else:
            TCP_d_euler = self.get_goal_orientation(delta_tcp_euler)

        if self.override_impedance_command:
            Kp = DEFAULT_KP
            Kd = DEFAULT_KD
        else:
            Kp = np.clip(action[0:6], self.kp_limit[0], self.kp_limit[1])
            if self.print_args:
                print('Kp: ', Kp)
            # use critical damping
            Kd = 2 * np.sqrt(Kp)
            # Kd = DEFAULT_KD
            
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
        if self.print_args:
            print('UDP command: ', UDP_cmd)
        self.controller.send(UDP_cmd)    