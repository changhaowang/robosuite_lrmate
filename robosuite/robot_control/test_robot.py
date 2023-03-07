import numpy as np
from ubuntu_controller import robot_controller
import torch
from scipy.spatial.transform import Rotation as R
from robosuite.robot_control.RL_agent import RL_Agent

if __name__ == "__main__":
    agent = RL_Agent(env_name='Door', controller_type='OSC_POSE', policy_folder='data/DoorLRmateOSC_POSEvariable_kp_2023_02_23_22_50_18_0000--s-0/params.pkl')
    robot_state = agent.get_robot_state(sim=False)
    TCP_pos = robot_state[0:3]
    TCP_euler = robot_state[3:6]
    TCP_d_pos = TCP_pos
    TCP_d_euler = TCP_euler + np.array([0, 0, -1/3 * np.pi])
    agent.send_commend(TCP_d_pos=TCP_d_pos, TCP_d_euler=TCP_d_euler)
    print('Finished')

