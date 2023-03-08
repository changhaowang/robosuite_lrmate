import numpy as np
from ubuntu_controller import robot_controller
import torch
from scipy.spatial.transform import Rotation as R
from robosuite.robot_control.RL_agent import RL_Agent

if __name__ == "__main__":
    agent = RL_Agent(env_name='Door', controller_type='OSC_POSE', policy_folder='data/03_05/DoorLRmateOSC-POSEvariable-kp/DoorLRmateOSC_POSEvariable_kp_2023_03_05_16_28_50_0000--s-0/params.pkl',override_impedance_command=True, fix_orientation=True, print_args=True)
    agent.set_object_state(np.array([0.4, -0.2, 0.18, 0.45, -0.2, 0.18, 0]))
    robot_state = agent.get_robot_state(sim=False)
    object_state = agent.get_object_state(sim=True)
    robot_state[3:6] = np.array([0.0, -0.0, -1.57])
    # agent.init_robot_pos(eef_pose=robot_state)

    for i in range(10):
        agent.rollout()
    print('Finished')

