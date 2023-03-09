import time
import numpy as np
from robosuite.robot_control.RL_agent import RL_Agent

INIT_EEF_POSE_SIM = np.array([0.0009748, -0.14462975, 1.15970618, 0, 0, 1.57])

if __name__ == "__main__":
    agent = RL_Agent(env_name='Door', controller_type='OSC_POSE', policy_folder='rlkit/data/DoorLRmateOSC-POSEvariable-kp/DoorLRmateOSC_POSEvariable_kp_2023_03_07_16_51_16_0000--s-0/params.pkl',override_impedance_command=False, fix_orientation=False, print_args=True)
    agent.set_object_state(np.array([0.52, -0.33, 0.01, 0.52, -0.37, 0.01, 0]))
    agent.init_robot_pos(eef_pose=INIT_EEF_POSE_SIM, sim_frame=True)
    time.sleep(2)

    for i in range(500):
        agent.rollout()
        time.sleep(1)