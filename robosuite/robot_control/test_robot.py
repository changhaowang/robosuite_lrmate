import numpy as np
from robosuite.robot_control.RL_agent import RL_Agent

if __name__ == "__main__":
    agent = RL_Agent(env_name='Door', controller_type='OSC_POSE', policy_folder='rlkit/data/DoorLRmateOSC-POSEvariable-kp/DoorLRmateOSC_POSEvariable_kp_2023_03_07_16_51_16_0000--s-0/params.pkl',override_impedance_command=True, fix_orientation=False, print_args=True)
    agent.set_object_state(np.array([0.52, -0.33, 0.05, 0.52, -0.33, 0.05, 0]))
    object_state = agent.get_object_state(sim=True)
    print(object_state)
    init_robot_state = np.array([0.47, -0.12, 0.0, 0.0, -0.0, 1.57])
    agent.init_robot_pos(eef_pose=init_robot_state)

    for i in range(500):
        agent.rollout()