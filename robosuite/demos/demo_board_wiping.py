import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


if __name__ == "__main__":
    render_options = {}
    options = {}

    render_options["headless"] = False
    
    options["env_name"] = "Wipe"
    options["robots"] = "LRmate"
    
    controller_name = "OSC_POSE"

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
    # Update admittance parameter
    # options["controller_configs"]["I_admittance"] = [50000, 50000, 50000],
    # options["controller_configs"]["kd_admittance"] = [0, 0, 0, 0, 0, 0]
    # options["controller_configs"]["kp_admittance"] = [1000000, 1000000, 4, 10000, 10000, 100000]
    options["controller_configs"]["control_delta"] = True
    # options["controller_configs"]["control_ori"] = False

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=not render_options["headless"],
        has_offscreen_renderer=render_options["headless"],
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()

    if not render_options["headless"]:
        env.viewer.set_camera(camera_id=0)

    action_dim = env.robots[0].action_dim + env.robots[0].gripper.dof
    desired_ee_pos = np.array([0.065, -0.012, 1.5])
    desired_ee_ori_mat = env.robots[0].controller.ee_ori_mat
    desired_ee_ori_euler = T.mat2euler(desired_ee_ori_mat)#np.array([-np.pi, 0, np.pi/2])

    while True:
        action = np.zeros((6,))
        # Push downwards
        action[:3] = desired_ee_pos
        action[3:6] = desired_ee_ori_euler
        
        env.step(action)
        if not render_options["headless"]:
            env.render()
        sensed_force = env.robots[0].controller.ee_force
        # calibrated_force = env.robots[0].controller.calibrate_force_sensor_measurement(sensed_force)
        # # print('Sensed Calibrated Effector Force: ', calibrated_force[-1])
        # qpos = env.robots[0]._joint_positions
        # qvel = env.robots[0]._joint_velocities
        # ee_pos = env.robots[0].controller.ee_pos
        # ee_vel = env.robots[0].controller.ee_pos_vel
        # # print("Robot Joint Position: ", qpos)
        # # print("Robot Joint Velocity: ", qvel)
        # print("Robot End-Effector Position: ", ee_pos)
        # # print("Robot End-Effector Velocity: ", ee_vel)
        # sensed_force = env.robots[0].controller.ee_force
        # calibrated_force = env.robots[0].controller.calibrate_force_sensor_measurement(sensed_force)
        # print('Sensed Calibrated Effector Force: ', calibrated_force)
        # print("Robot Desired Position: ", env.robots[0].controller.ee_pos_desired_admittance)
        # print("Robot Addmittance Error(X_d - X_r): ", env.robots[0].controller.ee_pos_desired_admittance - env.robots[0].controller.reference_pos)


    input('Begin to clean the table')
    
    while True:
        action = np.array([0, 0, -0.01, 0, 0, 0])
        random_action = np.random.uniform(-0.1,0.1,(2,))
        action[0:2] =random_action
        env.step(action)
        if not render_options["headless"]:
            env.render()
        sensed_force = env.robots[0].controller.ee_force
        calibrated_force = env.robots[0].controller.calibrate_force_sensor_measurement(sensed_force)
        print('Sensed Calibrated Effector Force: ', calibrated_force)
        # print("Robot Addmittance Error(X_d - X_r): ", env.robots[0].controller.ee_pos_desired_admittance - env.robots[0].controller.reference_pos)
        print('Pos Error: ', env.robots[0].controller.pos_err)
        # print('finished one step')