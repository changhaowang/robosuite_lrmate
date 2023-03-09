import robosuite as suite
from robosuite.utils.input_utils import *
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T

if __name__ == "__main__":
    render_options = {}
    options = {}

    render_options["headless"] = False
    
    options["env_name"] = "Door"
    options["robots"] = "LRmate"
    
    controller_name = "OSC_POSE"

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=not render_options["headless"],
        has_offscreen_renderer=render_options["headless"],
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True,
    )
    env.reset()

    if not render_options["headless"]:
        env.viewer.set_camera(camera_id=3)

    init_obs = env._get_observations()
    init_eef_pos = init_obs['robot0_eef_pos']
    init_eef_quat = init_obs['robot0_eef_quat']
    init_eef_euler_ZYX = R.from_quat(init_eef_quat).as_euler('ZYX', degrees=False)

    while True:
        observations = env._get_observations()
        action = np.zeros((7,))
        # give orientational action
        delta_euler_ZYX = np.array([0,0,np.pi/6])
        delta_quat = R.from_euler('ZYX', delta_euler_ZYX, degrees=False).as_quat()
        delta_axis_angle = T.quat2axisangle(delta_quat)
        action[3:6] = delta_axis_angle
        observations, reward, done, info = env.step(action)
        if not render_options["headless"]:
            env.render()
        print('One step finished.')