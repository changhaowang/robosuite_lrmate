import robosuite as suite
from robosuite.utils.input_utils import *
from scipy.spatial.transform import Rotation as R

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

    while True:
        action = np.zeros((7,))
        # give orientational action
        delta_euler_ZYX = np.array([0,0,np.pi/6])
        delta_euler_YZX = R.from_euler('ZYX', delta_euler_ZYX, degrees=False).as_euler('YZX', degrees=False)
        action[3:6] = delta_euler_YZX
        observations, reward, done, info = env.step(action)
        if not render_options["headless"]:
            env.render()
        print('One step finished.')