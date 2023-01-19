import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


if __name__ == "__main__":
    render_options = {}
    options = {}

    render_options["headless"] = True
    
    options["env_name"] = "Wipe"
    options["robots"] = "LRmate"
    
    controller_name = "OSC_ADM"

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
    )
    env.reset()

    if not render_options["headless"]:
        env.viewer.set_camera(camera_id=0)

    action_dim = env.robots[0].action_dim + env.robots[0].gripper.dof

    print('Begin to move down.')

    while abs(env.robots[0].ee_force[-1]) <= 20:
        # Push downwards
        action = 10 * np.array([0, 0, -0.1, 0, 0, 0])
        env.step(action)
        if not render_options["headless"]:
            env.render()
        print('End Effector Force: ', env.robots[0].ee_force[-1])

    input('Begin to clean the table')
    
    while True:
        action = np.zeros((6,))
        random_action = np.random.uniform(-0.5,0.5,(2,))
        action[0:2] =random_action
        env.step(action)
        if not render_options["headless"]:
            env.render()
        print('End Effector Force: ', env.robots[0].ee_force[-1])
