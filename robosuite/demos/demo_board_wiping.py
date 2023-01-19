import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


if __name__ == "__main__":
    options = {}
    options["env_name"] = "Wipe"
    options["robots"] = "LRmate"
    controller_name = "OSC_ADM"

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    action_dim = env.robots[0].action_dim + env.robots[0].gripper.dof

    print('Begin to move down.')

    while abs(env.robots[0].ee_force[-1]) <= 20:
        # Push downwards
        action = 10 * np.array([0, 0, -0.1, 0, 0, 0])
        env.step(action)
        env.render()

    print('Begin to clean the table')
    
    while True:
        action = np.zeros((6,))
        # random_action = np.random.uniform(-0.5,0.5,(2,))
        # action[0:2] =random_action
        env.step(action)
        env.render()
        print('End Effector Force: ', env.robots[0].ee_force[-1])
