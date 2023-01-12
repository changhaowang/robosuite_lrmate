import robosuite as suite
import numpy as np

if __name__ == "__main__":

    # Create Env
    env = suite.make(
        env_name="Door",
        robots="Lrmate",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
    )

    env.reset()
    for i in range(1000):
        action = np.random.randn(env.robots[0].dof) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        print(env.robots[0]._joint_positions)
        print(env.robots[0].ee_force)
        print(env.robots[0].ee_torque)
        env.render()  # render on display
