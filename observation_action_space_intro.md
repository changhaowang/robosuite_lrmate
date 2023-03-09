# Instruction for Real Robots Experiments

## How to know the order of observation and action space for RL traniing in robosuite
### Observations
1. check the file ```gym_wrapper.py``` and the ```_flatten_obs``` function can print the order of observations. However, this only tells ```object state``` and ```robot state``` order
2. Then check ```_get_observations``` function in the ```base.py``` to see the observation order
3. Normally, ```object_state``` goes first, which order is ```door_pos```, ```handle_pos```, ```door_to_eef_pos```, ```handle_to_eef_pos```, ```hinge_qpos```. Then comes the ```robot_related_state``` with the order of ```(robot_joint_pos)```, ```robot_eef_pos```, and ```robot_eef_quat```.
### Actions
1. check the function ```action_limits``` in the file ```singgle_arm.py```, and the ```control_limits``` in the corresponding controller file.
2. Normally, ```control_limits``` goes first, which contains ```kp``` value in 6 dimension then ```action``` in operational space. Then the gripper action.

## Importance Notice for real robot experiments
### Order of Euler Angles
1. The orientation for OSC_POSE is defined by ```axisangle``` in robosuite
2. Euler angle for observation used in simulink is defined in ```ZYX``` order in radians
3. (fixed in the admittance controller folder) ~~Euler angle for cartesian space impedance control is not clear, since the implementation is a bit messy. After converting everything in euler in ```ZYX``` in radians, we should change the order of the action to match the implementation of simulink. To be more specific, real robot [3, 2, -1] <- actually ```ZYX``` euler [1, 2, 3]~~

## Observation and Action space for each trianing environment
### Door Env
#### Initial robot pose
1.```np.array([-0.09303073354183951, 0.7610021225351699, -0.6254490139440789, 1.6162066969530529, -1.6089099031003165, 1.4742470650341504]) # Door environment```
#### Observations (can be seen in _setup_observables function in each environment file (like door.py))
1. ```[door_pos, handle_pos, robot0_joint_pos]```
#### Action 
1. ```kp/kd + position/velcocity + gripper. Can be seen in action_limits function in single_arm.py, and control_limits function in the controller file```
### Wipe Env
#### Initial robot pose
1. ```np.array([-0.03477543, 0.89188467, -0.57513507, 0.08426756, -0.11839037, 0.08335862]) # Wipe environment```