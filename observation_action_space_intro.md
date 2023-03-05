## Importance Notice for real robot experiments
### Order of Euler Angles
1. Euler angle used in robosuite is defined in ```YZX``` order in radians
2. Euler angle for observation used in simulink is defined in ```ZYX``` order in radians
3. Euler angle for cartesian space impedance control is not clear, since the implementation is a bit messy. Please calibrate on your self.

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