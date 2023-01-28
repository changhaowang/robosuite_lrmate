import numpy as np
from fanuc_kinematics import IK
from fanuc_kinematics import FK

handle_pos = np.array([-0.13189581, -0.25602616,  0.07500515, 0, np.pi, np.pi])
q = IK(handle_pos)
print(q)
