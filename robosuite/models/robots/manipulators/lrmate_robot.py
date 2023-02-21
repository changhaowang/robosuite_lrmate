import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class LRmate(ManipulatorModel):
    """
    Lrmate is a FANUC robot used in MSC Lab

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/lrmate/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def default_mount(self):
        return "RethinkMount" # None

    @property
    def default_gripper(self):
        return "PandaGripper" # None
    @property
    def default_controller_config(self):
        return "default_lrmate"

    @property
    def init_qpos(self):
        # return np.array([0.0, 0.0, 0.0, 0.0, -np.pi/2, 0.0])
        # return np.array([-0.09303073354183951, 0.7610021225351699, -0.6254490139440789, 1.6162066969530529, -1.6089099031003165, 1.4742470650341504]) # Door environment
        return np.array([-0.03477543, 0.89188467, -0.57513507, 0.08426756, -0.11839037, 0.08335862]) # wipe environment


    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, -0.1),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
