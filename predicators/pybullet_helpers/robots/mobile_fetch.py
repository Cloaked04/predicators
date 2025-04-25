from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot

"""
A version of the fetch robot with movable base.
"""

class MobileFetchRobot(MobileSingleArmPyBulletRobot):

	@classmethod
	def get_name(cls):
		return "fetch_mobile"


	@classmethod
	def urdf_path(cls):
		return utils.get_env_asset_path("urdf/fetch_description/robots/fetch.urdf")


	@property
	def wheel_joint_names(self):
		return ("l_wheel_joint", "r_wheel_joint")


	@property
	def end_effector_name(self):
		return "gripper_axis"


	@property
	def tool_link_name(self):
		return "gripper_link"


	@property
	def left_finger_joint_name(self):
		return "l_gripper_finger_joint"


	@property
	def right_finger_joint_name(self):
		return "r_gripper_finger_joint"
	

	@property
	def open_fingers(self):
		return 0.04

	@property
	def closed_fingers(self):
		return 0.01
	
	
	
	
			