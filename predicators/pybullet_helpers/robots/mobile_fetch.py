from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.ikfast import IKFastInfo

from typing import Optional


"""
A version of the fetch robot with movable base.
"""

class MobileFetchRobot(MobileSingleArmPyBulletRobot):

	@classmethod
	def get_name(cls):
		return "fetch_mobile"


	@classmethod
	def urdf_path(cls):
		return utils.get_env_asset_path("urdf/igibson_fetch_description/robots/fetch.urdf")


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
	def wrist_roll_link_name(self):
		return "wrist_roll_link"


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


	@classmethod
	def ikfast_info(cls) -> Optional[IKFastInfo]:
		return IKFastInfo(
			module_dir="fetch_arm",
			module_name="pyikfast_fetch",
			#base_link="base_link",
			base_link="torso_lift_link",
			ee_link="wrist_roll_link",
			# The 7-DOF arm chain for fetch includes shoulder_pan, shoulder_lift, upperarm_roll,
			# elbow_flex, forearm_roll, wrist_flex, wrist_roll. However since IKFast only computes 
			# on 6 joints out of the 7, one of the redundant joints is passed as a free joint.
			# Here, that is the upperarm_roll_joint. Could also be the wrist_roll_joint.
			# free_joints=["upperarm_roll_joint", "forearm_roll_joint", "wrist_roll_joint"]
			# According to Jorge's past experience, the IKFast requires the free joint to be
			# shoulder_pan_joint which was joint index 5 in the reference implementations.
			free_joints=["shoulder_pan_joint"]
		)
	
	
	
	
			