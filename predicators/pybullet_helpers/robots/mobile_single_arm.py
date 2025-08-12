import numpy as np
import pybullet as p
from gym.spaces import Box
from typing import Sequence, Tuple, Optional, Union, Iterator, Callable
from functools import cached_property
from numpy.typing import NDArray

from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.pybullet_helpers.link import get_link_state, get_link_pose
from predicators import utils
import time

"""
This code extends the Single Arm Pybullet Robot implementation from single_arm.py to support fetch 
(and maybe others later) robots with mobile base. Implements differential-drive kinematics for the
robot using velocity v and yaw omega.

Explanation of concepts: TODO!!!
"""

class MobileSingleArmPyBulletRobot(SingleArmPyBulletRobot):
    """Single arm, mobile-base robot with differential drive wheels."""


    def __init__(
        self,
        ee_home_pose: Pose,
        physics_client_id: int,
        base_pose: Pose = Pose.identity(),
    ) -> None:
        # Setting mobile flag true before initializing using super init:
        #self._mobile = True


        super().__init__(ee_home_pose, physics_client_id, base_pose=base_pose, use_fixed_base = False)

        # Cache wheel joint IDs
        self.wheel_ids = [
            self.joint_from_name(name)
            for name in self.wheel_joint_names
        ]

        # Initialize wheel parameters with default values (unit: meters)
        self._wheel_radius = 0.065
        self._wheel_separation = 0.3748

        for wid in self.wheel_ids:
            p.changeDynamics(
                self.robot_id,
                wid,
                lateralFriction=2.0,
                spinningFriction=0.1,
                rollingFriction=0.0,
                physicsClientId=self.physics_client_id)


    @property
    def footprint_radius(self) -> float:
        """Radius (m) of the circular footprint that encloses the wheels."""
        # half the axle length plus a small safety margin
        return self._wheel_separation * 0.5 + 0.03

    def footprint_circle_at(self,
                            pose: Tuple[float, float, float]) -> "utils.Circle":
        """Return a utils.Circle centred at (x,y) in the world frame."""
        from predicators import utils      # local import to avoid cycles
        x, y, _ = pose
        return utils.Circle(x, y, self.footprint_radius)

    def blocks_table_top(self,physics_client_id:int) -> utils.Rectangle:
        """
        Returns an axis‑aligned utils.Rectangle that represents the
        wood surface of the Blocks table 
        """
        # Dimensions taken from table.urdf
        # Currently this is from both refined_table.urdf 

        table_id = 1

        table_pos, table_orn = p.getBasePositionAndOrientation(table_id, physicsClientId=physics_client_id)

        width, height = 0.5, 0.9
        cx, cy, _ = table_pos                       # table centre
        llx = cx - width  / 2
        lly = cy - height / 2
        return utils.Rectangle(llx, lly, width, height, theta=0.0)

    @property
    def wheel_joint_names(self) -> Sequence[str]:
        """
        Names of the wheel joints. To be overridden in specific mobile robot class.
        """
        raise NotImplementedError("Override Me!!!!")

    @cached_property
    def wheel_ids(self) -> list[int]:
        "Return pybullet joint ids for wheels."
        return [self.joint_from_name(name) for  name in self.wheel_joint_names]
    

    # @property
    # def action_space(self) -> Box:
    #     """
    #     Extend the arm action space by appending linear velocity v
    #     and angular velocity ω for differential drive.
    #     """
    #     arm_box = super().action_space
    #     low = np.concatenate((arm_box.low, np.array([-1.0, -3.14])))
    #     high = np.concatenate((arm_box.high, np.array([1.0, 3.14])))
    #     return Box(low=low, high=high, dtype=np.float32)


    # def execute_path(
    #     self,
    #     path: Sequence[Tuple[float, float, float]],
    #     physics_client_id: int,
    #     timestep: float = 0.1,
    #     render_fn: Optional[Callable] = None
    # ) -> None:
    #     """
    #     Execute a path using the differential drive controller.

    #     Args:
    #         path: List of (x,y,theta) waypoints
    #         physics_client_id: PyBullet physics ID
    #         timestep: Time step for simulation
    #         render_fn: Optional function to call for rendering
    #     """
    #     for v, omega, steps in self.path_to_wheel_vels(path, dt=timestep):
    #         self.drive_base_twist(v, omega, physics_client_id)
    #         for _ in range(steps):
    #             p.stepSimulation(physicsClientId=physics_client_id)
    #             if render_fn:
    #                 render_fn()



    def get_base_pose(self, physics_client_id: int):
        """
        Get the current pose of the robot base.

        Args:
            physics_client_id: PyBullet physics client ID
            mode: "position" for (x,y,z) or "velocity" for (x,y,theta)

        Returns:
            Tuple(x,y,z) for mode="position" ; Tuple (x, y, theta) for mode="velocity"
        """
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=physics_client_id
        )

        # if mode == "position":
        #     return (pos[0], pos[1], pos[2])

        euler = p.getEulerFromQuaternion(orn)
        return (pos[0], pos[1], euler[2])

    def move_base_to(
        self,
        target_pose: Union[Tuple[float, float, float], Pose],
        physics_client_id: int,
        held_object_id: Optional[int] = None,
        ee_link_to_held_object: Optional[NDArray] = None,
    ) -> None:
        """
        Move the robot's base to the target pose directly.
        This function doesn't check for dynamics or collision.
        To use for initialization or collision free path.

        Args:
            target_pose: Tuple (x, y, theta) or Pose object
            physics_client_id: PyBullet physics client ID
        """
        if isinstance(target_pose, tuple):
            x, y, theta = target_pose
            #pos = [x, y, self._base_pose.position[2]]
            pos = [x, y, 0.05]
            orn = p.getQuaternionFromEuler([0, 0, theta])
        else:
            pos = target_pose.position
            #Extract yaw from the orientation quaternion and only use yaw
            #Understand these concepts better
            _, _, yaw = p.getEulerFromQuaternion(target_pose.orientation)
            orn = p.getQuaternionFromEuler([0, 0, yaw])

        p.resetBasePositionAndOrientation(
            self.robot_id, pos, orn, physicsClientId=physics_client_id
        )

        if held_object_id is not None:
            assert ee_link_to_held_object is not None
            world_to_ee_link = get_link_state(
                self.robot_id,
                self.end_effector_id,
                physics_client_id=physics_client_id).com_pose
            world_to_held_obj = p.multiplyTransforms(world_to_ee_link[0],
                                                     world_to_ee_link[1],
                                                     ee_link_to_held_object[0],
                                                     ee_link_to_held_object[1])
            p.resetBasePositionAndOrientation(
                held_object_id,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=physics_client_id)



    def set_wheel_motors(self, robot: SingleArmPyBulletRobot, omega_r: float, omega_l: float, physicsClientId:int) -> None:
        """Set velocity for both wheels.
        """

        # X, Y, THETA = robot.get_base_pose(physicsClientId)
        # vx, vy = 0.3 * np.cos(THETA), 0.3 * np.sin(THETA)
        # p.resetBaseVelocity(
        #     robot.robot_id,
        #     linearVelocity  = [vx, vy, 0.0],
        #     angularVelocity = [0.0, 0.0, omega],
        #     physicsClientId = physicsClientId)


        p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=[self.wheel_ids[1], self.wheel_ids[0]],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[omega_r, omega_l],
                forces=[100.0, 100.0],
                physicsClientId=physicsClientId
            )



    def move_base_smoothly(
        self,
        target_pose: Tuple[float, float, float],
        physics_client_id: int,
        step_size: float = 0.01,
        time_step: float = 0.01 
    ) -> None:
        """
        Move the robot's base to the target pose using small incremental steps.
        
        Args:
            target_pose: Target (x, y, theta) pose
            physics_client_id: PyBullet physics client ID
            step_size: Maximum distance to move in a single step (meters)
            time_step: Time to wait between steps (seconds)
        """
        # Get current position and orientation
        current_pos, current_orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=physics_client_id)
        current_x, current_y, current_z = current_pos
        current_euler = p.getEulerFromQuaternion(current_orn)
        current_theta = current_euler[2]
        
        # Target position and orientation
        target_x, target_y, target_theta = target_pose
        
        # Calculate total distance and angle difference
        dx, dy = target_x - current_x, target_y - current_y
        total_distance = np.sqrt(dx**2 + dy**2)
        
        # Normalize angle difference to [-pi, pi]
        angle_diff = ((target_theta - current_theta + np.pi) % (2*np.pi)) - np.pi
        
        # Calculate number of steps needed
        num_steps = max(int(total_distance / step_size), int(abs(angle_diff) / (step_size * 2)))
        num_steps = max(num_steps, 1)  # At least one step
        
        # Move in small steps
        for i in range(1, num_steps + 1):
            # Calculate intermediate position and orientation
            fraction = i / num_steps
            x = current_x + dx * fraction
            y = current_y + dy * fraction
            theta = current_theta + angle_diff * fraction
            
            #Set new position and orientation
            new_pos = [x, y, 0.01]  # Set z to safe height
            new_orn = p.getQuaternionFromEuler([0, 0, theta])
            
            p.resetBasePositionAndOrientation(
                self.robot_id, new_pos, new_orn, physicsClientId=physics_client_id)
            
            # Step simulation to update visuals and physics
            for _ in range(10):  # Multiple steps for stability
                p.stepSimulation(physicsClientId=physics_client_id)
            
            time.sleep(time_step)  # Sleep for smoother visualization


    