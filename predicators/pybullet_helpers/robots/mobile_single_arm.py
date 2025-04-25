import numpy as np
import pybullet as p
from gym.spaces import Box
from typing import Sequence, Tuple, Optional, Union, Iterator, Callable

from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators import utils
import time

"""
This code extends the Single Arm Pybullet Robot implementation from single_arm.py to support fetch (and maybe others later) robots with
mobile base. Implements differential-drive kinematics for the robot using velocity v
and yaw omega.

Explanation of concepts:
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

        #Initialize the ee_home_pose based on robot's base position:


        base_x, base_y, base_z = base_pose.position

        ee_home_pose = Pose(position=(base_x+0.3, base_y, 0.1),
                            orientation=ee_home_pose.orientation
                            )



        super().__init__(ee_home_pose, physics_client_id, base_pose=base_pose, use_fixed_base = False)

        # Cache wheel joint IDs
        self.wheel_ids = [
            self.joint_from_name(name)
            for name in self.wheel_joint_names
        ]

        # Initialize wheel parameters with default values (unit: meters)
        self._wheel_radius = 0.065
        self._wheel_separation = 0.37476


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

    @property
    def action_space(self) -> Box:
        """
        Extend the arm action space by appending linear velocity v
        and angular velocity ω for differential drive.
        """
        arm_box = super().action_space
        low = np.concatenate((arm_box.low, np.array([-1.0, -3.14])))
        high = np.concatenate((arm_box.high, np.array([1.0, 3.14])))
        return Box(low=low, high=high, dtype=np.float32)

    def drive_base_twist(
        self,
        v: float,
        omega: float,
        physics_client_id: int,
        max_force: float = 20
    ) -> None:
        """
        Set left/right wheel velocities that realize a (v, omega) twist.

        Args:
            v: linear velocity (m/s)
            omega: angular velocity (rad/s)
            physics_client_id: PyBullet physics client ID
            max_force: maximum force to apply to wheels
        """

        # Ensure proper contact with ground
        p.changeDynamics(
            0,  # Ground plane is typically object ID 0
            -1,  # -1 for the base
            restitution=0.0,  # No bouncing
            linearDamping=0.5,  # Add damping to prevent oscillation
            lateralFriction=1.0,
            contactStiffness=1000,  # Higher contact stiffness
            contactDamping=1000,  # Higher contact damping
            physicsClientId=physics_client_id
        )

        # Add proper dynamics to robot base
        p.changeDynamics(
            self.robot_id,
            -1,  # -1 for base link
            mass=5.0,  # Reasonable mass
            restitution=0.0,  # No bouncing off ground
            linearDamping=0.5, 
            angularDamping=0.9,  # Add high angular damping to prevent tipping
            lateralFriction=1.0,  # Good ground friction
            contactStiffness=1000,
            contactDamping=1000,
            physicsClientId=physics_client_id
        )

        # Disable default position motor control first
        for wheel_id in self.wheel_ids:
            p.setJointMotorControl2(
                self.robot_id,
                wheel_id,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,  # Force=0 disables the motor
                physicsClientId=physics_client_id
            )
            
            # Reduce friction for wheels
            p.changeDynamics(
                self.robot_id,
                wheel_id,
                lateralFriction=1.0,
                rollingFriction=0.01,
                spinningFriction=0.01,
                linearDamping=0.2,
                angularDamping=0.2,
                jointDamping=0.0,
                frictionAnchor=1,
                physicsClientId=physics_client_id
            )


        L = self._wheel_separation
        R = self._wheel_radius
        v_l = (2 * v - omega * L) / (2 * R)
        v_r = (2 * v + omega * L) / (2 * R)

        # current_v_l, current_v_r = 0, 0
        # ramp_factor = 0.1  # Gradual acceleration

        # # Desired velocities
        # desired_v_l = (2 * v - omega * L) / (2 * R)
        # desired_v_r = (2 * v + omega * L) / (2 * R)

        # # Gradual transition to desired velocities
        # v_l = current_v_l + ramp_factor * (desired_v_l - current_v_l)
        # v_r = current_v_r + ramp_factor * (desired_v_r - current_v_r)

        # # Save current velocities for next time
        # self.current_v_l, self.current_v_r = v_l, v_r

        # Apply wheel velocities:
        p.setJointMotorControl2(
            self.robot_id,
            self.wheel_ids[0],
            p.VELOCITY_CONTROL,
            targetVelocity=v_l,
            force=max_force,
            physicsClientId=physics_client_id
        )
        p.setJointMotorControl2(
            self.robot_id,
            self.wheel_ids[1],
            p.VELOCITY_CONTROL,
            targetVelocity=v_r,
            force=max_force,
            physicsClientId=physics_client_id
        )

    def path_to_wheel_vels(
        self,
        path: Sequence[Tuple[float, float, float]],
        dt: float = 0.1,
        v_max: float = 0.6,
        omega_max: float = 1.5
    ) -> Iterator[Tuple[float, float, int]]:
        """
        Convert a path of (x, y, θ) waypoints into (v, omega, steps).
        """
        for i in range(len(path) - 1):
            x1, y1, th1 = path[i]
            x2, y2, th2 = path[i + 1]

            # Compute distance and angle to next waypoint
            dx, dy = x2 - x1, y2 - y1
            distance = np.sqrt(dx * dx + dy * dy)
            target_heading = np.arctan2(dy, dx)

            # Normalize angle difference to [-π, π]
            angle_error = (target_heading - th1 + np.pi) % (2 * np.pi) - np.pi

            # Simple proportional control
            omega = min(max(2.0 * angle_error, -omega_max), omega_max)

            # Reduce velocity when turning sharply
            turning_factor = max(0, 1 - abs(angle_error) / (np.pi / 4))
            v = min(v_max * turning_factor, distance / dt)

            # Compute number of steps for this segment
            num_steps = max(1, int(distance / (v * dt)))

            yield v, omega, num_steps

    def execute_path(
        self,
        path: Sequence[Tuple[float, float, float]],
        physics_client_id: int,
        timestep: float = 0.1,
        render_fn: Optional[Callable] = None
    ) -> None:
        """
        Execute a path using the differential drive controller.

        Args:
            path: List of (x,y,theta) waypoints
            physics_client_id: PyBullet physics ID
            timestep: Time step for simulation
            render_fn: Optional function to call for rendering
        """
        for v, omega, steps in self.path_to_wheel_vels(path, dt=timestep):
            self.drive_base_twist(v, omega, physics_client_id)
            for _ in range(steps):
                p.stepSimulation(physicsClientId=physics_client_id)
                if render_fn:
                    render_fn()

    # def execute_path(
    #     self,
    #     path: Sequence[Tuple[float, float, float]],
    #     physics_client_id: int,
    #     timestep: float = 0.1,
    #     render_fn: Optional[Callable] = None
    # ) -> None:
    #     """
    #     Execute a path using position-based control instead of velocities.
    #     This is much more stable in PyBullet.
    #     """
    #     # Force proper initial height and dynamics
    #     self._reset_base_height(physics_client_id)
        
    #     # Use small position increments rather than velocities
    #     for i in range(1, len(path)):
    #         prev_x, prev_y, prev_theta = path[i-1]
    #         target_x, target_y, target_theta = path[i]
            
    #         # How many steps to take for this segment
    #         distance = np.sqrt((target_x - prev_x)**2 + (target_y - prev_y)**2)
    #         num_steps = max(10, int(distance * 50))  # At least 10 steps, more for longer segments
            
    #         # Interpolate between waypoints
    #         for step in range(1, num_steps + 1):
    #             t = step / num_steps
                
    #             # Linear interpolation for position
    #             x = prev_x + t * (target_x - prev_x)
    #             y = prev_y + t * (target_y - prev_y)
                
    #             # Angular interpolation (handle wrap-around)
    #             angle_diff = (target_theta - prev_theta + np.pi) % (2 * np.pi) - np.pi
    #             theta = prev_theta + t * angle_diff
                
    #             # Use teleportation (guaranteed stable)
    #             self.move_base_to((x, y, theta), physics_client_id)
                
    #             # Step simulation and render
    #             p.stepSimulation(physicsClientId=physics_client_id)
    #             if render_fn:
    #                 render_fn()
                    
    #             # Small delay for stability
    #             time.sleep(timestep / num_steps)


    def get_base_pose(self, physics_client_id: int, mode: str="velocity") -> Tuple[float, float, float]:
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

        if mode == "position":
            return (pos[0], pos[1], pos[2])

        euler = p.getEulerFromQuaternion(orn)
        return (pos[0], pos[1], euler[2])

    def move_base_to(
        self,
        target_pose: Union[Tuple[float, float, float], Pose],
        physics_client_id: int
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
            
            # Set new position and orientation
            new_pos = [x, y, 0.01]  # Set z to safe height
            new_orn = p.getQuaternionFromEuler([0, 0, theta])
            
            p.resetBasePositionAndOrientation(
                self.robot_id, new_pos, new_orn, physicsClientId=physics_client_id)
            
            # Step simulation to update visuals and physics
            for _ in range(5):  # Multiple steps for stability
                p.stepSimulation(physicsClientId=physics_client_id)
            
            time.sleep(time_step)  # Sleep for smoother visualization


    # def move_base_to(
    #     self,
    #     target_pose: Union[Tuple[float, float, float], Pose],
    #     physics_client_id: int
    # ) -> None:
    #     """
    #     Move the robot's base to the target pose directly with stability guarantees.
    #     """
    #     if isinstance(target_pose, tuple):
    #         x, y, theta = target_pose
    #         # Use higher height for stability
    #         pos = [x, y, 0.2]  # Increased to 0.2m for clearance
    #         orn = p.getQuaternionFromEuler([0, 0, theta])
    #     else:
    #         # Extract x, y and enforce safe height
    #         x, y, _ = target_pose.position
    #         pos = [x, y, 0.2]
    #         _, _, yaw = p.getEulerFromQuaternion(target_pose.orientation)
    #         orn = p.getQuaternionFromEuler([0, 0, yaw])

    #     # Apply the position change
    #     p.resetBasePositionAndOrientation(
    #         self.robot_id, pos, orn, physicsClientId=physics_client_id
    #     )
        
    #     # Make sure all wheels stay in contact with ground
    #     self._update_wheel_contact(physics_client_id)

    # def _reset_base_height(self, physics_client_id: int) -> None:
    #     """Reset the base height and ensure proper ground contact."""
    #     pos, orn = p.getBasePositionAndOrientation(
    #         self.robot_id, physicsClientId=physics_client_id)
        
    #     # Set a safe height
    #     safe_pos = [pos[0], pos[1], 0.2]
        
    #     # Reset position
    #     p.resetBasePositionAndOrientation(
    #         self.robot_id, safe_pos, orn, physicsClientId=physics_client_id)
        
    #     # Apply strong ground dynamics
    #     p.changeDynamics(
    #         0,  # Ground plane
    #         -1,
    #         restitution=0.0,
    #         lateralFriction=1.0,
    #         contactStiffness=10000,
    #         contactDamping=1000,
    #         physicsClientId=physics_client_id
    #     )
        
    #     # Step simulation to let physics settle
    #     for _ in range(10):
    #         p.stepSimulation(physicsClientId=physics_client_id)

    # def _update_wheel_contact(self, physics_client_id: int) -> None:
    #     """Ensure wheels maintain ground contact."""
    #     # For each wheel, project a ray downward and adjust wheel positions
    #     for wheel_id in self.wheel_ids:
    #         # Get wheel link state
    #         link_state = p.getLinkState(
    #             self.robot_id, wheel_id, physicsClientId=physics_client_id)
    #         wheel_pos = link_state[0]  # World position
            
    #         # Optional: Perform raycasting to find ground level
    #         # This would be the proper way to ensure wheel-ground contact
    #         # but simpler approach used here for stability
