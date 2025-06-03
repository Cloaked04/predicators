"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence, List, Tuple, Any

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.inverse_kinematics import InverseKinematicsError
from predicators.pybullet_helpers.geometry import Pose
from predicators.settings import CFG


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: Optional[int] = None,
    ee_to_held_object_transform: Optional[NDArray] = None,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.
    For MobileSingleArmPyBulletRobot, initial_positions and target_positions
    should pertain to arm joints only.
    """
    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    num_interp = CFG.pybullet_birrt_extend_num_interp

    # Determine the actual joint space for the arm
    if isinstance(robot, MobileSingleArmPyBulletRobot):
        # The arm_joints are defined in SingleArmPyBulletRobot.
        # They can be derived from the robot's full action space.
        arm_joint_limits_low = joint_space.low[:-2]
        arm_joint_limits_high = joint_space.high[:-2]

        # Create a new Box space specifically for arm joints
        current_arm_joint_space = Box(low=arm_joint_limits_low, high=arm_joint_limits_high)
        
        # Ensure initial_positions and target_positions are arm only
        num_arm_joints_expected = len(current_arm_joint_space.low)
        if len(initial_positions) != num_arm_joints_expected:
            raise ValueError(f"Initial positions length {len(initial_positions)} "
                            f"does not match arm space dim {num_arm_joints_expected}")

        if len(target_positions) != num_arm_joints_expected:
            raise ValueError(f"Target positions length {len(target_positions)} "
                            f"does not match arm space dim {num_arm_joints_expected}")
    else:
        current_arm_joint_space = joint_space

    def _sample_fn(pt: JointPositions) -> JointPositions:
        '''
        Samples a new joint configuration from the robot's action space.
        The joint_space.sample() method generates a random configuration for all joints.
        Ensures that the finger joint positions remain unchanged.

        Returns: returns new_pt, which is the new joint configuration with the finger
                 positions preserved.
        '''

        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
        new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        if held_object is not None:
            assert ee_to_held_object_transform is not None
            world_to_base_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=physics_client_id).com_pose
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     ee_to_held_object_transform[0],
                                                     ee_to_held_object_transform[1])
            p.resetBasePositionAndOrientation(
                held_object,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=physics_client_id)

    def _extend_fn(pt1: JointPositions,
                   pt2: JointPositions) -> Iterator[JointPositions]:
        """
         produces a generator that streams a densely sampled straight-line trajectory from pt1 to pt2.
        """
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointPositions) -> bool:
        _set_state(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_bodies:
            if p.getContactPoints(robot.robot_id,
                                  body,
                                  physicsClientId=physics_client_id):
                return True
            if held_object is not None and p.getContactPoints(
                    held_object, body, physicsClientId=physics_client_id):
                return True
        return False

    def _distance_fn(from_pt: JointPositions, to_pt: JointPositions) -> float:
        # NOTE: only using positions to calculate distance. Should use
        # orientations as well?.
        from_ee = robot.forward_kinematics(from_pt).position
        to_ee = robot.forward_kinematics(to_pt).position
        return sum(np.subtract(from_ee, to_ee)**2)

    birrt = utils.BiRRT(_sample_fn,
                        _extend_fn,
                        _collision_fn,
                        _distance_fn,
                        rng,
                        num_attempts=CFG.pybullet_birrt_num_attempts,
                        num_iters=CFG.pybullet_birrt_num_iters,
                        smooth_amt=CFG.pybullet_birrt_smooth_amt)

    return birrt.query(initial_positions, target_positions)


def run_base_motion_planning(
    robot: MobileSingleArmPyBulletRobot,
    target_pose: Tuple[float, float, float],
    collision_bodies: Collection[int],
    current_arm_positions: JointPositions,
    seed: int,
    physics_client_id: int,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    held_object_id: Optional[int] = None,
    ee_to_held_object_transform: Optional[Tuple[NDArray, NDArray]] = None,
) -> Optional[List[Tuple[float, float, float]]]:
    
    """
    Implements motion planning (BiRRT) for robot with mobile base.
    Accounts for a held object during collision checking.

    Args:
        robot: The mobile robot
        target_pose: Target pose as (x,y,theta)
        collision_bodies: Collection of body IDs to avoid
        seed: Random seed
        physics_client_id: PyBullet physics client ID
        workspace_bounds: Optional(min_x, min_y, max_x, max_y) bounds
        current_arm_positions: The joint positions of the arm to maintain during checks.
        held_object_id: Optional ID of an object held by the robot.
        ee_to_held_object_transform: Optional transform (pos, orn) from EE to held object.

    Returns:
        List of (x,y,theta) waypoints or None if no path is found
    """

    rng = np.random.default_rng(seed)

    #Get robot's current pose in simulation.
    current_base_sim_pose = robot.get_base_pose(physics_client_id)

    # Workspace bounds (use default values if not provided)
    if workspace_bounds is None:
        workspace_bounds = (0.0, 0.0, 3.0, 3.0)
    min_x, min_y, max_x, max_y = workspace_bounds

    def _sample_fn() -> Tuple[float, float, float]:
        """Sample a random base pose in the workspace.
           Return a value between the min-max x and y coords and between 0-2pi for theta.
        """
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        theta = rng.uniform(0, 2*np.pi)
        return (x, y, theta)

    def _extend_fn(pose1: Tuple[float, float, float],
                    pose2: Tuple[float, float, float]) -> Iterator[Tuple[float, float, float]]:
        """
        Generate interpolated poses between pose1 and pose2
        """
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2

        # Calculate distance and steps
        distance = np.sqrt((x2-x1)**2+(y2-y1)**2)
        num_interpolation = max(2, int(distance*10))

        # Normalize angle difference to [-pi, pi]
        # Normalization of the angle difference is needed to ensure shortest interpolation
        # path across the circle.
        angle_diff = ((theta2-theta1+np.pi)%(2*np.pi))

        for i in range(1, num_interpolation+1):
            alpha = i/num_interpolation
            x = x1+alpha*(x2-x1)
            y = y1+alpha*(y2-y1)
            theta = theta1+alpha*angle_diff

            # normalize theta just to keep it in the range [0,2pi).
            theta = theta%(2*np.pi)
            yield (x, y, theta)

    def _collision_fn(pose_to_check: Tuple[float, float, float]) -> bool:
        """
        Check if the robot at the given base pose collides with any objects.
        Also considers a held object to check for collision if provided.
        """
        # Save robot's original full state
        original_robot_base_pos, original_robot_base_orn = p.getBasePositionAndOrientation(
            robot.robot_id, physicsClientId=physics_client_id)
        # current_arm_positions is already passed and represents the arm state to check with

        # If robot is holding an object, save its orientation as well.
        original_held_object_pos_orn: Optional[Tuple[NDArray, NDArray]] = None
        if held_object_id is not None:
            original_held_object_pos_orn = p.getBasePositionAndOrientation(
                held_object_id, physicsClientId=physics_client_id)

        # Set robot base to the test pose
        # robot.move_base_to uses p.resetBasePositionAndOrientation
        robot.move_base_to(pose_to_check, physics_client_id)
        # Ensure arm is in the desired configuration. Recall that arm joint positions are 
        # not connected to base position.
        robot.set_joints(current_arm_positions)

        # If holding an object, update its pose based on new base and arm state
        if held_object_id is not None:
            assert ee_to_held_object_transform is not None
            # It's crucial to use the correct link for the transform.
            # Assuming the grasp constraint is on robot.end_effector_id
            ee_link_world_pose = get_link_state(
                robot.robot_id,
                robot.end_effector_id, # Or robot.tool_link_id, depending on grasp def
                physics_client_id=physics_client_id
            ).pose # This is a Pose(position, orientation)

            new_held_object_pos, new_held_object_orn = p.multiplyTransforms(
                ee_link_world_pose.position,
                ee_link_world_pose.orientation,
                ee_to_held_object_transform[0],  # position part of transform
                ee_to_held_object_transform[1]   # orientation part of transform
            )
            p.resetBasePositionAndOrientation(
                held_object_id,
                new_held_object_pos,
                new_held_object_orn,
                physicsClientId=physics_client_id
            )

        # Perform collision detection
        p.performCollisionDetection(physicsClientId=physics_client_id)
        
        collides = False
        # Check robot body collisions
        for body_id in collision_bodies:
            if p.getContactPoints(robot.robot_id, body_id, physicsClientId=physics_client_id):
                collides = True
                break
        
        # Check held object collisions (if any and no collision found yet)
        if not collides and held_object_id is not None:
            for body_id in collision_bodies:
                # Don't check collision between held object and robot itself -- LOL!!
                if body_id == robot.robot_id:
                    continue
                if p.getContactPoints(held_object_id, body_id, physicsClientId=physics_client_id):
                    collides = True
                    break
        
        # Restore original robot base state
        p.resetBasePositionAndOrientation(
            robot.robot_id, original_robot_base_pos, original_robot_base_orn,
            physicsClientId=physics_client_id)
        # Restore arm joints (important as resetBasePositionAndOrientation can affect them)
        robot.set_joints(current_arm_positions) # Or original arm positions if they could change

        # Restore original held object state
        if held_object_id is not None and original_held_object_pos_orn is not None:
            p.resetBasePositionAndOrientation(
                held_object_id,
                original_held_object_pos_orn[0],
                original_held_object_pos_orn[1],
                physicsClientId=physics_client_id
            )
        return collides

    def _distance_fn(pose1: Tuple[float, float, float],
                     pose2: Tuple[float, float, float]) -> float:
        """
        Compute weighted distance between two base poses
        """
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2

        # Compute position distance
        pos_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Angular distance (normalized to [0, pi])
        angle_diff = abs((theta2-theta1)+np.pi)%(2*np.pi)-np.pi

        # Return Weighted sum:
        return pos_distance+0.3*angle_diff

    # Use BiRRT for planning
    birrt = utils.BiRRT(
        _sample_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=CFG.pybullet_birrt_num_attempts,
        num_iters=CFG.pybullet_birrt_num_iters,
        smooth_amt=CFG.pybullet_birrt_smooth_amt
        )

    path = birrt.query(current_base_sim_pose, target_pose)
    return path


# Helper collision checking function used inside run_coordinated_motion_planning
def _collision_fn(pose: Tuple[float, float, float],
                  robot: MobileSingleArmPyBulletRobot,
                  collision_bodies: Collection[int],
                  physics_client_id: int,
                  current_arm_positions: Optional[JointPositions] = None,
                  held_object_id: Optional[int] = None,
                  ee_to_held_object_transform: Optional[Tuple[NDArray, NDArray]] = None
                  ) -> bool:
    """
    Check if the robot at the given base pose collides with any objects.
    Modifies the simulation state temporarily.
    """
    x, y, theta = pose

    # Save robot's current state(simulation state)
    original_pos, original_orn = p.getBasePositionAndOrientation(
                                    robot.robot_id, physicsClientId=physics_client_id)
    original_joints = robot.get_joints() # These are arm joints

    original_held_object_pos_orn: Optional[Tuple[NDArray, NDArray]] = None
    if held_object_id is not None:
        original_held_object_pos_orn = p.getBasePositionAndOrientation(
            held_object_id, physicsClientId=physics_client_id)

    # Temporarily set robot to test pose.
    robot.move_base_to(pose, physics_client_id)
    # Ensure joints are reset as well.
    if current_arm_positions is not None: # Use specified arm positions if given
        robot.set_joints(current_arm_positions)
    else: # Otherwise, use the ones from before moving the base
        robot.set_joints(original_joints)


    # If holding an object, update its pose based on new base and arm state
    if held_object_id is not None and ee_to_held_object_transform is not None and current_arm_positions is not None:
        # Ensure arm is in the correct configuration for this check
        robot.set_joints(current_arm_positions) 
        # Then get the World-frame Pose of the robot's end-effector after setting joints.
        ee_link_world_pose = get_link_state(
            robot.robot_id,
            robot.end_effector_id, # Assuming transform is relative to end_effector_id
            physics_client_id=physics_client_id
        ).pose

        # Compute the world-frame pose of the held object
        new_held_object_pos, new_held_object_orn = p.multiplyTransforms(
            ee_link_world_pose.position,
            ee_link_world_pose.orientation,
            ee_to_held_object_transform[0], # position part of transform
            ee_to_held_object_transform[1]  # orientation part of transform
        )
        p.resetBasePositionAndOrientation(
            held_object_id,
            new_held_object_pos,
            new_held_object_orn,
            physicsClientId=physics_client_id
        )

    # Perform collision detection without stepping physics
    p.performCollisionDetection(physicsClientId=physics_client_id)

    collides = False
    # If robot collides with any of the objects in collision_bodies,
    # set collides flag to True
    for body_id in collision_bodies:
        if p.getContactPoints(robot.robot_id, body_id, physicsClientId=physics_client_id):
            collides = True
            break
    
    if not collides and held_object_id is not None:
        for body_id in collision_bodies:
            if body_id == robot.robot_id: # Should not happen if collision_bodies is just env objects
                continue
            if p.getContactPoints(held_object_id, body_id, physicsClientId=physics_client_id):
                collides = True
                break

    # Restore original state
    p.resetBasePositionAndOrientation(
        robot.robot_id, original_pos, original_orn,
        physicsClientId=physics_client_id
        )

    robot.set_joints(original_joints) # Restore original joints

    if held_object_id is not None and original_held_object_pos_orn is not None:
        p.resetBasePositionAndOrientation(
            held_object_id,
            original_held_object_pos_orn[0],
            original_held_object_pos_orn[1],
            physicsClientId=physics_client_id
        )

    return collides


# def run_coordinated_motion_planning(
#     robot: MobileSingleArmPyBulletRobot,
#     target_ee_pose: Pose,
#     collision_bodies: Collection[int],
#     seed: int,
#     physics_client_id: int,
#     try_arm_only_first: bool = True,
#     base_path_planner_max_tries: int = 30,
#     workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
#     rng: Optional[np.random.Generator] = None,
#     final_finger_state: Optional[float] = None,
#     held_object_id_at_start: Optional[int] = None,
#     ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None,
# ) -> Optional[Tuple[List[Tuple[float, float, float]], List[JointPositions]]]:
#     """
#     Run co-ordinated motion planning for both base and arm.

#     Attempts to reach the target EE pose, potentially moving the base if
#     necessary.

#     Args:
#         robot: The mobile robot
#         target_ee_pose: Target end-effector pose in world frame.
#         collision_bodies: collection of body ids to avoid while moving
#         seed: Random seed for planning.
#         physics_client_id: PyBullet physics client id.
#         try_arm_only_first: If True, try to reach target orientation only
#                             by using arm movements first.
#         base_path_planner_max_tries: Number of candidate base poses to check.
#         workspace_bounds: Optional workspace limits (min_X, min_Y, max_X, max_y).
#         rng: Optional random number generator.
#         final_finger_state: Optional final finger state for the arm.
#         held_object_id_at_start: Optional ID of an object held by the robot at the start of the coordinated plan.
#         ee_to_held_object_transform_at_start: Optional transform (pos, orn) from EE to held object at the start of the coordinated plan.

#     Returns:
#         Tuple of (base_path, arm_path);

#         base_path: List of (x, y, theta) waypoints for the base. Can contain just the current pose
#                     if only arm movement is needed.
#         arm_path: List of JointPositions waypoints for the arm. Can be empty if only base movement occurs
#                   or if arm planning fails after base movement (should ideally
#                   not happen if reachable base pose was found correctly).

#         Returns None if no plan is found.
#     """
#     if rng is None:
#         rng = np.random.default_rng(seed)

#     # Get robot's current state to restore later.
#     current_base_pose = robot.get_base_pose(physics_client_id)
#     current_joint_positions = robot.get_joints()

#     # First: check whether target is reachable from current
#     # base positions (if try_arm_only_first flag is True)
#     if try_arm_only_first:
#         try:
#             # Try to solve with inverse kinematics
#             target_joint_solution = robot.inverse_kinematics(
#                 target_ee_pose, validate=True, set_joints=False)

#             if final_finger_state is not None and target_joint_solution is not None:
#                 mutable_joint_solution = list(target_joint_solution)
#                 # Ensure indices are valid for the joint_solution array length and only move
#                 # ahead if finger joint indices are present.
#                 # Set the left and right finger values to the same value: either open or closed value.
#                 if robot.left_finger_joint_idx < len(mutable_joint_solution) and \
#                    robot.right_finger_joint_idx < len(mutable_joint_solution):
#                     mutable_joint_solution[robot.left_finger_joint_idx] = final_finger_state
#                     mutable_joint_solution[robot.right_finger_joint_idx] = final_finger_state
#                     target_joint_solution = tuple(mutable_joint_solution)
#                 else:
#                     print(f"Warning: Finger joint indices out of bounds for target_joint_solution in arm-only.")


#             # If IK succeeded, try planning the arm path
#             # Check if this solution is collision-free
#             arm_path = run_motion_planning(
#                 robot,
#                 current_joint_positions,
#                 target_joint_solution,
#                 collision_bodies,
#                 seed, # Consider using rng.integers for better seed generation in loops/retries
#                 physics_client_id,
#                 held_object=held_object_id_at_start,
#                 ee_to_held_object_transform=ee_to_held_object_transform_at_start
#                 )

#             if arm_path is not None:
#                 print("Coordinated planning: Succeeded with arm-only movement.")

#                 #Padding the returned arm_path to include fingers as well so
#                 #set_motors function doesn't run into an error:

#                 padded_arm_path: List[JointPositions] = []

#                 for arm_wp in arm_path:
#                     if len(list(arm_wp)) == 9:
#                         continue
#                     wp7 = list(arm_wp)
#                     #Insert left finger:
#                     wf = robot.open_fingers
#                     wp7.insert(robot.left_finger_joint_idx, wf)
#                     #Insert right finger:
#                     wp7.insert(robot.right_finger_joint_idx, wf)

#                     padded_arm_path.append(wp7)

#                 # Target position was reachable via arm motion only; return it
#                 # Restore the robot to initial positions (in the case these were
#                 # modified during above steps,)
#                 robot.move_base_to(current_base_pose, physics_client_id)
#                 robot.set_joints(current_joint_positions)
#                 return ([current_base_pose], padded_arm_path)

#         except InverseKinematicsError:
#             print("Coordinated Planning: Arm-only IK failed. Moving to base planning.")
#             pass

#         # Restore initial/current state of the robot if planning modified it
#         robot.move_base_to(current_base_pose, physics_client_id)
#         robot.set_joints(current_joint_positions)

#     # Then find a base position that makes the target reachable
#     best_reachable_base_pos = None
#     best_joint_solution = None
#     min_base_distance = float('inf')

#     # target x,y
#     target_pos = np.array(target_ee_pose.position[:2])

#     # try multiple candidate base positions
#     for attempt in range(base_path_planner_max_tries):
#         # Sample a base position near the target EE pose
#         dist_to_target = rng.uniform(0.4, 0.8)  # Sample distance from EE target
#         angle_to_target = rng.uniform(-np.pi/4, np.pi/4)  # Sample angle offset relative to target

#         # ----- Calculate base position relative to target---------
#         # Angle from current base pose to target
#         target_angle = np.arctan2(target_pos[1] - current_base_pose[1],
#                                 target_pos[0] - current_base_pose[0])

#         # Base orientation relative to world
#         candidate_angle_offset = angle_to_target+target_angle

#         # Position behind the target
#         candidate_x = target_pos[0] - dist_to_target*np.cos(candidate_angle_offset)
#         candidate_y = target_pos[1] - dist_to_target*np.sin(candidate_angle_offset)
#         # Orientation towards the target
#         candidate_theta = (candidate_angle_offset+np.pi)%(2*np.pi)

#         # Make the candidate base pose
#         candidate_base_pose = (candidate_x, candidate_y, candidate_theta)
#         # ----------------------------------------------------------

#         # -------- Temp. move base to check IK -------------
#         # Save current state
#         current_sim_base_pos, current_sim_base_orn = p.getBasePositionAndOrientation(
#             robot.robot_id, physicsClientId=physics_client_id)

#         # Temporarily move base to candidate position
#         robot.move_base_to(candidate_base_pose, physics_client_id)

#         # Check if target is reachable from this base pose
#         try:
#             # Check IK reachability without validating collisons
#             joint_solution = robot.inverse_kinematics(
#                 target_ee_pose, validate=True, set_joints=False)

#             # If reachable, check if this candidate is closer
#             base_distance = np.sqrt(
#                 (candidate_x - current_base_pose[0])**2 + 
#                 (candidate_y - current_base_pose[1])**2
#             )

#             # If this is a better solution, save it
#             if base_distance < min_base_distance:
#                 # Basic collision check for candidate base pose itself
#                 if not _collision_fn(
#                     candidate_base_pose, robot, collision_bodies, physics_client_id,
#                     current_arm_positions=current_joint_positions, 
#                     held_object_id=held_object_id_at_start,
#                     ee_to_held_object_transform=ee_to_held_object_transform_at_start
#                 ):
#                     min_base_distance = base_distance
#                     best_reachable_base_pos = candidate_base_pose
#                     # Apply final_finger_state to this joint_solution as well
#                     if final_finger_state is not None and joint_solution is not None:
#                         mutable_js = list(joint_solution)
#                         if robot.left_finger_joint_idx < len(mutable_js) and \
#                            robot.right_finger_joint_idx < len(mutable_js):
#                             mutable_js[robot.left_finger_joint_idx] = final_finger_state
#                             mutable_js[robot.right_finger_joint_idx] = final_finger_state
#                             best_joint_solution = tuple(mutable_js)
#                         else:
#                             print(f"Warning: Finger joint indices out of bounds for joint_solution in base candidate.")
#                             best_joint_solution = joint_solution # Use original if indices invalid
#                     else:
#                         best_joint_solution = joint_solution
#                     print(f"Coordinated planning: Found reachable base candidate")

#         except InverseKinematicsError:
#             # Not reachable from this base position
#             pass

#         # Restore original base pose after checking
#         p.resetBasePositionAndOrientation(robot.robot_id,
#                                           current_sim_base_pos,
#                                           current_sim_base_orn,
#                                           physicsClientId=physics_client_id)

#         # Also reset the robot's internal joint state if IK modified it
#         robot.set_joints(current_joint_positions)

#     if best_reachable_base_pos is None:
#         print("Coordinated planning: Failed to find any reachable base pose.")
#         return None

#     print(f"Coordinated planning: Selected best base pose:{best_reachable_base_pos}")

#     # Plan Base Path
#     # Restore initial state before planning base path
#     robot.move_base_to(current_base_pose, physics_client_id)
#     robot.set_joints(current_joint_positions)

#     base_path = run_base_motion_planning(
#         robot,
#         best_reachable_base_pos,
#         collision_bodies,
#         current_joint_positions,
#         seed,
#         physics_client_id,
#         workspace_bounds=workspace_bounds,
#         held_object_id=held_object_id_at_start,
#         ee_to_held_object_transform=ee_to_held_object_transform_at_start,
#     )

#     if base_path is None:
#         print("Coordinated planning: Base path planning failed.")
#         # Restore initial state
#         robot.move_base_to(current_base_pose, physics_client_id)
#         robot.set_joints(current_joint_positions)
#         return None

#     print(f"Coordinated planning: Base path found with {len(base_path)} waypoints.")

#     # Plan Arm Path
#     # Temporarily move the robot to the end of the planned base path
#     final_base_pose = base_path[-1]
#     robot.move_base_to(final_base_pose, physics_client_id)

#     # Need the joint positions *after* potential base movement influence,
#     # or more robustly, start arm planning from the initial joints *before*
#     # base movement, assuming the target `best_joint_solution` is valid
#     # from the `final_base_pose`. Let's assume the latter for simplicity now.
#     # A more robust method might re-run IK at the final base pose if needed.
    
#     # Ensure best_joint_solution (which is target_positions for arm planner)
#     # has the correct final_finger_state if it was determined.
#     if final_finger_state is not None and best_joint_solution is not None:
#         mutable_bjs = list(best_joint_solution)
#         # Check if finger indices are valid for the length of best_joint_solution
#         if robot.left_finger_joint_idx < len(mutable_bjs) and \
#            robot.right_finger_joint_idx < len(mutable_bjs):
#             mutable_bjs[robot.left_finger_joint_idx] = final_finger_state
#             mutable_bjs[robot.right_finger_joint_idx] = final_finger_state
#             best_joint_solution = tuple(mutable_bjs)
#         else:
#             # This case should be rare if IK solution was valid
#             print(f"Warning: Finger joint indices out of bounds for best_joint_solution. Length: {len(mutable_bjs)}")


#     # It's crucial that run_motion_planning starts from the *correct*
#     # joint state corresponding to the state *before* arm motion begins,
#     # which is current_joint_positions in this sequential execution model.
#     arm_path = run_motion_planning(
#         robot,
#         current_joint_positions,  # Joints before any movement starts
#         best_joint_solution,      # Target joints from IK at final base pose (with fingers updated)
#         collision_bodies,
#         seed,                     # Consider using rng.integers here as well
#         physics_client_id,
#         held_object=held_object_id_at_start,
#         ee_to_held_object_transform=ee_to_held_object_transform_at_start,
#     )

#     # Restore initial state after planning is complete
#     robot.move_base_to(current_base_pose, physics_client_id)
#     robot.set_joints(current_joint_positions)

#     if arm_path is None:
#         print("Coordinated planning: Arm path planning failed after base movement.")
#         # Base path was found, but arm path failed. Depending on desired behavior,
#         # we could return just the base path, but returning None indicates failure
#         # to reach the final coordinated state.
#         return None

#     print(f"Coordinated planning: Arm path found with {len(arm_path)} waypoints.")
#     print("Coordinated planning: Succeeded.")
#     return (base_path, arm_path)


def _compute_facing_theta(robot_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """Compute theta such that robot faces the target."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    return np.arctan2(dy, dx)

def run_coordinated_motion_planning(
    robot: MobileSingleArmPyBulletRobot,
    target_ee_pose: Pose,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    try_arm_only_first: bool = True,
    base_path_planner_max_tries: int = 30,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    rng: Optional[np.random.Generator] = None,
    final_finger_state: Optional[float] = None,
    held_object_id_at_start: Optional[int] = None,
    ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None,
) -> Optional[Tuple[List[Tuple[float, float, float]], List[JointPositions]]]:
    
    if rng is None:
        rng = np.random.default_rng(seed)

    # Get robot's current state to restore later
    current_base_pose = robot.get_base_pose(physics_client_id)
    current_joint_positions = robot.get_joints()

    # First: check whether target is reachable from current base position
    if try_arm_only_first:
        try:
            target_joint_solution = robot.inverse_kinematics(
                target_ee_pose, validate=True, set_joints=False)
            
            if final_finger_state is not None and target_joint_solution is not None:
                mutable_joint_solution = list(target_joint_solution)
                if robot.left_finger_joint_idx < len(mutable_joint_solution) and \
                   robot.right_finger_joint_idx < len(mutable_joint_solution):
                    mutable_joint_solution[robot.left_finger_joint_idx] = final_finger_state
                    mutable_joint_solution[robot.right_finger_joint_idx] = final_finger_state
                    target_joint_solution = tuple(mutable_joint_solution)

            arm_path = run_motion_planning(
                robot,
                current_joint_positions,
                target_joint_solution,
                collision_bodies,
                seed,
                physics_client_id,
                held_object=held_object_id_at_start,
                ee_to_held_object_transform=ee_to_held_object_transform_at_start
            )

            if arm_path is not None:
                print("Coordinated planning: Succeeded with arm-only movement.")

                #Padding the returned arm_path to include fingers as well so
#               #set_motors function doesn't run into an error:

                padded_arm_path: List[JointPositions] = []

                for arm_wp in arm_path:
                    if len(list(arm_wp)) == 9:
                        continue
                    wp7 = list(arm_wp)
                    #Insert left finger:
                    wf = robot.open_fingers
                    wp7.insert(robot.left_finger_joint_idx, wf)
                    #Insert right finger:
                    wp7.insert(robot.right_finger_joint_idx, wf)

                    padded_arm_path.append(wp7)

                robot.move_base_to(current_base_pose, physics_client_id)
                robot.set_joints(current_joint_positions)
                return ([current_base_pose], padded_arm_path)

        except InverseKinematicsError:
            print("Coordinated Planning: Arm-only IK failed. Moving to base planning.")
            pass

    # Restore robot state before base planning
    robot.move_base_to(current_base_pose, physics_client_id)
    robot.set_joints(current_joint_positions)

    
    # Get target base pose near the target EE position
    target_pos = np.array(target_ee_pose.position[:2])
    
    # Position the base at a reasonable distance from target (0.6m behind it)
    dist_to_target = 0.5

    # Calculate distance to target first
    dist_to_target_actual = np.sqrt((target_pos[0] - current_base_pose[0])**2 + 
                                    (target_pos[1] - current_base_pose[1])**2)

    if dist_to_target_actual < 0.3:  # If very close to target
        print("Robot already close to target, skipping base movement")
        return None
    
    # Calculate angle from current base to target
    target_angle = np.arctan2(target_pos[1] - current_base_pose[1],
                             target_pos[0] - current_base_pose[0])
    
    print(f"Target angle:({target_angle}).")

    # Use adaptive distance based on how far we are from target
    adaptive_dist = min(dist_to_target, max(0.3, dist_to_target_actual * 0.7))

    # Position base behind the target, facing toward it
    target_base_x = target_pos[0] - adaptive_dist*np.cos(target_angle)
    target_base_y = target_pos[1] - adaptive_dist*np.sin(target_angle)

    print(f"Target base pose:({target_base_x, target_base_y}).")
    target_base_theta = target_angle  # Face toward the target
    
    target_base_pose = (target_base_x, target_base_y, target_base_theta)
    
    print(f"Coordinated planning: Moving base to {target_base_pose}")

    # Plan base path to target position
    base_path = run_base_motion_planning(
        robot,
        target_base_pose,
        collision_bodies,
        current_joint_positions,
        seed,
        physics_client_id,
        workspace_bounds=workspace_bounds,
        held_object_id=held_object_id_at_start,
        ee_to_held_object_transform=ee_to_held_object_transform_at_start,
    )

    if base_path is None:
        print("Coordinated planning: Base path planning failed.")
        robot.move_base_to(current_base_pose, physics_client_id)
        robot.set_joints(current_joint_positions)
        return None

    print(f"Coordinated planning: Base path found with {len(base_path)} waypoints.")

    # Move robot to final base position and check if target is reachable
    final_base_pose = base_path[-1]
    robot.move_base_to(final_base_pose, physics_client_id)
    
    try:
        # Check if target is reachable from this base position
        joint_solution = robot.inverse_kinematics(
            target_ee_pose, validate=True, set_joints=False)
        
        # Apply final finger state if specified
        if final_finger_state is not None and joint_solution is not None:
            mutable_js = list(joint_solution)
            if robot.left_finger_joint_idx < len(mutable_js) and \
               robot.right_finger_joint_idx < len(mutable_js):
                mutable_js[robot.left_finger_joint_idx] = final_finger_state
                mutable_js[robot.right_finger_joint_idx] = final_finger_state
                joint_solution = tuple(mutable_js)
        
        # Plan arm path from final base position
        arm_path = run_motion_planning(
            robot,
            current_joint_positions,
            joint_solution,
            collision_bodies,
            seed,
            physics_client_id,
            held_object=held_object_id_at_start,
            ee_to_held_object_transform=ee_to_held_object_transform_at_start
        )

        padded_arm_path: List[JointPositions] = []

        for arm_wp in arm_path:
            if len(list(arm_wp)) == 9:
                continue
            wp7 = list(arm_wp)
            #Insert left finger:
            wf = robot.open_fingers
            wp7.insert(robot.left_finger_joint_idx, wf)
            #Insert right finger:
            wp7.insert(robot.right_finger_joint_idx, wf)

            padded_arm_path.append(wp7)
        
        # Restore robot state
        robot.move_base_to(current_base_pose, physics_client_id)
        robot.set_joints(current_joint_positions)
        
        if arm_path is not None:
            print("Coordinated planning: Succeeded with base + arm movement.")
            return (base_path, padded_arm_path)
        else:
            print("Coordinated planning: Arm planning failed from target base position.")
            return None
            
    except InverseKinematicsError:
        print("Coordinated planning: Target not reachable from planned base position.")
        robot.move_base_to(current_base_pose, physics_client_id)
        robot.set_joints(current_joint_positions)
        return None


