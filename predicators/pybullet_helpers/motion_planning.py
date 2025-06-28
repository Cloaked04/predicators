"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence, List, Tuple, Any

import sys
import logging

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

logger = logging.getLogger(__name__)


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
    logger.info(f"\n Performing arm motion planning inside FN:run_motion_planning.")

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

    logger.info("\nrun_motion_planning: Checking collisions for start and goal")
    logger.info("\nStart:", initial_positions, "Collision:", _collision_fn(initial_positions))
    logger.info("\nGoal:", target_positions, "Collision:", _collision_fn(target_positions))
    if _collision_fn(target_positions):
        logger.info("\nGoal is in COLLISION, thus planning fails.")
        sys.exit(0)
    logger.info("\nJoint limits low:", current_arm_joint_space.low)
    logger.info("\nJoint limits high:", current_arm_joint_space.high)
    logger.info("\nStart in limits:", np.all(initial_positions >= current_arm_joint_space.low) and np.all(initial_positions <= current_arm_joint_space.high))
    logger.info("\nGoal in limits:", np.all(target_positions >= current_arm_joint_space.low) and np.all(target_positions <= current_arm_joint_space.high))
    

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
    logger.info("Performing base motion planning inside run_base_motion_p")

    rng = np.random.default_rng(seed)

    #Get robot's current pose in simulation.
    current_base_sim_pose = robot.get_base_pose(physics_client_id)

    # Workspace bounds (use default values if not provided)
    if workspace_bounds is None:
        workspace_bounds = (0.0, 0.0, 3.0, 3.0)
    min_x, min_y, max_x, max_y = workspace_bounds

    def _sample_fn(_: Tuple[float, float, float]) -> Tuple[float, float, float]:
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
        num_interpolation = max(2, int(distance*100))

        # Normalize angle difference to [-pi, pi]
        # Normalization of the angle difference is needed to ensure shortest interpolation
        # path across the circle.
        angle_diff = (((theta2-theta1)+np.pi)%(2*np.pi)) - np.pi

        for i in range(1, num_interpolation+1):
            alpha = i/num_interpolation
            x = x1+alpha*(x2-x1)
            y = y1+alpha*(y2-y1)
            theta = wrap_to_pi(theta1+alpha*angle_diff)

            # normalize theta just to keep it in the range [0,2pi).
            #theta = theta%(2*np.pi)
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

        # Angular distance (normalized to [-pi, pi])
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



def _collision_fn(robot: MobileSingleArmPyBulletRobot,
                  collision_bodies: Collection[int],
                  physics_client_id: int,
                  held_object_id: Optional[int] = None,
                  ee_to_held_object_transform: Optional[Tuple[NDArray, NDArray]] = None
                  ) -> bool:
    """
    Helper collision checking function used inside run_coordinated_motion_planning.
    Check if the robot at the given base pose collides with any objects.
    Receives the robot with base and joints set to the values for which collision needs to be checked.

    
    """
    pos = pose.position
    orn = pose.orientation

    # Save robot's current state(simulation state)
    # original_pos, original_orn = p.getBasePositionAndOrientation(
    #                                 robot.robot_id, physicsClientId=physics_client_id)
    # original_joints = robot.get_joints() # These are arm joints

    #If the arm holds an object, get its pose
    original_held_object_pos_orn: Optional[Tuple[NDArray, NDArray]] = None
    if held_object_id is not None:
        original_held_object_pos_orn = p.getBasePositionAndOrientation(
            held_object_id, physicsClientId=physics_client_id)

    # Temporarily set robot to test pose.
    #robot.move_base_to(pose, physics_client_id)
    # Ensure joints are reset as well.
    # if current_arm_positions is not None: # Use specified arm positions if given
    #     robot.set_joints(current_arm_positions)
    # else: # Otherwise, use the ones from before moving the base
    #     robot.set_joints(original_joints)


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
    # in order to get updates values when using get.ContactPoints
    p.performCollisionDetection(physicsClientId=physics_client_id)

    #####################################################################
    #TODO: (i) Figure out how to get body name from body id in this repo# 
    #(ii) BUG/CODE improvement here. In case of collision, should log   #
    #the exact body that the robot collides with and print its name.    #
    #####################################################################

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
    # p.resetBasePositionAndOrientation(
    #     robot.robot_id, original_pos, original_orn,
    #     physicsClientId=physics_client_id
    #     )

    # robot.set_joints(original_joints) # Restore original joints

    if held_object_id is not None and original_held_object_pos_orn is not None:
        p.resetBasePositionAndOrientation(
            held_object_id,
            original_held_object_pos_orn[0],
            original_held_object_pos_orn[1],
            physicsClientId=physics_client_id
        )

    return collides


def wrap_to_pi(angle: float) -> float:
    """Map an angle to (-π, π]."""
    return (angle + np.pi) % (2*np.pi) - np.pi


def _compute_facing_theta(robot_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """Compute theta such that robot faces the target."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]

    theta = np.arctan2(dy, dx)
    #wrap-around since above theta is in [-pi, pi]:
    #theta = (theta+2*np.pi)%(2*np.pi)

    return wrap_to_pi(theta)

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

            print(f"\n[DEBUG] IK SUCCESSFUL!! Found solution:{target_joint_solution}.")
            
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

                padded_arm_path: List[JointPositions] = []

                expected_len = len(robot.arm_joints)
                left_idx = robot.left_finger_joint_idx
                right_idx = robot.right_finger_joint_idx

                for arm_wp in arm_path:
                    wp = list(arm_wp)
                    if len(wp) == expected_len:
                        padded_arm_path.append(wp)
                    elif len(wp) == expected_len - 2:
                        # Insert finger joints at the correct indices
                        wf = robot.open_fingers
                        # Insert the higher index first to not affect the lower index
                        for idx in sorted([left_idx, right_idx], reverse=True):
                            wp.insert(idx, wf)
                        padded_arm_path.append(wp)
                    else:
                        raise ValueError(f"Arm path waypoint has length {len(wp)}, expected {expected_len} (arm/finger joints). Waypoint: {wp}")

                robot.move_base_to(current_base_pose, physics_client_id)
                robot.set_joints(current_joint_positions)
                return ([current_base_pose], padded_arm_path)

        except InverseKinematicsError:
            print("Coordinated Planning: Arm-only IK failed. Moving to base planning.")
            pass

    # Restore robot state before base planning
    robot.move_base_to(current_base_pose, physics_client_id)
    robot.set_joints(current_joint_positions)

    def _base_collision_fn() -> bool:
        """
        Check if the robot at the given base pose collides with any objects.
        Also considers a held object to check for collision if provided.

        NOTE: THIS FUNCTION DOES NOT RETURN THE ROBOT'S POSE, JOINTS TO INITIAL POSITION.
              ENSURE THAT THE CALLING CODE/FN. TAKES CARE OF THIS.
        """
        # Save robot's original full state
        #original_robot_base_pos, original_robot_base_orn = p.getBasePositionAndOrientation(
        #    robot.robot_id, physicsClientId=physics_client_id)
        # current_arm_positions is already passed and represents the arm state to check with

        # If robot is holding an object, save its orientation as well.
        original_held_object_pos_orn: Optional[Tuple[NDArray, NDArray]] = None
        if held_object_id_at_start is not None:
            original_held_object_pos_orn = p.getBasePositionAndOrientation(
                held_object_id_at_start, physicsClientId=physics_client_id)

        # Set robot base to the test pose
        # robot.move_base_to uses p.resetBasePositionAndOrientation
        #robot.move_base_to(pose_to_check, physics_client_id)
        # Ensure arm is in the desired configuration. Recall that arm joint positions are 
        # not connected to base position.
        #robot.set_joints(current_joint_positions)

        # If holding an object, update its pose based on new base and arm state
        if held_object_id_at_start is not None:
            assert ee_to_held_object_transform_at_start is not None
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
                held_object_id_at_start,
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
        if not collides and held_object_id_at_start is not None:
            for body_id in collision_bodies:
                # Don't check collision between held object and robot itself -- LOL!!
                if body_id == robot.robot_id:
                    continue
                if p.getContactPoints(held_object_id_at_start, body_id, physicsClientId=physics_client_id):
                    collides = True
                    break
        
        # Restore original robot base state
        #p.resetBasePositionAndOrientation(
        #    robot.robot_id, original_robot_base_pos, original_robot_base_orn,
        #    physicsClientId=physics_client_id)
        # Restore arm joints (important as resetBasePositionAndOrientation can affect them)
        #robot.set_joints(current_joint_positions) # Or original arm positions if they could change

        # Restore original held object state
        if held_object_id_at_start is not None and original_held_object_pos_orn is not None:
            p.resetBasePositionAndOrientation(
                held_object_id_at_start,
                original_held_object_pos_orn[0],
                original_held_object_pos_orn[1],
                physicsClientId=physics_client_id
            )
        return collides

    
    """
    Compute sampling based full-body IK:

    Determine the ideal radius to sample within (This code is commented below.)
        - set min and max radius values to define the ring to sample values in
            - randomly sample base coordinates, and compute the orientation facing the target
            - determine if IK has solution from the sampled pose
            - if solution, determine if the pose is in collision and find a path from current
              pose to sampled pose
            - Perform arm-only motion planning to get the ee to target pose
    
    -----------------------------------------------------------------------------------------

    To get suitable base positions that satisfy IK and give collision-free paths:

        - Sample a position inside the radius to sample at
        - Move robot to that base position
        - Call IK to determine whether target ee position is reachable from this
          position.
            - If IK returns success, check the returned joint positions for collision
            - Return if true else continue


    """

    # #Distance from target within which no sampling will be done.
    # #Set to 0.6 because most of the samples during monte-carlo 
    # #sampling process were in collision with the table.
    # cutoff_distance = 0.6

    # #Define parameters for the ring/circle to sample within:
    # rad_min = 0.0
    # #rad_limit = 3.0
    # rad_max = 1.0

    # # target x,y
    # target_pos = np.array(target_ee_pose.position[:2])

    # rad_delta = -0.2

    # success_stats_dict = {}

    # #Loop over rings/circle of various sizes to determine the best radius to sample within:

    # #Pause physics rendering:
    # print("Disabling rendering for Monte Carlo experiment...")
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=physics_client_id)

    # for _ in range(1000):

    #     #Sample the maximum radius to sample within:
    #     rad_max = rng.uniform(rad_min, rad_limit)

    #     successful_sample_count = 0

    #     success_stats_dict[rad_max] = []

    #     for attempt in range(1000):

    #         #sample distance, position on the circumference from target:
    #         rad_test = rng.uniform(rad_min, rad_max)
    #         theta = rng.uniform(0.0, 2*np.pi)

    #         #Compute the exact base coordinates for sampled (radius, theta):
            
    #         test_x = target_pos[0]+rad_test*np.cos(theta)
    #         test_y = target_pos[1]+rad_test*np.sin(theta)

    #         #Compute robot's orientation such that robot faces the target
    #         test_orn = _compute_facing_theta([test_x, test_y], target_pos)

    #         #candidate pose:

    #         test_pose = (test_x, test_y, test_orn)
            
    #         # Check if target is reachable from this base pose
    #         try:

    #             #Temporarily move base to candidate position
    #             robot.move_base_to(test_pose, physics_client_id)

    #             # Check IK reachability without validating collisons
    #             # validate = False skips checking for collisions for now.
    #             candidate_joint_solution = robot.inverse_kinematics(
    #                 target_ee_pose, validate=False, set_joints=False)

    #             if candidate_joint_solution is not None:

    #                 successful_sample_count+=1
    #                 #success_stats_dict[rad_max].append(rad_test)
    #                 print(f"Found a sample inside radius {rad_max} at attempt {attempt}.")

    #                 #Restore robot to initial position
    #                 robot.move_base_to(current_base_pose, physics_client_id)
    #                 robot.set_joints(current_joint_positions)
    #                 continue

    #             #Restore robot to initial position in case IK returned None:
    #             robot.move_base_to(current_base_pose, physics_client_id)
    #             robot.set_joints(current_joint_positions)

    #         except InverseKinematicsError:
    #             print(f"Bad sample inside radius {rad_max} at attempt {attempt}. Continuing...")
    #             #Restore robot to initial position
    #             robot.move_base_to(current_base_pose, physics_client_id)
    #             robot.set_joints(current_joint_positions)


    #     success_stats_dict[rad_max] = successful_sample_count

    # # Resume physics rendering
    # print("Re-enabling rendering.")
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=physics_client_id)


    # #Compute stats for each radius value:

    # for rad in success_stats_dict:
    #     print(f"Prob. of finding a successful sample inside radius {rad}: {success_stats_dict[rad]/1000}.")

    # sys.exit(0)

    ###################################################################################################
    ########################-----FULL BODY IK CODE TO BE USED DURING EXECUTION----#####################
    ###################################################################################################

    #Distance from target within which no sampling will be done.
    #Set to 0.6 because most of the samples during monte-carlo 
    #sampling process were in collision with the table.
    cutoff_distance = 0.6

    #Define parameters for the ring/circle to sample within:
    rad_min = 0.0
    #rad_limit = 3.0
    rad_max = 1.0

    # target x,y
    target_pos = np.array(target_ee_pose.position[:2])

    #candidate_base_poses = []
    #candidate_joint_solutions = {}

    #Get a bunch of base positions that allow reaching the target
    for _ in range(base_path_planner_max_tries):

        logger.debug(f"\nSample attempt {_+1}/{base_path_planner_max_tries}.")

        #sample distance, position on the circumference from target:
        rad_test = 0.0

        while rad_test<=cutoff_distance:
            rad_test = rng.uniform(rad_min, rad_max)
        #rad_test = rng.uniform(rad_min, rad_max)
        theta = wrap_to_pi(rng.uniform(0.0, 2*np.pi))

        logger.info(f"\nSampled radius:{rad_test}.")

        #Compute the exact base coordinates for sampled (radius, theta):
            
        test_x = target_pos[0]+rad_test*np.cos(theta)
        test_y = target_pos[1]+rad_test*np.sin(theta)

        #Compute robot's orientation such that robot faces the target
        test_orn = _compute_facing_theta([test_x, test_y], target_pos)

        #candidate pose:

        test_pose = (test_x, test_y, test_orn)

        
        #print(f"[DEBUG]: Sampled base pose -- {test_pose}.")

        # Check if target is reachable from this base pose
        try:

            #Temporarily move base to candidate position
            robot.move_base_to(test_pose, physics_client_id)

            # Check IK reachability
            candidate_joint_solution = robot.inverse_kinematics(
                target_ee_pose, validate=True, set_joints=False)

            if candidate_joint_solution is not None:

                logger.info(f"\n IK SUCCESSFUL. Proceeding to check collision for joint solution: {candidate_joint_solution}.")

                # candidate_base_poses.append(test_pose)
                # #candidate_joint_solutions[test_pose] = candidate_joint_solution

                # #Restore robot to initial position
                # robot.move_base_to(current_base_pose, physics_client_id)
                # robot.set_joints(current_joint_positions)
                # continue

                #Set joints to  soltn. returned by IK:
                robot.set_joints(candidate_joint_solution)
                #Check collison:
                is_colliding = _base_collision_fn()
                if is_colliding:
                    logger.warning(f"\n Collsion check:COLLIDES :-(")
                    robot.move_base_to(current_base_pose, physics_client_id)
                    robot.set_joints(current_joint_positions)
                    continue
                elif not is_colliding:
                    logger.info(f"\n Collision Check: OK!!!")
                    robot.move_base_to(current_base_pose, physics_client_id)
                    robot.set_joints(current_joint_positions)

                    logger.info(f"Attempting base path planning:")

                    base_path = run_base_motion_planning(
                        robot,
                        test_pose,
                        collision_bodies,
                        current_joint_positions,
                        seed,
                        physics_client_id,
                        workspace_bounds=workspace_bounds,
                        held_object_id=held_object_id_at_start,
                        ee_to_held_object_transform=ee_to_held_object_transform_at_start,
                    )

                    if base_path is None:
                        logger.info("Coordinated planning: Base path planning failed.")
                        # Restore initial state
                        robot.move_base_to(current_base_pose, physics_client_id)
                        robot.set_joints(current_joint_positions)
                        continue

                    elif base_path is not None:
                        logger.info(f"\n Coordinated planning: Base path found with {len(base_path)} waypoints.")
                        logger.info(f"\n Proceeding with arm motion planning for joint solution: {candidate_joint_solution}.")

                        #Arm motion planning:
                        final_base_pose = base_path[-1]
                        logger.info(f"Fnial base pose:{final_base_pose}.")
                        robot.move_base_to(final_base_pose, physics_client_id)

                        start_joint_positions = robot.get_joints()

                        logger.debug(f"Initial joint positions:{current_joint_positions}.")
                        logger.debug(f"New joint positions after moving base:{start_joint_positions}.")

                        # target_joint_positions = robot.inverse_kinematics(target_ee_pose, validate=True, set_joints=False)

                        # if target_joint_positions is not None:
                        #     logger.info("Found IK from final base pose during arm motion planning. Proceeding...")
                        #     print(f"Target joint positions are: {target_joint_positions}.")

                        # Ensure target_joint_position has the correct final_finger_state if it was determined.
                        if final_finger_state is not None and candidate_joint_solution is not None:
                            mutable_bjs = list(candidate_joint_solution)
                            # Check if finger indices are valid for the length of best_joint_solution
                            if robot.left_finger_joint_idx < len(mutable_bjs) and \
                               robot.right_finger_joint_idx < len(mutable_bjs):
                                mutable_bjs[robot.left_finger_joint_idx] = final_finger_state
                                mutable_bjs[robot.right_finger_joint_idx] = final_finger_state
                                candidate_joint_solution = tuple(mutable_bjs)
                            else:
                                # This case should be rare if IK solution was valid
                                logger.warning(f"Warning: Finger joint indices out of bounds for \
                                            best_joint_solution. Length: {len(mutable_bjs)}")


                        arm_path = run_motion_planning(robot,
                                                    start_joint_positions,    # Joints before any movement starts
                                                    candidate_joint_solution, # Target joints from IK at final base pose (with fingers updated)
                                                    collision_bodies,
                                                    seed,
                                                    physics_client_id,
                                                    held_object=held_object_id_at_start,
                                                    ee_to_held_object_transform=ee_to_held_object_transform_at_start,
                                                )

                        # Restore initial state after planning is complete
                        robot.move_base_to(current_base_pose, physics_client_id)
                        robot.set_joints(current_joint_positions)

                        if arm_path is None:
                            logger.warning("Coordinated planning: Arm path planning failed after base movement.")
                            continue

                        if arm_path is not None:
                            logger.info("Coordinated planning: Succeeded with arm-only movement after base motion.")

                            padded_arm_path: List[JointPositions] = []

                            expected_len = len(robot.arm_joints)
                            left_idx = robot.left_finger_joint_idx
                            right_idx = robot.right_finger_joint_idx

                            for arm_wp in arm_path:
                                wp = list(arm_wp)
                                if len(wp) == expected_len:
                                    padded_arm_path.append(wp)
                                elif len(wp) == expected_len - 2:
                                    # Insert finger joints at the correct indices
                                    wf = robot.open_fingers
                                    # Insert the higher index first to not affect the lower index
                                    for idx in sorted([left_idx, right_idx], reverse=True):
                                        wp.insert(idx, wf)
                                    padded_arm_path.append(wp)
                                else:
                                    raise ValueError(f"Arm path waypoint has length {len(wp)}, expected {expected_len} (arm/finger joints). Waypoint: {wp}")


                        logger.info(f"Coordinated planning: Arm path found with {len(arm_path)} waypoints.")
                        logger.info("Coordinated planning: Succeeded.")
                        return (base_path, padded_arm_path)


            #Restore robot to initial position in case IK returned None:
            # robot.move_base_to(current_base_pose, physics_client_id)
            # robot.set_joints(current_joint_positions)

        except InverseKinematicsError:
            logger.info(f"\nIK FAILED.Trying again.")
            robot.move_base_to(current_base_pose, physics_client_id)
            robot.set_joints(current_joint_positions)
            continue

        logger.critical(f"\nIKFast failed {base_path_planner_max_tries}. Something's off.")


###############################################################################################################
############################-------END OF FULL BODY IK CODE------##############################################
###############################################################################################################






















