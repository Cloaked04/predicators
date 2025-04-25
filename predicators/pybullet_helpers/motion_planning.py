"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence, List, Tuple, Any

import numpy as np
import pybullet as p
from numpy.typing import NDArray

from predicators import utils
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
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
    base_link_to_held_obj: Optional[NDArray] = None,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.
    """
    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    num_interp = CFG.pybullet_birrt_extend_num_interp

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
            assert base_link_to_held_obj is not None
            world_to_base_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=physics_client_id).com_pose
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     base_link_to_held_obj[0],
                                                     base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                held_object,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=physics_client_id)

    def _extend_fn(pt1: JointPositions,
                   pt2: JointPositions) -> Iterator[JointPositions]:
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
        # orientations as well in the near future.
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
    seed: int,
    physics_client_id: int,
    workspace_bounds: Optional[Tuple[float, float, float, float]]=None,
    ) -> Optional[List[Tuple[float, float, float]]]:
    
    """
    Implements motion planning (BiRRT) for robot with mobile base.

    Args:
        robot: The mobile robot
        target_pose: Target pose as (x,y,theta)
        collision_bodies: Collection of body IDs to avoid
        seed: Random seed
        physics_client_id: PyBullet physics client ID
        workspace_bounds: Optional(min_x, min_y, max_x, max_y) bounds

    Returns:
        List of (x,y,theta) waypoints or None if no path is found
    """

    rng = np.random.default_rng(seed)

    #Get robot's current Pose
    current_pose = robot.get_base_pose(physics_client_id)

    #Worksapce bounds(use default values if not provided)

    if workspace_bounds is None:
        workspace_bounds = (0.0,0.0, 3.0, 3.0) #Why these values as bounds?
    min_x, min_y, max_x, max_y = workspace_bounds

    def _sample_fn() -> Tuple[float, float, float]:
        """Sample a random base pose in the workspace.
           Return a value between the min-max x and y coords and between 0-2pi for theta.
        """

        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        theta = rng.uniform(0, 2*np.pi)
        return (x,y,theta)

    def _extend_fn(pose1: Tuple[float, float, float],
                    pose2: Tuple[float, float, float]) -> Iterator[Tuple[float, float, float]]:

        """
        Generate interpolated poses between pose1 and pose2
        """
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2

        #Calculate distance and steps
        distance = np.sqrt((x2-x1)**2+(y2-y1)**2)
        num_interpolation = max(2, int(distance*10))

        #Normalize angle difference to [-pi, pi]
        angle_diff = ((theta2-theta1+np.pi)%(2*np.pi))

        for i in range(1, num_interpolation+1):
            alpha = i/num_interpolation
            x = x1+alpha*(x2-x1)
            y = y1+alpha*(y2-y1)
            theta = theta1+alpha*angle_diff

            #normalize theta
            theta = theta%(2*np.pi)
            yield (x, y, theta)


    def _collision_fn(pose: Tuple[float, float, float]) -> bool:
        """
        Check if the robot at the given base pose collides with any objects.
        """

        x, y, theta = pose

        if robot.footprint_circle_at(pose).intersects(robot.blocks_table_top(physics_client_id=physics_client_id)):
            return True

        # Save robot's current state
        original_pos, original_orn = p.getBasePositionAndOrientation(
                                        robot.robot_id, physicsClientId=physics_client_id)

        # Set robot to test pose
        z = original_pos[2]     # Maintains current height
        new_position = [x,y,z]
        new_orn = p.getQuaternionFromEuler([0,0,theta])

        #Temporarily move robot to next position to check if it collides with anything
        p.resetBasePositionAndOrientation(
            robot.robot_id, new_position, new_orn,
            physicsClientId=physics_client_id
            )

        # Steps to Check for collisions

        p.performCollisionDetection(physicsClientId=physics_client_id)
        
        collides = False

        for body_id in collision_bodies:
            contact_points = p.getContactPoints(
                robot.robot_id, body_id,
                physicsClientId=physics_client_id
                )

            if len(contact_points)>0:
                collides=True
                break

        p.resetBasePositionAndOrientation(
            robot.robot_id, original_pos, original_orn,
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


    #Use BiRRT for planning

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

    path = birrt.query(current_pose, target_pose)
    return path


    def run_coordiated_motion_planning(
        robot: MobileSingleArmPyBulletRobot,
        target_ee_pose: Pose,
        collision_bodies:Collection[int],
        seed: int,
        physics_client_id:int,
        try_arm_only_first:bool = True,
        workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
        ) -> Optional[Tuple[List[Tuple[float, float, float]], Optional[List[JointPositions]]]]:

        """
        Run co-ordinated motion planning for both base and arm.

        Args:
            robot: The mobile robot
            target_ee_pose: Target end-effector pose
            collision_bodies: collection of body ids to avoid while moving
            seed: Random seed
            physics_client_id: PyBullet physics client id
            try_arm_only_first: try to reach target orientation only by using arm movements first
            workspace_bounds: Optional(min_X, min_Y, max_X, max_y) 

        Returns:
            Tuple of (base_path, arm_path) where base_path is always provided and arm_path may be None if 
            only base movemnt is needed.
        """

        rng = np.random.default_rng(seed)

        #Get robot's current state
        current_base_pose = robot.get_base_pose(physics_client_id)
        current_joint_positions = robot.get_joints()

        # First: check whether target is reachable from current base positions (if try_arm_only_first flag is True)
        if try_arm_only_first:
            try:
                # Try to solve with inverse kinematics
                joint_solutions = robot.inverse_kinematics(
                    target_ee_pose, validate=True, set_joints=False)

                # Check if this solution is collision-free
                arm_path = run_motion_planning(
                    robot,
                    current_joint_positions,
                    joint_solution,
                    collision_bodies,
                    seed,
                    physics_client_id
                    )

                if arm_path is not None:
                    #Target position was reach, able via arm motion only; return it
                    return ([current_base_pose], arm_path)

            except Exception:
                pass

        #Then: find a base position that makes the target reachable
        best_base_pos = None
        best_joint_solution = None
        best_distance = float('inf')

        #try multiple candidate base positions

        for _ in range(10):

            #Sample a base position near the target
            distance = rng.uniform(0.5, 0.8) #Keep some distance from target
            angle = rng.uniform(0, 2*np.pi)

            # Calculate position relative to target
            dx = distance*np.cos(angle)
            dy = distance*np.sin(angle)


            #Save base position
            candidate_x = target_ee_pose.position[0] - dx
            candidate_y = target_ee_pose.position[1] - dy
            candidate_theta = angle

            candidate_base_pose = (candidate_x, candidate_y, candidate_theta)

            #Save current state
            original_pose, original_orn = p.getBasePositionAndOrientation(
                robot.robot_id, physicsClientId=physics_client_id)

            # Temporarily move base to candidate position
            robot.move_base_to(candidate_base_pose, physics_client_id)

            # Check if target is reachable from this base pose
            try:
                # Try to solve with inverse kinematics
                joint_solution = robot.inverse_kinematics(
                    target_ee_pose, validate=True, set_joints=False)

                #Calculate distance from current base to candidate

                pose_distance = np.sqrt(
                    (candidate_x - current_base_pose[0])**2 + 
                    (candidate_y - current_base_pose[1])**2
                )

                # If this is a better solution, save it

                if pos_distance < best_distance:
                    best_distance = pos_distance
                    best_base_pos = candidate_base_pose
                    best_joint_solution = joint_solutions

            except Exception:
                # Not reachable from this base position
                pass

            #Restore original position
            p.resetBasePositionAndOrientation(
                robot.robot_id, physicsClientId=physics_client_id)


            # Return None if no solution found

            if best_base_pos is None:
                return None


            # 3rd Step: Plan base motion to the best positon

            base_path = run_base_motion_planning(
                robot,
                best_base_pos,
                collision_bodies,
                seed,
                physics_client_id,
                workspace_bounds
                )

            if base_path is None:
                return None


            # Step 4: Plan arm motion
            # First move robot to final base position
            original_pos, original_orn = p.getBasePositionAndOrientation(
                robot.robot_id, physicsClientId=physics_client_id)

            robot.move_base_to(best_base_pos, physics_client_id)

            # Plan arm motion

            arm_path = run_motion_planning(
                robot,
                robot, get_joints(), #Current joint position after moving base
                best_joint_solution,
                collision_bodies,
                seed,
                physics_client_id
                )

            # Restore original position

            p.resetBasePositionAndOrientation(
                robot.robot_id, original_pos,
                original_orn, physicsClientId=physics_client_id
                )

            return (base_path, arm_path)



