"""Generic controllers for the robots."""
#from typing import Callable, Dict, Sequence, Set, Tuple, cast, Optional, Any
import sys
import logging
import ipdb
from typing import Any, Callable, Collection, DefaultDict, Dict, Iterator, \
    List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast

import numpy as np
from collections import deque
from numpy.typing import NDArray
from gym.spaces import Box
import pybullet as p
from predicators.settings import CFG

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import\
    MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.joint import JointInfo, JointPositions
from predicators.pybullet_helpers.motion_planning import run_coordinated_motion_planning, run_motion_planning
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type
from predicators.pybullet_helpers.link import get_link_state

logger = logging.getLogger(__name__)

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda", "fetch_mobile"}

#Constants for grasp/place offsets:
#Meters above object center for pre-grasp/ finsh grasp
PICK_PRE_GRASP_Z_OFFSET = 0.075
#Meters above target surface for release
PLACE_RELEASE_Z_OFFSET = 0.075


def create_move_end_effector_to_pose_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose, Pose, str]],
    move_to_pose_tol: float,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for moving the end
    effector to a target pose, given a function that takes in the current
    state, objects, and parameters, and returns the current pose and target
    pose of the end effector, and the finger status."""

    logger.info(f"\nStarting with arm motion:")

    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Move end effector to pose option " +
        f"not implemented for robot {robot_name}.")

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # unused
        # Sync the joints.
        assert isinstance(state, utils.PyBulletState)
        
        robot.set_joints(state.joint_positions)
        # First handle the main arm joints.
        # This gives the current and desired end-effector pose and
        # whether the gripper should be "open" or "closed".
        current_pose, target_pose, finger_status = \
            get_current_and_target_pose_and_finger_status(
            state, objects, params)

        # This option currently assumes a fixed end effector orientation.
        # Why: The option is only designed to move the end-effector's position — not its orientation.
        # So when it interpolates between the current and target positions, it reuses the same orientation.
        assert np.allclose(current_pose.orientation, target_pose.orientation)
        orn = current_pose.orientation
        current = current_pose.position
        target = target_pose.position
        # Run IK to determine the target joint positions.
        #1. Compute the vector from current to target position.
        ee_delta = np.subtract(target, current)
        # Reduce the target to conform to the max velocity constraint.
        #2. Compute the length (Euclidean norm) of that vector — how far you are from the goal right now.
        ee_norm = np.linalg.norm(ee_delta)
        logger.warning(f"\nEE norm: {ee_norm}.")
        logger.warning(f"\nMax vel norm: {max_vel_norm}.")
        #3. If the distance is greater than allowed,reduce the step size so that the end-effector
        # moves at most max_vel_norm meters in this time step. Done to keep the motion smooth , 
        # avoiding abrupt jumps.
        if ee_norm > max_vel_norm:
            ee_delta = ee_delta * max_vel_norm / ee_norm
        #4. Compute the new intermediate goal by adding the limited motion vector to current
        # position to get the next position.
        logger.warning(f"\nEE delta: {ee_delta}.")
        dx, dy, dz = np.add(current, ee_delta)
        #5. Make a new Pose object with this new position and original orientation. 
        # This is the next desired pose to be tracked by IK.
        ee_action = Pose((dx, dy, dz), orn)
        # Keep validate as False because validate=True would update the
        # state of the robot during simulation, which overrides physics.
        try:
            # For the panda, always set the joints after running IK because
            # IKFast is very sensitive to initialization, and it's easier to
            # find good solutions on subsequent calls if we are already near
            # a solution from the previous call. The fetch robot does not
            # use IKFast, and in fact gets screwed up if we set joints here.
            joint_positions = robot.inverse_kinematics(ee_action,
                                                       validate=True,
                                                       set_joints=False)

        except InverseKinematicsError:
            raise utils.OptionExecutionFailure("Inverse kinematics failed.")
        # Handle the fingers. Fingers drift if left alone.
        # When the fingers are not explicitly being opened or closed, we
        # nudge the fingers toward being open or closed according to the
        # finger status.
        if finger_status == "open":
            finger_delta = finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_delta = -finger_action_nudge_magnitude
        # Extract the current finger state.
        state = cast(utils.PyBulletState, state)
        
        finger_position = state.joint_positions[robot.left_finger_joint_idx]
        # The finger action is an absolute joint position for the fingers.
        f_action = finger_position + finger_delta
        # Override the meaningless finger values in joint_position.
        # This is required because IK is only responsible for joints and would have 
        # given random values for fingers. Hence, they are controlled/assigned manually.
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        print(f"\nThe retruned joint_positions are of shape:{len(joint_positions)}.")
        # This clipping is needed sometimes for the joint limits.
        action_arr = np.clip(action_arr, robot.action_space.low,
                             robot.action_space.high)
        assert robot.action_space.contains(action_arr)
        return Action(action_arr)

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del memory  # unused
        current_pose, target_pose, _ = \
            get_current_and_target_pose_and_finger_status(
                state, objects, params)
        # This option currently assumes a fixed end effector orientation.
        assert np.allclose(current_pose.orientation, target_pose.orientation)
        current = current_pose.position
        target = target_pose.position
        squared_dist = np.sum(np.square(np.subtract(current, target)))
        return squared_dist < move_to_pose_tol

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=_policy,
                               initiable=lambda _1, _2, _3, _4: True,
                               terminal=_terminal)


def create_change_fingers_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                         Tuple[float, float]],
    max_vel_norm: float,
    grasp_tol: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for changing the
    robot fingers, given a function that takes in the current state, objects,
    and parameters, and returns the current and target finger joint values."""

    assert robot.get_name() in _SUPPORTED_ROBOTS, (
        "Change fingers option not " +
        f"implemented for robot {robot.get_name()}.")

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # unused
        current_val, target_val = get_current_and_target_val(
            state, objects, params)
        f_delta = target_val - current_val
        f_delta = np.clip(f_delta, -max_vel_norm, max_vel_norm)
        f_action = current_val + f_delta
        # Don't change the rest of the joints.
        state = cast(utils.PyBulletState, state)
        target = np.array(state.joint_positions, dtype=np.float32)
        target[robot.left_finger_joint_idx] = f_action
        target[robot.right_finger_joint_idx] = f_action
        # This clipping is needed sometimes for the joint limits.
        target = np.clip(target, robot.action_space.low,
                         robot.action_space.high)
        assert robot.action_space.contains(target)
        return Action(target)

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del memory  # unused
        #ipdb.set_trace()
        current_val, target_val = get_current_and_target_val(
            state, objects, params)
        squared_dist = (target_val - current_val)**2
        return squared_dist < grasp_tol

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=_policy,
                               initiable=lambda _1, _2, _3, _4: True,
                               terminal=_terminal)



# def create_move_base_option(
#     robot: MobileSingleArmPyBulletRobot,
#     name: str,
#     types: Sequence[Type],
#     params_space: Box,
#     get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
#                                                  Tuple[Pose, JointPositions]],
#     base_path: List[Tuple[float, float, float]],
#     target_base_pose = Tuple[float, float, float],
#     move_to_pose_tol: float = 1e-5,
#     vel: float = 0.4,
#     LOOKAHEAD: float = 0.25,
#     wheel_radius: float = 0.065,
#     track_width: float = 0.3748,
#     force: float = 5.0,
#     omega_max: float = 17.4,
#     ) -> ParameterizedOption:
    
#     """Returns a Parameterized Option that performs differential wheel drive for fetch 
#     using Pure-Pursuit.

#     Default values for wheel_radius, track_width(sep. b/w both wheels), omega taken from fetch.urdf.
#     """

#     #Should this receive held_object_id or just determine inside here; the Callable can be used to get it.

#     logger.info(f"\nStarting with Differential-wheel drive:")

#     robot_name = robot.get_name()
#     assert robot_name in _SUPPORTED_ROBOTS, (
#         "Move base option " +
#         f"not implemented for robot {robot_name}.")

#     vel_max = omega_max*wheel_radius

#     assert vel <= vel_max

#     def _initiable(state: State, memory: dict, objs: Sequence[Object],
#                    params: Array) -> bool:
#         memory["path"] = base_path          
#         memory["path_pointer"] = 0
#         memory["target_base_pose"] = target_base_pose
#         return True

#     def _policy(state: State, memory: Dict, objects: Sequence[Object],
#         params: Array) -> Action:

#         #del memory

#         assert memory['path'] is not None, (f"\nNo base path provided.")
#         assert len(memory['path']) !=0, (f"\nNo waypoints in base path.")

#         coarse_thresh = 20*move_to_pose_tol
#         granular_speed = 0.01

#         #Determine robot's current location, define params for Pure-pursuit, compute
#         #vars, and IK

#         if "path_pointer" not in memory:
#             memory["path_pointer"] = 0

#         # ——— STOP‑IF‑DONE: if within tolerance, return zero‑motion ——
#         current_base_pose, current_arm_pose = get_current_base_and_arm_pose(
#             robot, state, objects, params)
#         cur_xy = np.array(current_base_pose[:2])
#         targ_xy = np.array(memory["target_base_pose"][:2])
#         dist = np.linalg.norm(targ_xy - cur_xy)
#         if dist <= move_to_pose_tol:
#             action = Action(np.zeros_like(robot.action_space.low))
#             # freeze arm
#             action._arr[:len(current_arm_pose)] = current_arm_pose
#             # zero both linear and angular motion
#             action.set_base_motion(params=(0.0, 0.0), mode="velocity")
#             return action

#         #Note (Pratyush): currently this function is defined takes in the robot because, the state doesn't have
#         #base_pose object/ doesn't define robot's base pose as a its features.
#         #Additionally, I don't know how to get the robot's current arm pose from state.
#         current_base_pose, current_arm_pose = get_current_base_and_arm_pose(robot, state, objects, params)

#         x, y, theta = current_base_pose

#         #Get coordinates for lookahead
#         path_pointer = memory['path_pointer']
#         base_path_waypoints = memory["path"]

#         #Update path pointer to the next closest waypoint:
#         while path_pointer + 1 < len(base_path_waypoints):
#             # squared distance to NEXT point
#             dx_next = x - base_path_waypoints[path_pointer+1][0]
#             dy_next = y - base_path_waypoints[path_pointer+1][1]
#             squared_dist_next = dx_next*dx_next + dy_next*dy_next

#             # squared distance to CURRENT point
#             dx_current = x - base_path_waypoints[path_pointer][0]
#             dy_current = y - base_path_waypoints[path_pointer][1]
#             squared_dist_current = dx_current*dx_current + dy_current*dy_current

#             if squared_dist_next < squared_dist_current:
#                 path_pointer += 1
#             else:
#                 break

#         memory["path_pointer"] = path_pointer

#         lookahead_idx = min(path_pointer+int(LOOKAHEAD/0.01), len(base_path_waypoints)-1)
#         x_L, y_L, _ = base_path_waypoints[lookahead_idx]

#         #Transform lookahead coords from World to robot's body frame
#         #Rotation matrix for World to robot frame is: [[cos sin],[-sin cos]]
#         dx_W, dy_W = x_L - x,  y_L - y          # world error
#         cos_theta, sin_theta = np.cos(theta), np.sin(theta)
#         dx_B =  cos_theta*dx_W + sin_theta*dy_W 
#         dy_B = -sin_theta*dx_W + cos_theta*dy_W          

#         alpha = np.arctan2(dy_B, dx_B)        # signed heading error

#         #dynamic lookahead:
#         la = max(0.05, min(LOOKAHEAD, dist))
#         vel_adapted = vel* min(dist / LOOKAHEAD, 1.0)
#         kappa = 2.0 * np.sin(alpha) / la

#         if dist <= coarse_thresh:
#             # fine crawl: constant small speed + P‐yaw
#             # print(f"\n Distance less than coarse_threshold")
#             # input()
#             vel_adapted = granular_speed
#             k_yaw = 0.5
#             omega = np.clip(k_yaw * alpha, -omega_max, omega_max)
#         elif dist < LOOKAHEAD:
#             # final‐stage: simple proportional yaw control for small dist
#             # tune k_yaw as needed
#             # print(f"\n Distance less than LOOKAHEAD")
#             # input()
#             k_yaw = 0.5
#             omega = k_yaw * alpha
#         else:
#             # pure pursuit otherwise
#             omega = kappa * vel_adapted


#         # kappa = 2.0 * np.sin(alpha) / LOOKAHEAD
#         # omega = kappa * vel
#         #omega = kappa * vel_adapted
#         omega = float(np.clip(omega, -omega_max, omega_max))

#         # IK:
#         omega_r = np.clip((2*vel+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
#         omega_l = np.clip((2*vel-omega*track_width)/(2*wheel_radius), -omega_max, omega_max)

#         action = Action(np.zeros(len(robot.action_space.low), dtype=float))
#         action._arr = np.array(current_arm_pose)
#         action.set_base_motion(params=(omega_r, omega_l), mode="velocity")
#         #action.set_base_motion(params=(vel_adapted,omega), mode="velocity")

#         return action


#     def _terminal(state: State, memory: Dict, objects: Sequence[Object],
#                   params: Array) -> bool:

#         #Check whether current and target coords are within tolerance limits:
#         current_base_pose, _ = get_current_base_and_arm_pose(robot, state, objects, params)

#         current_coords, target_coords = np.array(current_base_pose[:2]), np.array(memory["target_base_pose"][:2])
#         diff = target_coords - current_coords
#         eucledian_dist = np.linalg.norm(diff)
#         print(f"\nDistance from the target: {eucledian_dist}.")

#         return eucledian_dist < move_to_pose_tol

#     return ParameterizedOption(name,
#                                types=types,
#                                params_space=params_space,
#                                policy=_policy,
#                                initiable=_initiable,
#                                terminal=_terminal)
##########################################################################################################

# def create_move_base_option(
#     robot: MobileSingleArmPyBulletRobot,
#     name: str,
#     types: Sequence[Type],
#     params_space: Box,
#     get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
#                                                  Tuple[Pose, JointPositions]],
#     base_path: List[Tuple[float, float, float]],
#     target_base_pose: Tuple[float, float, float],
#     move_to_pose_tol: float = 0.03,
#     vel: float = 0.4,
#     LOOKAHEAD: float = 0.25,
#     wheel_radius: float = 0.065,
#     track_width: float = 0.3748,
#     force: float = 5.0,
#     omega_max: float = 17.4,
#     orientation_tol: float = 0.05,
#     orientation_gain: float = 1.0,
#     ) -> ParameterizedOption:
    
#     """Returns a Parameterized Option that performs differential wheel drive for fetch 
#     with debug instrumentation to trace the final‐stage behavior.
#     """

#     logger.info(f"\nStarting with Differential-wheel drive:")

#     robot_name = robot.get_name()
#     assert robot_name in _SUPPORTED_ROBOTS, (
#         f"Move base option not implemented for robot {robot_name}.")

#     vel_max = omega_max * wheel_radius
#     assert vel <= vel_max

#     def _initiable(state: State, memory: dict, objs: Sequence[Object], params: Array) -> bool:
#         memory["path"] = base_path          
#         memory["path_pointer"] = 0
#         memory["target_base_pose"] = target_base_pose
#         # initialize debug trace
#         memory["debug_trace"] = []
#         return True

#     def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
#         assert memory.get("path") is not None, "No base path provided."
#         assert len(memory["path"]) != 0, "No waypoints in base path."

#         # thresholds
#         coarse_thresh = 20 * move_to_pose_tol
#         alpha_deadband = 0.05   # radians

#         # 1) get current pose
#         current_base_pose, current_arm_pose = get_current_base_and_arm_pose(
#             robot, state, objects, params)
#         x, y, theta = current_base_pose
#         goal_xy = np.array(memory["target_base_pose"][:2])
#         goal_yaw = memory["target_base_pose"][-1]
#         cur_xy = np.array([x, y])
#         dist_to_goal = np.linalg.norm(goal_xy - cur_xy)
#         yaw_error = ((goal_yaw - theta + np.pi) % (2*np.pi)) - np.pi

#         # 2) STOP-IF-DONE
#         if dist_to_goal <= move_to_pose_tol and abs(yaw_error) <= orientation_tol:
#             # memory["debug_trace"].append({
#             #     "t": state.time,
#             #     "dist":  dist_to_goal,
#             #     "branch":"done",
#             #     "path_ptr": memory["path_pointer"],
#             #     "look_idx": None,
#             #     "alpha_W": None,
#             #     "alpha_G": None,
#             #     "deadband": None,
#             #     "v": 0.0,
#             #     "ω": 0.0
#             # })
#             action = Action(np.zeros_like(robot.action_space.low))
#             action._arr[:len(current_arm_pose)] = current_arm_pose
#             action.set_base_motion((0.0,0.0), "velocity")
#             return action

#         # if dist_to_goal <= move_to_pose_tol:

#         #     if abs(yaw_error) <= orientation_tol:
#         #         action = Action(np.zeros_like(robot.action_space.low))
#         #         action._arr[:len(current_arm_pose)] = current_arm_pose
#         #         action.set_base_motion((0.0, 0.0), "velocity")
#         #         return action

#         #     omega = np.clip(orientation_gain * yaw_error, -omega_max, omega_max)
#         #     action = Action(np.zeros_like(robot.action_space.low))
#         #     action._arr[:len(current_arm_pose)] = current_arm_pose
#         #     action.set_base_motion((0.0, float(omega)), "velocity")
#         #     return action


#         # 3) update path_pointer
#         path_pointer = memory.get("path_pointer", 0)
#         waypoints = memory["path"]
#         while path_pointer + 1 < len(waypoints):
#             dxn = x - waypoints[path_pointer+1][0]
#             dyn = y - waypoints[path_pointer+1][1]
#             dcur = (x - waypoints[path_pointer][0])**2 + (y - waypoints[path_pointer][1])**2
#             if dxn*dxn + dyn*dyn < dcur:
#                 path_pointer += 1
#             else:
#                 break

#         memory["path_pointer"] = path_pointer

#         # 4) choose lookahead waypoint
#         lookahead_idx = min(path_pointer + int(LOOKAHEAD/0.01), len(waypoints)-1)
#         goal_W = np.array(waypoints[lookahead_idx][:2])

#         # 5) compute two heading errors
#         #    W = lookahead, G = true goal
#         def angle_to(pt):
#             return (np.arctan2(pt[1]-y, pt[0]-x) - theta + np.pi) % (2*np.pi) - np.pi

#         alpha_W = angle_to(goal_W)
#         alpha_G = angle_to(goal_xy)

#         # 6) pick branch & compute commands
#         if dist_to_goal <= coarse_thresh:
#             branch = "coarse"
#             v_cmd = 0.0
#             in_dead = abs(alpha_G) < alpha_deadband
#             omega = 0.0 if in_dead else np.clip(0.5 * alpha_G, -omega_max, omega_max)

#         elif dist_to_goal < LOOKAHEAD:
#             branch = "middle"
#             v_cmd = 0.0
#             in_dead = abs(alpha_G) < alpha_deadband
#             omega  = 0.0 if in_dead else np.clip(0.5 * alpha_G, -omega_max, omega_max)

#         else:
#             branch = "far"
#             v_cmd = vel
#             la = max(0.05, min(LOOKAHEAD, dist_to_goal))
#             kappa = 2.0 * np.sin(alpha_W) / la
#             omega = np.clip(kappa * v_cmd, -omega_max, omega_max)
#             in_dead = None

#         # 7) record debug
#         # memory["debug_trace"].append({
#         #     "dist":  dist_to_goal,
#         #     "branch":branch,
#         #     "path_ptr": path_pointer,
#         #     "look_idx": lookahead_idx,
#         #     "alpha_W": float(alpha_W),
#         #     "alpha_G": float(alpha_G),
#         #     "deadband": in_dead,
#         #     "v": float(v_cmd),
#         #     "ω": float(omega),
#         # })

#         print(f"[DBG] d={dist_to_goal:.3f}  br={branch}  ptr={path_pointer}"
#               f"  look={lookahead_idx}  α_W={alpha_W:.2f}  α_G={alpha_G:.2f}"
#               f"  dead={in_dead}  v={v_cmd:.2f}  ω={omega:.2f}")

#         # IK:
#         omega_r = np.clip((2*vel+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
#         omega_l = np.clip((2*vel-omega*track_width)/(2*wheel_radius), -omega_max, omega_max)

#         # 8) build and return Action
#         action = Action(np.zeros(len(robot.action_space.low), dtype=float))
#         action._arr = np.array(current_arm_pose)
#         action.set_base_motion((omega_r, omega_l), "velocity")
#         return action


#     def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
#         current_base_pose, _ = get_current_base_and_arm_pose(
#             robot, state, objects, params)
#         current_coords = np.array(current_base_pose[:2])
#         target_coords = np.array(memory["target_base_pose"][:2])
#         curr_dist = np.linalg.norm(target_coords - current_coords)
#         #current_yaw = current_base_pose[2]
#         #target_yaw = memory["target_base_pose"][2]
#         #curr_yaw_diff  = ((target_yaw - current_yaw  + np.pi) % (2*np.pi)) - np.pi
#         return (curr_dist <= move_to_pose_tol)

#     return ParameterizedOption(
#         name=name,
#         types=types,
#         params_space=params_space,
#         policy=_policy,
#         initiable=_initiable,
#         terminal=_terminal
#     )

# def create_move_base_option(  
#     robot: MobileSingleArmPyBulletRobot,  
#     name: str,  
#     types: Sequence[Type],  
#     params_space: Box,  
#     get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],  
#                                                  Tuple[Pose, JointPositions]],  
#     base_path: List[Tuple[float, float, float]],  
#     target_base_pose: Tuple[float, float, float],  
#     move_to_pose_tol: float = 0.05,  
#     vel: float = 0.4,  
#     LOOKAHEAD: float = 0.25,  
#     wheel_radius: float = 0.065,  
#     track_width: float = 0.3748,  
#     force: float = 5.0,  
#     omega_max: float = 17.4,  
#     orientation_tol: float = 0.1,  
#     orientation_gain: float = 1.0,  
#     ) -> ParameterizedOption:  
      
#     # Even more conservative parameters to prevent spinning  
#     ORIENTATION_ONLY_DISTANCE = 0.10  # Smaller distance threshold  
#     MAX_ORIENTATION_VEL = 0.10        # Much slower angular velocity  
#     ORIENTATION_DEADBAND = 0.01       # Tighter deadband  
#     COMPLETE_STOP_DISTANCE = 0.05     # Distance for complete stop  
      
#     def _initiable(state: State, memory: dict, objs: Sequence[Object], params: Array) -> bool:  
#         memory["path"] = base_path            
#         memory["path_pointer"] = 0  
#         memory["target_base_pose"] = target_base_pose  
#         memory["control_phase"] = "NAVIGATION"  
#         memory["step_count"] = 0  
#         return True  
  
#     def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:  
#         memory["step_count"] += 1  
          
#         # Get current state  
#         current_base_pose, current_arm_pose = get_current_base_and_arm_pose(robot, state, objects, params)  
#         x, y, theta = current_base_pose  
#         goal_xy = np.array(memory["target_base_pose"][:2])  
#         goal_yaw = memory["target_base_pose"][-1]  
#         cur_xy = np.array([x, y])  
#         dist_to_goal = np.linalg.norm(goal_xy - cur_xy)  
#         yaw_error = ((goal_yaw - theta + np.pi) % (2*np.pi)) - np.pi  
  
#         action = Action(np.zeros_like(robot.action_space.low))  
#         action._arr[:len(current_arm_pose)] = current_arm_pose  
  
#         # Phase 1: Complete stop if very close  
#         if dist_to_goal <= COMPLETE_STOP_DISTANCE:  
#             print(f"[STEP {memory['step_count']}] COMPLETE STOP - d={dist_to_goal:.4f} < {COMPLETE_STOP_DISTANCE}")  
#             action.set_base_motion((0.0, 0.0), "velocity")  
#             memory["control_phase"] = "COMPLETE_STOP"  
#             return action  
  
#         # Phase 2: Check if completely done  
#         if dist_to_goal <= move_to_pose_tol and abs(yaw_error) <= orientation_tol:  
#             print(f"[STEP {memory['step_count']}] GOAL REACHED - d={dist_to_goal:.4f}, yaw_err={yaw_error:.4f}")  
#             action.set_base_motion((0.0, 0.0), "velocity")  
#             memory["control_phase"] = "COMPLETE"  
#             return action  
  
#         # Phase 3: Orientation-only mode when position is reached  
#         if dist_to_goal <= ORIENTATION_ONLY_DISTANCE:  
#             memory["control_phase"] = "ORIENTATION_ONLY"  
              
#             # Apply deadband to prevent micro-oscillations  
#             if abs(yaw_error) <= ORIENTATION_DEADBAND:  
#                 omega = 0.0  
#                 print(f"[STEP {memory['step_count']}] ORIENTATION DEADBAND - d={dist_to_goal:.4f}, yaw_err={yaw_error:.4f}, ω=0.0")  
#             else:  
#                 # Very conservative orientation correction  
#                 omega = np.clip(0.5 * yaw_error, -MAX_ORIENTATION_VEL, MAX_ORIENTATION_VEL)  
#                 print(f"[STEP {memory['step_count']}] ORIENTATION ONLY - d={dist_to_goal:.4f}, yaw_err={yaw_error:.4f}, ω={omega:.4f}")

#             # IK:
#             omega_r = np.clip((2*0.0+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
#             omega_l = np.clip((2*0.0-omega*track_width)/(2*wheel_radius), -omega_max, omega_max)  
              
#             # NO forward motion in orientation-only mode  
#             #action.set_base_motion((0.0, omega), "velocity")  

#             action.set_base_motion((omega_r, omega_l), "velocity")

#             return action  
  
#         # Phase 4: Normal navigation (your existing logic with debug prints)  
#         memory["control_phase"] = "NAVIGATION"  
          
#         path_pointer = memory.get("path_pointer", 0)  
#         waypoints = memory["path"]  
#         while path_pointer + 1 < len(waypoints):  
#             dxn = x - waypoints[path_pointer+1][0]  
#             dyn = y - waypoints[path_pointer+1][1]  
#             dcur = (x - waypoints[path_pointer][0])**2 + (y - waypoints[path_pointer][1])**2  
#             if dxn*dxn + dyn*dyn < dcur:  
#                 path_pointer += 1  
#             else:  
#                 break  
  
#         memory["path_pointer"] = path_pointer  
#         lookahead_idx = min(path_pointer + int(LOOKAHEAD/0.01), len(waypoints)-1)  
#         goal_W = np.array(waypoints[lookahead_idx][:2])  
  
#         def angle_to(pt):  
#             return (np.arctan2(pt[1]-y, pt[0]-x) - theta + np.pi) % (2*np.pi) - np.pi  
  
#         alpha_W = angle_to(goal_W)  
#         #alpha_G = angle_to(goal_xy)  
  
         
#         # coarse_thresh = 20 * move_to_pose_tol  
#         # alpha_deadband = 0.05  
  
#         # if dist_to_goal <= coarse_thresh:  
#         #     branch = "coarse"  
#         #     v_cmd = 0.0  
#         #     in_dead = abs(alpha_G) < alpha_deadband  
#         #     omega = 0.0 if in_dead else np.clip(0.5 * alpha_G, -omega_max, omega_max)  
#         # elif dist_to_goal < LOOKAHEAD:  
#         #     branch = "middle"  
#         #     v_cmd = 0.0  
#         #     in_dead = abs(alpha_G) < alpha_deadband  
#         #     omega = 0.0 if in_dead else np.clip(0.5 * alpha_G, -omega_max, omega_max)  
#         # else:  
#         #     branch = "far"  
#         #     v_cmd = vel  
#         #     la = max(0.05, min(LOOKAHEAD, dist_to_goal))  
#         #     kappa = 2.0 * np.sin(alpha_W) / la  
#         #     omega = np.clip(kappa * v_cmd, -omega_max, omega_max)  
#         #     in_dead = None 
#         if dist_to_goal > ORIENTATION_ONLY_DISTANCE:  
#             # Only use the "far" branch logic  
#             v_cmd = vel  
#             la = max(0.05, min(LOOKAHEAD, dist_to_goal))  
#             kappa = 2.0 * np.sin(alpha_W) / la  
#             omega = np.clip(kappa * v_cmd, -omega_max, omega_max)  
              
#             print(f"[STEP {memory['step_count']}] NAVIGATION - d={dist_to_goal:.3f} ptr={path_pointer} "  
#                                     f"look={lookahead_idx} α_W={alpha_W:.2f} v={v_cmd:.2f} ω={omega:.2f}")  
#         else:  
#             # This should never happen due to phase ordering, but safety fallback  
#             print(f"[STEP {memory['step_count']}] NAVIGATION FALLBACK - switching to orientation mode")  
#             action.set_base_motion((0.0, 0.0), "velocity")  
#             return action 
  
#         # Debug print matching your original format  
#         # print(f"[STEP {memory['step_count']}] NAVIGATION - d={dist_to_goal:.3f} br={branch} ptr={path_pointer} "  
#         #       f"look={lookahead_idx} α_W={alpha_W:.2f} α_G={alpha_G:.2f} dead={in_dead} v={v_cmd:.2f} ω={omega:.2f}")

#         # IK:
#         omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
#         omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -omega_max, omega_max)  
          
#         # NO forward motion in orientation-only mode  
#         #action.set_base_motion((0.0, omega), "velocity")  
#         # action.set_base_motion((v_cmd, omega), "velocity")  

#         action.set_base_motion((omega_r, omega_l), "velocity")  
  
#         return action  
  
#     def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:  
#         current_base_pose, _ = get_current_base_and_arm_pose(robot, state, objects, params)  
#         current_coords = np.array(current_base_pose[:2])  
#         target_coords = np.array(memory["target_base_pose"][:2])  
#         curr_dist = np.linalg.norm(target_coords - current_coords)  
          
#         # Check both position and orientation  
#         current_yaw = current_base_pose[2]  
#         target_yaw = memory["target_base_pose"][2]  
#         curr_yaw_diff = ((target_yaw - current_yaw + np.pi) % (2*np.pi)) - np.pi  
          
#         is_done = (curr_dist <= move_to_pose_tol) and (abs(curr_yaw_diff) <= orientation_tol)  
          
#         if is_done:  
#             print(f"[TERMINAL] SUCCESS - Position error: {curr_dist:.4f}, Orientation error: {abs(curr_yaw_diff):.4f}")  
          
#         return is_done  
  
#     return ParameterizedOption(  
#         name=name,  
#         types=types,  
#         params_space=params_space,  
#         policy=_policy,  
#         initiable=_initiable,  
#         terminal=_terminal  
#     )

def create_move_base_option(
    robot: MobileSingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
                                            Tuple[Pose, JointPositions]],
    base_path: List[Tuple[float, float, float]],
    target_base_pose: Tuple[float, float, float],
    move_to_pose_tol: float = 0.04,     # 5 cm
    vel: float = 0.2,                   # nominal cruise speed
    LOOKAHEAD: float = 0.25,
    wheel_radius: float = 0.065,
    track_width: float = 0.3748,
    force: float = 5.0,                 # (unused here; wheel control path would use it)
    omega_max: float = 17.4,            # wheel joint limit (rad/s)
    orientation_tol: float = 0.04,     # ~5°
    orientation_gain: float = 2.0,      # yaw P gain
) -> ParameterizedOption:

    # Phase thresholds / gating
    # ORIENTATION_ONLY_DISTANCE = 0.10    # within 10 cm: rotate in place
    # e.g., 0.06–0.07 m
    ORIENTATION_ONLY_DISTANCE = max(0.06, move_to_pose_tol + 0.01)
    ORIENTATION_DEADBAND      = 0.02    # ~0.6° deadband for micro-oscillation
    COMPLETE_STOP_DISTANCE    = move_to_pose_tol

    # Convert wheel limits -> base limits (use these for clipping v, ω)
    v_max = wheel_radius * omega_max
    w_max = 2.0 * wheel_radius * omega_max / track_width

    def _initiable(state: State, memory: dict, objs: Sequence[Object], params: Array) -> bool:
        memory["path"] = base_path
        memory["path_pointer"] = 0
        memory["target_base_pose"] = target_base_pose
        memory["control_phase"] = "NAVIGATION"
        memory["step_count"] = 0
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        memory["step_count"] += 1

        # Pose
        # ipdb.set_trace()
        (x, y, theta), arm_q = get_current_base_and_arm_pose(robot, state, objects, params)
        goal_xy = np.array(memory["target_base_pose"][:2])
        goal_yaw = float(memory["target_base_pose"][2])

        cur_xy = np.array([x, y])
        dist_to_goal = float(np.linalg.norm(goal_xy - cur_xy))
        yaw_error = float((goal_yaw - theta + np.pi) % (2*np.pi) - np.pi)

        # Build action shell
        action = Action(np.zeros_like(robot.action_space.low))
        action._arr[:len(arm_q)] = arm_q

        # 1) Hard stop region
        if dist_to_goal <= COMPLETE_STOP_DISTANCE and abs(yaw_error) <= orientation_tol:
            action.set_base_motion((0.0, 0.0), "velocity")
            memory["control_phase"] = "COMPLETE"
            print(f"[STEP {memory['step_count']}] GOAL REACHED - d={dist_to_goal:.4f}, yaw={yaw_error:.4f}")
            return action

        # 2) Orientation-only (rotate in place)
        if dist_to_goal <= ORIENTATION_ONLY_DISTANCE:
            memory["control_phase"] = "ORIENTATION_ONLY"
            if abs(yaw_error) <= ORIENTATION_DEADBAND:
                #v_cmd, omega = 0.0, 0.0
                # slow nudge
                v_cmd = min(0.15, 1.5 * dist_to_goal)

                # small steering toward goal
                bearing = np.arctan2(goal_xy[1]-y, goal_xy[0]-x)
                yaw_to_goal = ((bearing - theta + np.pi) % (2*np.pi)) - np.pi
                omega = np.clip(0.5 * yaw_to_goal, -w_max*0.2, w_max*0.2)

                v_cmd = float(np.clip(v_cmd, 0.0, v_max))
                omega = float(np.clip(omega, -w_max, w_max))

                # print(f"[STEP {memory['step_count']}] ORIENT DEADBAND - d={dist_to_goal:.3f}, yaw={yaw_error:.3f}")
            else:
                # proportional yaw control; clip with base yaw limit
                omega = float(np.clip(orientation_gain * yaw_error, -w_max, w_max))
                v_cmd = 0.0
                # print(f"[STEP {memory['step_count']}] ORIENT ONLY - d={dist_to_goal:.3f}, yaw={yaw_error:.3f}, ω={omega:.3f}")

            # IK:
            omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -w_max, w_max)
            omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -w_max, w_max) 

            action.set_base_motion((omega_r, omega_l), "velocity")

            # action.set_base_motion((v_cmd, omega), "velocity")
            return action

        # 3) Navigation (pure pursuit-like on lookahead, with gating)
        memory["control_phase"] = "NAVIGATION"

        # advance waypoint pointer if closer to next
        path_pointer = int(memory.get("path_pointer", 0))
        waypoints = memory["path"]
        while path_pointer + 1 < len(waypoints):
            nx, ny = waypoints[path_pointer + 1][:2]
            cx, cy = waypoints[path_pointer][:2]
            if (x - nx)**2 + (y - ny)**2 < (x - cx)**2 + (y - cy)**2:
                path_pointer += 1
            else:
                break
        memory["path_pointer"] = path_pointer

        # lookahead waypoint
        lookahead_idx = min(path_pointer + max(1, int(LOOKAHEAD/0.01)), len(waypoints) - 1)
        wx, wy = waypoints[lookahead_idx][:2]

        # heading to lookahead
        alpha_W = float((np.arctan2(wy - y, wx - x) - theta + np.pi) % (2*np.pi) - np.pi)

        # curvature -> ω, plus velocity gating by heading and distance
        la = max(0.05, min(LOOKAHEAD, dist_to_goal))
        v_cmd = vel
        v_cmd *= float(np.exp(-0.8 * abs(alpha_W)))       # slow if misaligned
        v_cmd = float(min(v_cmd, 1.5 * dist_to_goal))     # don’t overdrive when close
        v_cmd = float(np.clip(v_cmd, 0.0, v_max))

        kappa = 2.0 * np.sin(alpha_W) / la
        omega = float(np.clip(kappa * v_cmd, -w_max, w_max))

        # print(f"[STEP {memory['step_count']}] NAV - d={dist_to_goal:.3f} ptr={path_pointer} look={lookahead_idx} "
              # f"αW={alpha_W:.2f} v={v_cmd:.2f} ω={omega:.2f}")

        # action.set_base_motion((v_cmd, omega), "velocity")

        # IK:
        omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -w_max, w_max)
        omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -w_max, w_max) 

        action.set_base_motion((omega_r, omega_l), "velocity")



        return action

    def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        (x, y, th), _ = get_current_base_and_arm_pose(robot, state, objects, params)
        gx, gy, gth = memory["target_base_pose"]
        pos_ok = (np.hypot(gx - x, gy - y) <= move_to_pose_tol)
        yaw_ok = (abs((gth - th + np.pi) % (2*np.pi) - np.pi) <= orientation_tol)
        if pos_ok and yaw_ok:
            print(f"[TERMINAL] SUCCESS - Position error: {np.hypot(gx - x, gy - y):.4f}, "
                  f"Orientation error: {abs((gth - th + np.pi) % (2*np.pi) - np.pi):.4f}")
        return pos_ok and yaw_ok

    return ParameterizedOption(
        name=name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )















#############################################################################################

def execute_coordinated_path(
    robot: MobileSingleArmPyBulletRobot,
    base_path:List[Tuple[float, float, float]],
    arm_path: List[JointPositions],
    physics_client_id: int,
    ) -> List[Action]:
    
    """
    Generate a sequence of low-level Action objects to execute a coordinated
    base path. Intuitively, think of this function as the next step after you have 
    outputs from motion planning functions in motion_planning.py. The path generated in those functions
    are converted to Actions by this function.

    Args:
        robot: The mobile robot instance.
        base_path: List of (x,y,theta) waypoints for the base. First element should be current pose.
        arm_path: List of Joint positions.
        physics_client_id: PyBullet physics client ID.

    Returns:
        List of Action objects for execution by the environment.
    """

    actions: List[Action] = []
    
    # num_arm_finger_joints = len(robot.joint_lower_limits[:-2]) # This was arm joints only, excluding fingers.

    # Get the number of arm + finger joints from robot.arm_joints.
    # robot.get_joints() returns positions for robot.arm_joints which includes fingers.
    # Note that there are two get_joints function in the repo: one defined in joints.py
    # and the other defined in single_arm.py. Here, we use the one defined for the robot
    # which calls get_joint_positions in joint.py and arm_joints calls get_joints() function
    # in joints.py.
    num_controllable_arm_joints = len(robot.arm_joints)


    #Get initial arm joint positions
    current_arm_joint_positions = robot.get_joints()
    assert len(current_arm_joint_positions) == num_controllable_arm_joints, \
                        "Mismatch in current arm joint dimensions."

    #Generate base movement actions
    #The arm will be held at its current_arm_joint_positions during base movement.

    # Ensure base_path is not empty
    if base_path and len(base_path) > 0: 
        # Gets (x,y,theta)
        current_robot_base_pose_xytheta = robot.get_base_pose(physics_client_id)
        # If base_path has only one point and it's effectively the current pose, no base actions needed.
        if len(base_path) == 1 and np.allclose(base_path[0], current_robot_base_pose_xytheta, atol=1e-5):
             print("Base path is just the current pose, no base movement actions generated.")
        else:
            # The first waypoint in base_path is assumed to be the current/starting pose.
            # We generate actions to move from base_path[i] to base_path[i+1].
            # So, we iterate up to len(base_path) - 1, using base_path[i+1] as the target.
            print(f"Generating {len(base_path) -1 } base movement actions from path of length {len(base_path)}...")
            for i in range(len(base_path) -1):
                # Target is the next waypoint
                target_x, target_y, target_theta = base_path[i+1]
                action_arr = np.zeros_like(robot.action_space.low) 
                
                #Arm position stays fixed during base motion
                action_arr[:num_controllable_arm_joints] = current_arm_joint_positions

                #logging.debug(f"Creating Action: arr length = {len(action_arr)}, arr = {action_arr}")

                action = Action(action_arr.copy())
                action.set_base_motion((target_x, target_y, target_theta), mode="smooth_position") 
                actions.append(action)

    #Second: Generate Arm movement actions
    #The base is assumed to be at the final base_path waypoint after base actions.

    if arm_path:
        print(f"Generating {len(arm_path)} arm waypoint actions...")
        for i, target_arm_joint_positions_waypoint in enumerate(arm_path):
            if len(target_arm_joint_positions_waypoint)!=num_controllable_arm_joints:
                raise ValueError(
                        f"Arm path waypoint {i} has length {len(target_arm_joint_positions_waypoint)},"
                        f"expected {num_controllable_arm_joints} (arm/finger joints)."
                        )

            action_arr = np.zeros_like(robot.arm_joints, dtype=float)
            action_arr[:num_controllable_arm_joints] = target_arm_joint_positions_waypoint

            #print(f"[DEBUG] Creating Action: arr length = {len(action_arr)}, arr = {action_arr}")            

            #We don't deal with finger values/states here. It's assumed that the waypoints already have
            #the appropriate values based on the option that invoked planning.
            action = Action(action_arr.copy())
            actions.append(action)

            #For conceptual clarity so that currenT_arm_joint_positions don't get stale
            current_arm_joint_positions = target_arm_joint_positions_waypoint

    print(f"Generated total {len(actions)} low-level actions.")
    return actions


#---------------------------Creating Options for Pick/Place motion---------------------------


def create_coordinated_motion_option(
    robot: MobileSingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_target_ee_pose: Callable[[State, Sequence[Object], Array], Pose],
    #block_to_pick_id: int,
    physics_client_id: int,
    try_arm_only_first: bool = True,
    final_finger_state_fn: Optional[Callable[[State, Sequence[Object], Array], Optional[float]]]=None,
    grabbable_object_type: Optional[Type] = None,
    initiable_fn: Optional[Callable[[State, Dict, Sequence[Object], Array], bool]] = None
    ) -> ParameterizedOption:
    
    """
    Create a ParameterizedOption for performing coordinated motion planning and execution.
    The policy plans the full motion on the first call and then executes one from the planned sequence calls.
    """

    def _policy(state:State, memory: Dict, objects:Sequence[Object], params: Array) -> Action:
        if "actions" not in memory or not memory["actions"] or memory.get("current_action_idx",0)==0:
            #Get target end-effector pose
            target_ee_pose = get_target_ee_pose(state, objects, params)
            logger.critical(f"\nTarget ee pose: {target_ee_pose}.")

            #Determine final finger state for this specific option execution
            current_final_finger_state = None
            if final_finger_state_fn:
                current_final_finger_state = final_finger_state_fn(state, objects, params)

            #Get collision bodies (all bodies except robot)
            # - get all body ids first,
            # - then remove the robot's id to get body id of objects that robot can collide with
            all_body_ids = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                            for i in range(p.getNumBodies(physicsClientId=physics_client_id))]

            collision_bodies = [body_id for body_id in all_body_ids if body_id !=robot.robot_id]
            #collision_bodies = collision_bodies[:len(collision_bodies)-1]

            logger.debug(f"\n List of collision bodies at pick option: {collision_bodies}.")
            #sys.exit(0)

            # Synchronize robot state with current for planning.
            # Reset the PyBullet robot to reflect the joint positions and base pose in the symbolic state
            # for planning.

            # Save robot's current base pose and orientation
            current_sim_base_pose_before_sync = robot.get_base_pose(physics_client_id)
            current_sim_joint_positions_before_sync = robot.get_joints()

            assert isinstance(state, utils.PyBulletState)
            
            #Set robot base pose from state if it's a PyBulletState with base_pose
            if hasattr(state, 'base_pose') and state.base_pose is not None:
                #Assuming state.base_pose is (x,y, theta)
                #Note that move_base_to transports the robot to the given position directly.
                robot.move_base_to(state.base_pose, physics_client_id)
            robot.set_joints(state.joint_positions)

            #Determine held_object_id_at_start and its transform
            held_object_id_at_start: Optional[int] = None
            ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None

            symbolic_held_object: Optional[Object] = None

            # Go over all objects and check if they are held by checking if the value is >0.5
            # and if so set that as the symbolic_held_object assuming only one object can be held at
            # a time.
            if grabbable_object_type is not None:
                logger.info("\nLooking for held object.")
                for obj_in_state in state:
                    if obj_in_state.is_instance(grabbable_object_type):
                        try:
                            if state.get(obj_in_state, "held") > 0.5:
                                symbolic_held_object = obj_in_state
                                break 
                        except (KeyError, ValueError): # Feature "held" might not exist for all grabbable types
                            pass

            logger.info(f"\nHeld object is:{symbolic_held_object}.")
            
            # Now, try to get the pybullet ID using the robot's internal _held_obj_id,
            # which should be managed by the PyBulletEnv grasping logic.
            if symbolic_held_object is not None:
                if hasattr(robot, '_held_obj_id') and isinstance(robot, PyBulletEnv) and robot._held_obj_id is not None:
                    # We have a symbolic held object, and the PyBulletEnv robot instance also reports a held_obj_id.
                    # So we have it for future motion planning use.
                    held_object_id_at_start = robot._held_obj_id 
                    # We assume that if symbolic says held, and pybullet env says held, they are the same.
                    logger.info(f"Option '{name}': Symbolic state indicates '{symbolic_held_object.name}' is held. "
                          f"Using PyBullet ID {held_object_id_at_start} from robot instance (PyBulletEnv._held_obj_id).")

                    # Calculate the transform from end-effector to this held object
                    # Ensure robot is in the state specified by `state` for this calculation

                    #1. Get EE pose in world frame.
                    ee_world_pose = get_link_state(robot.robot_id, robot.end_effector_id, physics_client_id).pose
                    #2. Get object pose in world frame.
                    obj_world_pos, obj_world_orn = p.getBasePositionAndOrientation(held_object_id_at_start, physics_client_id)
                    obj_world_pose = Pose(obj_world_pos, obj_world_orn)

                    # Transform: T_ee_obj = (T_world_ee)^-1 * T_world_obj
                    # Pose of object relative to gripper frame.
                    # This can now be used when the base or arm moves, you can move the object by applying
                    # this fixed transform from the updated gripper pose.
                    ee_T_obj = ee_world_pose.invert().multiply(obj_world_pose)

                    ee_to_held_object_transform_at_start = (
                        np.array(ee_T_obj.position, dtype=np.float32),
                        np.array(ee_T_obj.orientation, dtype=np.float32)
                    )
                else:
                    logger.critical(f"Warning: Option '{name}': Symbolic state says '{symbolic_held_object.name}' is held, "
                          "but robot instance (PyBulletEnv) does not report a _held_obj_id or is not a PyBulletEnv. "
                          "Cannot get PyBullet ID for held object planning.")
                    sys.exit(0)
            elif name == "PlaceObject": # If it's a place action, something should ideally be held.
                 logger.critical(f"Warning: Option '{name}' called, but no symbolically held grabbable object found in state.")
                 sys.exit(0)


            # Remove the identified held_object_id_at_start from the set of collision bodies.
            if held_object_id_at_start is not None:
                collision_bodies = [b for b in collision_bodies if b != held_object_id_at_start]

            planning_rng = np.random.default_rng(CFG.pybullet_rng_seed)

            # Store original sim state to restore after planning if needed
            # This is important because run_coordinated_motion_planning itself also modifies state
            # We want the robot to be in the `state` specified by the option call before planning.
            # The set_joints and move_base_to above achieve this.

            logger.info(f"Calling run_coordinated_motion_planning to get sequence of actions.")

            result = run_coordinated_motion_planning(
                        robot=robot,
                        target_ee_pose=target_ee_pose,
                        collision_bodies=collision_bodies, 
                        seed=CFG.seed, 
                        physics_client_id=physics_client_id,
                        try_arm_only_first=try_arm_only_first,
                        rng=planning_rng,
                        final_finger_state=current_final_finger_state,
                        held_object_id_at_start=held_object_id_at_start,
                        ee_to_held_object_transform_at_start=ee_to_held_object_transform_at_start
                        )
            
            # After planning, restore the robot to the state it was in when the option policy was first called,
            # as the execution of the planned path will occur from this state.
            # The planners (run_coordinated_motion_planning, run_motion_planning, run_base_motion_planning)
            # should ideally restore the robot state to what it was before they were called.
            # The `robot.set_joints(state.joint_positions)` and potential `robot.move_base_to(state.base_pose, ...)`
            # at the start of this policy sets the robot up for planning.
            # The planners themselves also do internal state saving/restoring for their sub-routines.
            # The final restoration inside run_coordinated_motion_planning should bring it back to its pre-planning state.


            if result is None:
                # Planning failed, restore robot to state at option call to be safe before throwing error.
                if hasattr(state, 'base_pose') and state.base_pose is not None:
                     robot.move_base_to(state.base_pose, physics_client_id)
                else: # Fallback to sim state before this policy's sync
                     robot.move_base_to(current_sim_base_pose_before_sync, physics_client_id)
                robot.set_joints(current_sim_joint_positions_before_sync)
                raise utils.OptionExecutionFailure(f"{name} planning failed for target {target_ee_pose}")

            base_path, arm_path = result

            memory["actions"] = execute_coordinated_path(
                                robot=robot,
                                base_path=base_path,
                                arm_path=arm_path,
                                physics_client_id=physics_client_id
                                )
            #If no actions were generated(eg. already at target)
            if not memory["actions"]:
                raise utils.OptionExecutionFailure(f"{name} resulted in an empty action sequence.")

            memory["current_action_idx"] = 0

        #Get the next action in the sequence
        action_idx = memory["current_action_idx"]
        if action_idx >= len(memory["actions"]):
            raise utils.OptionExecutionFailure(f"{name} policy called after completion.")

        action = memory["actions"][action_idx]
        memory["current_action_idx"]+=1

        return action


    def _terminal(state:State, memory:Dict, objects: Sequence[Object], params: Array) -> bool:
        if "current_action_idx" not in memory or "actions" not in memory:
            #Not started or planning failed before populating memory
            return False
        # Terminal if all planned actions have been executed
        return memory["current_action_idx"] >= len(memory["actions"])


    # Use the provided initiable_fn or the default
    current_initiable = initiable_fn if initiable_fn is not None else lambda _1, _2, _3, _4: True

    return ParameterizedOption(
            name=name,
            types=types,
            params_space=params_space,
            policy=_policy,
            initiable=current_initiable,  # Use the determined initiable
            terminal=_terminal
            )


def create_pick_object_option(
    robot: MobileSingleArmPyBulletRobot,
    robot_type: Type,
    object_to_pick_type: Type,
    physics_client_id: int,
    #block_to_pick_id: int,
    grasp_height_offset: float = PICK_PRE_GRASP_Z_OFFSET,
    #Common grasp orientation: Pointing down -pi/2
    grasp_euler_orn: Tuple[float, float, float] = None,
    initiable_fn: Optional[Callable[[State, Dict, Sequence[Object], Array], bool]] = None
) -> ParameterizedOption:

    """
    Creates a ParameterizedOption for picking a specified object.
    """

    # param_space could be used to control small variations in grasp
    # if needed; currently empty;
    params_space = Box(low = np.array([]), high = np.array([]), dtype=np.float32)

    def _get_target_ee_pose_for_pick(state: State, objects: Sequence[Object], params: Array) -> Pose:
        del params
        #object[0] is robot, object[1] is target_obj
        _robot_obj, target_obj = objects

        obj_x = state.get(target_obj, "pose_x")
        obj_y = state.get(target_obj, "pose_y")
        obj_z = state.get(target_obj, "pose_z")

        #logger.critical(f"Position of object given as assessed from state inside pick option definition: \
        #                                                                                ({obj_x, obj_y, obj_z}).")

        
        ee_grasp_pos = [obj_x,
                        obj_y , 
                        obj_z + grasp_height_offset]

        # return Pose(ee_grasp_pos, ee_grasp_orn_quat)
        return Pose(ee_grasp_pos, robot._ee_home_pose.orientation)


    def _get_final_finger_state_for_pick(state: State, objects: Sequence[Object], params: Array) -> float:
        del state, objects, params
        return robot.open_fingers


    return create_coordinated_motion_option(
            robot=robot,
            name="PickObject",
            types=[robot_type, object_to_pick_type],
            params_space=params_space,
            get_target_ee_pose=_get_target_ee_pose_for_pick,
            #block_to_pick_id=block_to_pick_id,
            physics_client_id=physics_client_id,
            try_arm_only_first=True,
            final_finger_state_fn=_get_final_finger_state_for_pick,
            grabbable_object_type=object_to_pick_type,
            initiable_fn=initiable_fn
            )

def create_place_object_option(
    robot: MobileSingleArmPyBulletRobot,
    robot_type: Type,
    location_type: Type,
    object_to_place_type: Type,  # This is the type of object being placed
    physics_client_id:int,
    place_height_offset: float = PLACE_RELEASE_Z_OFFSET,
    place_euler_orn: Tuple[float, float, float] = (0, -np.pi/2, 0),
    initiable_fn: Optional[Callable[[State, Dict, Sequence[Object], Array], bool]] = None
) -> ParameterizedOption:
    
    """
    Creates ParamterizedOption for placing the currently held object at a location.
    The location_obj is expected to have 'pose_x', 'pose_y', 'pose_z' feature.
    representing the target placement spot (e.g., center of a region, top of another object.)
    """

    logger.info("Planning Place action.")

    params_space = Box(low=np.array([]), high=np.array([]), dtype=np.float32)


    def _get_target_ee_pose_for_place(state: State, objects: Sequence[Object], params: Array) -> Pose:
        del params

        _robot_obj, target_loc_obj = objects

        loc_x = state.get(target_loc_obj, "pose_x")
        loc_y = state.get(target_loc_obj, "pose_y")
        loc_z = state.get(target_loc_obj, "pose_z")

        #Target EE position for releasing (e.g., slightly above the target location.)

        ee_release_pos = [loc_x, loc_y, loc_z+place_height_offset]
        ee_relase_orn_quat = p.getQuaternionFromEuler(place_euler_orn)

        return Pose(ee_release_pos, ee_relase_orn_quat)


    def _get_final_finger_state_for_place(state: State, objects: Sequence[Object], params: Array) -> float:
        del state, objects, params
        return robot.open_fingers


    return create_coordinated_motion_option(
            robot=robot,
            name="PlaceObject",
            # For Place, types might be [robot_type, location_type] if the held object is implicit,
            # or [robot_type, held_object_type, location_type] if explicit.
            # The current `objects` sequence passed to `get_target_ee_pose_for_place`
            # implies `_robot_obj, target_loc_obj`. The held object is not explicitly passed as a parameter here.
            types=[robot_type, location_type], 
            params_space=params_space,
            get_target_ee_pose=_get_target_ee_pose_for_place,
            physics_client_id=physics_client_id,
            try_arm_only_first=True, 
            final_finger_state_fn=_get_final_finger_state_for_place,
            grabbable_object_type=object_to_place_type,
            initiable_fn=initiable_fn
        )
    


############################################################################################################
#------------------New options that perform planning internally as well.-----------------------------------#
############################################################################################################

def create_arm_motion_planning_option(
    name: str,
    robot: SingleArmPyBulletRobot,
    types: Sequence[Type],
    params_space: Box,
    physics_client_id: int,
    initial_joint_positions: JointPositions,
    z_func: Union[Callable[[float], float], float],
    home_orn: Sequence[float],
    collision_bodies: Collection[int],
    seed: int,
    held_obj_id: Optional[int] = None,
    base_link_to_held_object: Optional[NDArray] = None,
) -> ParameterizedOption:
    """
    Create a ParameterizedOption that allows running arm_motion_planning using
    run_motion_planning function in the background and the policy returns one
    waypoint at a time for simulation.

    Flow:
      - On first policy call, we plan and cache a sequence of arm joint waypoints.
      - Each subsequent call returns the next Action (arm joints only).
      - Base is untouched by this option.
    """

    def _plan_once_and_cache_actions(robot: MobileSingleArmPyBulletRobot, state: State, objects: Sequence[Object], memory: Dict) -> None:

        filtered_collision_bodies = collision_bodies
        if held_obj_id is not None:
            # Exclude the held object from obstacle set
            filtered_collision_bodies = [b for b in collision_bodies if b != held_obj_id or b!=0]

        waypoints: Optional[Sequence[JointPositions]] = None
        ipdb.set_trace()
        if "Grasp" in name or "Stack" in name:

            _, block = objects
    
            block_x, block_y, block_z = (state.get(block, "pose_x"),
                                         state.get(block, "pose_y"),
                                         state.get(block, "pose_z"))
            if callable(z_func):
                target_z = z_func(block_z)
            else:
                target_z = z_func

            target_ee_position = (block_x, block_y, target_z)
            target_ee_pose = Pose(position=target_ee_position, orientation=home_orn)

            initial_left_finger_val = initial_joint_positions[robot.left_finger_joint_idx]
            initial_right_finger_val = initial_joint_positions[robot.right_finger_joint_idx]
            world_robot_base_pose = robot.get_base_pose(physics_client_id)
            world_robot_joint_positions = robot.get_joints()
            simulator_robot_base_pose = state.base_pose
            #Move robot to simulator's base pose for ik:
            robot.move_base_to(target_pose=simulator_robot_base_pose, physics_client_id=physics_client_id)
            print(f"Calling IK for arm motion planning for target_ee_pose: {target_ee_pose}.")
            input()
            try:
                target_joint_positions = robot.inverse_kinematics(target_ee_pose, validate=False, set_joints=False)
                if target_joint_positions is not None:
                    target_joint_positions[robot.left_finger_joint_idx] = initial_left_finger_val
                    target_joint_positions[robot.right_finger_joint_idx] = initial_right_finger_val
                    waypoints = run_motion_planning(
                                                    robot=robot,
                                                    initial_positions=initial_joint_positions,
                                                    target_positions=target_joint_positions,
                                                    collision_bodies=filtered_collision_bodies,
                                                    seed=seed,
                                                    physics_client_id=physics_client_id,
                                                    held_object=held_obj_id,
                                                    base_link_to_held_object=base_link_to_held_object, 
                                                    )
                    #Reset robot to world base pose:
                    robot.move_base_to(target_pose=world_robot_base_pose, physics_client_id=physics_client_id)
                    robot.set_joints(world_robot_joint_positions)
            except InverseKinematicsError:
                robot.move_base_to(target_pose=world_robot_base_pose, physics_client_id=physics_client_id)
                robot.set_joints(world_robot_joint_positions)
                raise utils.OptionExecutionFailure(f"\nInverse Kinematics failed.")
            
            

        elif "OnTable" in name:

            _, table = objects

            table_x, table_y = (state.get(table, "pose_x"),
                                state.get(table, "pose_y"))

            if not callable(z_func):
                target_z = z_func

            x_workspace = (table_x-0.125, table_x+0.125)
            y_workspace = (table_y-0.2, table_y+0.2)

            x_sample = np.random.uniform(*x_workspace, n=1)
            y_sample = np.random.uniform(*y_workspace, n=1)

            target_ee_pose = Pose(position=(x_sample, y_sample, target_z), orientation=home_orn)

            initial_left_finger_val = initial_joint_positions[robot.left_finger_joint_idx]
            initial_right_finger_val = initial_joint_positions[robot.right_finger_joint_idx]
            world_robot_base_pose = robot.get_base_pose(physics_client_id)
            world_robot_joint_positions = robot.get_joints()
            simulator_robot_base_pose = state.base_pose
            #Move robot to simulator's base pose for ik:
            robot.move_base_to(target_pose=simulator_robot_base_pose, physics_client_id=physics_client_id)
            print(f"Calling IK for arm motion planning for target_ee_pose: {target_ee_pose}.")
            input()
            try:
                target_joint_positions = robot.inverse_kinematics(target_ee_pose, validate=False, set_joints=False)
                if target_joint_positions is not None:
                    target_joint_positions[robot.left_finger_joint_idx] = initial_left_finger_val
                    target_joint_positions[robot.right_finger_joint_idx] = initial_right_finger_val
                    waypoints = run_motion_planning(
                                                    robot=robot,
                                                    initial_positions=initial_joint_positions,
                                                    target_positions=target_joint_positions,
                                                    collision_bodies=filtered_collision_bodies,
                                                    seed=seed,
                                                    physics_client_id=physics_client_id,
                                                    held_object=held_obj_id,
                                                    base_link_to_held_object=base_link_to_held_object, 
                                                    )
                    #Reset robot to world base pose:
                    robot.move_base_to(target_pose=world_robot_base_pose, physics_client_id=physics_client_id)
                    robot.set_joints(world_robot_joint_positions)
            except InverseKinematicsError:
                robot.move_base_to(target_pose=world_robot_base_pose, physics_client_id=physics_client_id)
                robot.set_joints(world_robot_joint_positions)
                raise utils.OptionExecutionFailure(f"\nInverse Kineamtics failed.")

        if waypoints is None or len(waypoints) == 0:
            raise utils.OptionExecutionFailure(f"{name}: motion planning failed or returned empty path.")

        # 6) Convert waypoints -> Actions
        actions: List[Action] = []
        for q in waypoints:
            # Build a full-size action array and set arm joints. Fingers included in q if you put them there.
            arr = np.zeros_like(robot.action_space.low, dtype=np.float32)
            # Place the planned arm (and fingers, if present) into arr:
            # Assuming run_motion_planning used the same ordering/dim as robot.get_joints()
            arr[:len(q)] = np.array(q, dtype=np.float32)
            # Clip to action space just in case
            arr = np.clip(arr, robot.action_space.low, robot.action_space.high)
            assert robot.action_space.contains(arr)
            action = Action(arr)
            action.set_base_motion((0.0,0.0), "velocity")
            actions.append(action)

        memory["actions"] = actions
        memory["idx"] = 0


    def _initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        # Reset cache each new initiation
        memory.clear()
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        # plan on first call
        if "actions" not in memory or "idx" not in memory:
            _plan_once_and_cache_actions(robot, state, objects, memory)

        i = memory["idx"]
        actions: List[Action] = memory["actions"]
        if i >= len(actions):
            # If the caller kept invoking after terminal, treat as failure (like your other options)
            raise utils.OptionExecutionFailure(f"{name}: policy called after completion.")

        act = actions[i]
        memory["idx"] = i + 1
        return act

    def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        return ("actions" in memory and "idx" in memory and memory["idx"] >= len(memory["actions"]))

    return ParameterizedOption(
        name=name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )




def create_integrated_move_base_option(
    name: str,
    robot: MobileSingleArmPyBulletRobot,
    types: Sequence[Type],
    params_space: Box,
    get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
                                            Tuple[Pose, JointPositions]],
    home_orn: Sequence[float],
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object_id_at_start: Optional[int] = None,
    ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None,
    rng: Optional[np.random.Generator] = None,
    try_arm_only_first: bool = False,
    base_path_planner_max_tries: int = 30,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    final_finger_state: Optional[float] = None,
    move_to_pose_tol: float = 0.05,     # 5 cm
    vel: float = 0.2,                   # nominal cruise speed
    LOOKAHEAD: float = 0.25,
    wheel_radius: float = 0.065,
    track_width: float = 0.3748,
    force: float = 5.0,                 # (unused here; wheel control path would use it)
    omega_max: float = 17.4,            # wheel joint limit (rad/s)
    orientation_tol: float = 0.06,     # ~5°
    orientation_gain: float = 2.0,      # yaw P gain
) -> ParameterizedOption:

    # Phase thresholds / gating
    # ORIENTATION_ONLY_DISTANCE = 0.10    # within 10 cm: rotate in place
    # e.g., 0.06–0.07 m
    ORIENTATION_ONLY_DISTANCE = max(0.06, move_to_pose_tol + 0.01)
    ORIENTATION_DEADBAND      = 0.02    # ~1.15° deadband for micro-oscillation
    COMPLETE_STOP_DISTANCE    = move_to_pose_tol

    # Convert wheel limits -> base limits (use these for clipping v, ω)
    v_max = wheel_radius * omega_max
    w_max = 2.0 * wheel_radius * omega_max / track_width


    def _plan_and_cache_base_motion(robot: MobileSingleArmPyBulletRobot, state: State, objects: Sequence[Object],
                                   memory: Dict) -> None:
        
        filtered_collision_bodies = list(collision_bodies)
        if held_object_id_at_start is not None:
            # Exclude the held object from obstacle set
            filtered_collision_bodies = [b for b in filtered_collision_bodies if b != held_object_id_at_start]

        

        # if "Grasp" in name or "Stack" in name:

        #     _, block = objects
    
        #     block_x, block_y, block_z = (state.get(block, "pose_x"),
        #                                  state.get(block, "pose_y"),
        #                                  state.get(block, "pose_z"))
        #     target_z = z_func(block_z)
        #     target_ee_position = (block_x, block_y, target_z)
        #     target_ee_pose = Pose(position=target_ee_position, orientation=home_orn)

        _, table = objects

        table_x, table_y, table_z = (state.get(table, "pose_x"),
                                     state.get(table, "pose_y"),
                                     state.get(table, "pose_z"))

        target_z = table_z * 2.25
        x_workspace = (table_x-0.125, table_x+0.125)
        y_workspace = (table_y-0.2, table_y+0.2)

        x_sample = np.random.uniform(*x_workspace, size=None)
        y_sample = np.random.uniform(*y_workspace, size=None)

        target_ee_pose = Pose(position=(x_sample, y_sample, target_z), orientation=home_orn)
            

        base_path_waypoints: List[Tuple[float, float, float]] = run_coordinated_motion_planning(
                                                                        robot=robot,
                                                                        target_ee_pose=target_ee_pose,
                                                                        collision_bodies=filtered_collision_bodies,
                                                                        seed=seed,
                                                                        physics_client_id=physics_client_id,
                                                                        try_arm_only_first=try_arm_only_first,
                                                                        base_path_planner_max_tries=base_path_planner_max_tries,
                                                                        workspace_bounds=workspace_bounds,
                                                                        rng=np.random.default_rng(seed),
                                                                        final_finger_state=final_finger_state,
                                                                        held_object_id_at_start=held_object_id_at_start,
                                                                        ee_to_held_object_transform_at_start=ee_to_held_object_transform_at_start,
                                                                    )

        if base_path_waypoints is None or len(base_path_waypoints) == 0:
            raise utils.OptionExecutionFailure(f"{name}: Base path planning failed or returned empty path.")

        target_base_pose = base_path_waypoints[-1]
        memory["path"] = base_path_waypoints
        memory["target_base_pose"] = target_base_pose
        memory["path_pointer"] = 0

        return

        

    def _initiable(state: State, memory: dict, objs: Sequence[Object], params: Array) -> bool:
        memory["control_phase"] = "NAVIGATION"
        memory["step_count"] = 0
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        if "path" not in memory or "target_base_pose" not in memory or "path_pointer" not in memory:
            _plan_and_cache_base_motion(robot, state, objects, memory)
        
        
        memory["step_count"] += 1

        # Pose
        # ipdb.set_trace()
        (x, y, theta), arm_q = get_current_base_and_arm_pose(robot, state, objects, params)
        goal_xy = np.array(memory["target_base_pose"][:2])
        goal_yaw = float(memory["target_base_pose"][2])

        cur_xy = np.array([x, y])
        dist_to_goal = float(np.linalg.norm(goal_xy - cur_xy))
        yaw_error = float((goal_yaw - theta + np.pi) % (2*np.pi) - np.pi)

        # Initialize action:
        action = Action(np.zeros_like(robot.action_space.low))
        action._arr[:len(arm_q)] = arm_q

        # 1) Hard stop region
        if dist_to_goal <= COMPLETE_STOP_DISTANCE and abs(yaw_error) <= orientation_tol:
            action.set_base_motion((0.0, 0.0), "velocity")
            memory["control_phase"] = "COMPLETE"
            print(f"[STEP {memory['step_count']}] GOAL REACHED - d={dist_to_goal:.4f}, yaw={yaw_error:.4f}")
            return action

        # 2) Orientation-only (rotate in place)
        if dist_to_goal <= ORIENTATION_ONLY_DISTANCE:
            memory["control_phase"] = "ORIENTATION_ONLY"
            if abs(yaw_error) <= ORIENTATION_DEADBAND:
                #v_cmd, omega = 0.0, 0.0
                # slow nudge
                v_cmd = min(0.15, 1.5 * dist_to_goal)

                # small steering toward goal
                bearing = np.arctan2(goal_xy[1]-y, goal_xy[0]-x)
                yaw_to_goal = ((bearing - theta + np.pi) % (2*np.pi)) - np.pi
                omega = np.clip(0.5 * yaw_to_goal, -w_max*0.2, w_max*0.2)

                v_cmd = float(np.clip(v_cmd, 0.0, v_max))
                omega = float(np.clip(omega, -w_max, w_max))

                print(f"[STEP {memory['step_count']}] ORIENT DEADBAND - d={dist_to_goal:.3f}, yaw={yaw_error:.3f}")
            else:
                # proportional yaw control; clip with base yaw limit
                omega = float(np.clip(orientation_gain * yaw_error, -w_max, w_max))
                v_cmd = 0.0
                print(f"[STEP {memory['step_count']}] ORIENT ONLY - d={dist_to_goal:.3f}, yaw={yaw_error:.3f}, ω={omega:.3f}")

            # IK:
            omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
            omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -omega_max, omega_max) 

            action.set_base_motion((omega_r, omega_l), "velocity")

            # action.set_base_motion((v_cmd, omega), "velocity")
            return action

        # 3) Navigation (pure pursuit-like on lookahead, with gating)
        memory["control_phase"] = "NAVIGATION"

        # advance waypoint pointer if closer to next
        path_pointer = int(memory.get("path_pointer", 0))
        waypoints = memory["path"]
        while path_pointer + 1 < len(waypoints):
            next_x, next_y = waypoints[path_pointer + 1][:2]
            current_x, current_y = waypoints[path_pointer][:2]
            if (x - next_x)**2 + (y - next_y)**2 < (x - current_x)**2 + (y - current_y)**2:
                path_pointer += 1
            else:
                break
        memory["path_pointer"] = path_pointer

        # lookahead waypoint
        lookahead_idx = min(path_pointer + max(1, int(LOOKAHEAD/0.01)), len(waypoints) - 1)
        wx, wy = waypoints[lookahead_idx][:2]

        # heading to lookahead
        alpha_W = float((np.arctan2(wy - y, wx - x) - theta + np.pi) % (2*np.pi) - np.pi)

        # curvature -> ω, plus velocity gating by heading and distance
        la = max(0.05, min(LOOKAHEAD, dist_to_goal))
        v_cmd = vel
        v_cmd *= float(np.exp(-0.8 * abs(alpha_W)))       # slow if misaligned
        v_cmd = float(min(v_cmd, 1.5 * dist_to_goal))     # don’t overdrive when close
        v_cmd = float(np.clip(v_cmd, 0.0, v_max))

        kappa = 2.0 * np.sin(alpha_W) / la
        omega = float(np.clip(kappa * v_cmd, -w_max, w_max))

        print(f"[STEP {memory['step_count']}] NAV - d={dist_to_goal:.3f} ptr={path_pointer} look={lookahead_idx} "
              f"αW={alpha_W:.2f} v={v_cmd:.2f} ω={omega:.2f}")

        # action.set_base_motion((v_cmd, omega), "velocity")

        # IK:
        omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
        omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -omega_max, omega_max) 

        action.set_base_motion((omega_r, omega_l), "velocity")



        return action

    def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        (x, y, th), _ = get_current_base_and_arm_pose(robot, state, objects, params)
        if "path" not in memory or "target_base_pose" not in memory or "path_pointer" not in memory:
            _plan_and_cache_base_motion(robot, state, objects, memory)
        gx, gy, gth = memory["target_base_pose"]
        pos_ok = (np.hypot(gx - x, gy - y) <= move_to_pose_tol)
        yaw_ok = (abs((gth - th + np.pi) % (2*np.pi) - np.pi) <= orientation_tol)
        if pos_ok and yaw_ok:
            print(f"[TERMINAL] SUCCESS - Position error: {np.hypot(gx - x, gy - y):.4f}, "
                  f"Orientation error: {abs((gth - th + np.pi) % (2*np.pi) - np.pi):.4f}")
        return pos_ok and yaw_ok

    return ParameterizedOption(
        name=name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )


def create_disjoint_move_base_option(
    name: str,
    robot: MobileSingleArmPyBulletRobot,
    types: Sequence[Type],
    params_space: Box,
    get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
                                            Tuple[Pose, JointPositions]],
    home_orn: Sequence[float],
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object_id_at_start: Optional[int] = None,
    ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None,
    rng: Optional[np.random.Generator] = None,
    try_arm_only_first: bool = False,
    base_path_planner_max_tries: int = 30,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    final_finger_state: Optional[float] = None,
    move_to_pose_tol: float = 0.05,     # 5 cm
    vel: float = 0.3,                   # nominal cruise speed
    LOOKAHEAD: float = 0.25,
    wheel_radius: float = 0.065,
    track_width: float = 0.3748,
    force: float = 5.0,                 # (unused here; wheel control path would use it)
    omega_max: float = 17.4,            # wheel joint limit (rad/s)
    orientation_tol: float = 0.045,     # ~5°
    orientation_gain: float = 2.0,      # yaw P gain
    dt: float = 0.062
) -> ParameterizedOption:

    # Phase thresholds / gating
    # ORIENTATION_ONLY_DISTANCE = 0.10    # within 10 cm: rotate in place
    # e.g., 0.06–0.07 m
    ORIENTATION_ONLY_DISTANCE = max(0.06, move_to_pose_tol + 0.01)
    ORIENTATION_DEADBAND      = 0.05    # ~1.15° deadband for micro-oscillation
    COMPLETE_STOP_DISTANCE    = move_to_pose_tol

    # Convert wheel limits -> base limits (use these for clipping v, ω)
    v_max = wheel_radius * omega_max
    w_max = 2.0 * wheel_radius * omega_max / track_width


    def _plan_and_cache_base_motion(robot: MobileSingleArmPyBulletRobot, state: State, objects: Sequence[Object],
                                   memory: Dict) -> None:
        
        filtered_collision_bodies = list(collision_bodies)
        if held_object_id_at_start is not None:
            # Exclude the held object from obstacle set
            filtered_collision_bodies = [b for b in filtered_collision_bodies if b != held_object_id_at_start]

        _, table = objects

        table_x, table_y, table_z = (state.get(table, "pose_x"),
                                     state.get(table, "pose_y"),
                                     state.get(table, "pose_z"))

        target_z = table_z * 2.25
        x_workspace = (table_x-0.125, table_x+0.125)
        y_workspace = (table_y-0.2, table_y+0.2)

        x_sample = np.random.uniform(*x_workspace, size=None)
        y_sample = np.random.uniform(*y_workspace, size=None)

        target_ee_pose = Pose(position=(x_sample, y_sample, target_z), orientation=home_orn)
            

        base_path_waypoints: List[Tuple[float, float, float]] = run_coordinated_motion_planning(
                                                                        robot=robot,
                                                                        target_ee_pose=target_ee_pose,
                                                                        collision_bodies=filtered_collision_bodies,
                                                                        seed=seed,
                                                                        physics_client_id=physics_client_id,
                                                                        try_arm_only_first=try_arm_only_first,
                                                                        base_path_planner_max_tries=base_path_planner_max_tries,
                                                                        workspace_bounds=workspace_bounds,
                                                                        rng=np.random.default_rng(seed),
                                                                        final_finger_state=final_finger_state,
                                                                        held_object_id_at_start=held_object_id_at_start,
                                                                        ee_to_held_object_transform_at_start=ee_to_held_object_transform_at_start,
                                                                    )

        if base_path_waypoints is None or len(base_path_waypoints) == 0:
            raise utils.OptionExecutionFailure(f"{name}: Base path planning failed or returned empty path.")

        target_base_pose = base_path_waypoints[-1]
        memory["path"] = base_path_waypoints
        memory["target_base_pose"] = target_base_pose
        memory["path_pointer"] = 0

        return

        

    def _initiable(state: State, memory: dict, objs: Sequence[Object], params: Array) -> bool:
        memory["control_phase"] = "NAVIGATION"
        memory["step_count"] = 0
        memory["COMPLETE_STOP_DISTANCE"] = COMPLETE_STOP_DISTANCE
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        if "path" not in memory or "target_base_pose" not in memory or "path_pointer" not in memory:
            _plan_and_cache_base_motion(robot, state, objects, memory)
        
        
        memory["step_count"] += 1

        # Pose
        # ipdb.set_trace()
        if memory["step_count"] == 1:
            (x, y, theta), arm_q = get_current_base_and_arm_pose(robot, state, objects, params)
            memory["current_internal_base_pose"] = (x,y,theta)
            memory["arm_joints"] = arm_q
        goal_xy = np.array(memory["target_base_pose"][:2])
        goal_yaw = float(memory["target_base_pose"][2])

        cur_xy = np.array([memory["current_internal_base_pose"][0], memory["current_internal_base_pose"][1]])
        dist_to_goal = float(np.linalg.norm(goal_xy - cur_xy))
        yaw_error = float((goal_yaw - memory["current_internal_base_pose"][2] + np.pi) % (2*np.pi) - np.pi)

        # Initialize action:
        action = Action(np.zeros_like(robot.action_space.low))
        action._arr[:len(memory["arm_joints"])] = memory["arm_joints"]

        # 1) Hard stop region
        if dist_to_goal <= memory["COMPLETE_STOP_DISTANCE"] and abs(yaw_error) <= orientation_tol:
            action.set_base_motion((0.0, 0.0), "velocity")
            memory["control_phase"] = "COMPLETE"
            print(f"[STEP {memory['step_count']}] GOAL REACHED - d={dist_to_goal:.4f}, yaw={yaw_error:.4f}")
            return action

        # 2) Orientation-only (rotate in place)
        if dist_to_goal <= ORIENTATION_ONLY_DISTANCE:
            memory["control_phase"] = "ORIENTATION_ONLY"
            if abs(yaw_error) <= ORIENTATION_DEADBAND:
                # Match the orientation first:
                if abs(yaw_error) > orientation_tol:
                    v_cmd = 0.0
                    # small steering toward goal
                    bearing = np.arctan2(goal_xy[1]-memory["current_internal_base_pose"][1], goal_xy[0]-memory["current_internal_base_pose"][0])
                    yaw_to_goal = ((bearing - memory["current_internal_base_pose"][2] + np.pi) % (2*np.pi)) - np.pi
                    omega = np.clip(0.5 * yaw_to_goal, -w_max*0.2, w_max*0.2)
                    omega = float(np.clip(omega*0.05, -w_max, w_max))
                else:
                    #v_cmd, omega = 0.0, 0.0
                    # slow nudge
                    v_cmd = min(0.008, 0.5 * dist_to_goal)

                    # small steering toward goal
                    # bearing = np.arctan2(goal_xy[1]-memory["current_internal_base_pose"][1], goal_xy[0]-memory["current_internal_base_pose"][0])
                    # yaw_to_goal = ((bearing - memory["current_internal_base_pose"][2] + np.pi) % (2*np.pi)) - np.pi
                    # omega = np.clip(0.5 * yaw_to_goal, -w_max*0.2, w_max*0.2)

                    v_cmd = float(np.clip(v_cmd, 0.0, v_max))
                    # omega = float(np.clip(omega*0.2, -w_max, w_max))
                    omega = 0.0

                print(f"[STEP {memory['step_count']}] ORIENT DEADBAND - d={dist_to_goal:.3f}, yaw={yaw_error:.3f}")
                if (yaw_error <= orientation_tol) and (dist_to_goal <= move_to_pose_tol+0.02):
                        memory["COMPLETE_STOP_DISTANCE"] = dist_to_goal

            else:
                # proportional yaw control; clip with base yaw limit
                omega = float(np.clip(orientation_gain * yaw_error, -w_max, w_max))
                v_cmd = 0.0
                print(f"[STEP {memory['step_count']}] ORIENT ONLY - d={dist_to_goal:.3f}, yaw={yaw_error:.3f}, ω={omega:.3f}")

            # IK:
            omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
            omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -omega_max, omega_max) 

            action.set_base_motion((omega_r, omega_l), "velocity")

            #Forward kinematics to compute internal base pose:
            v_fwd = (wheel_radius * 0.5) * (omega_r + omega_l)
            omega_fwd = (wheel_radius/track_width) * (omega_r - omega_l)
            # x_next = memory["current_internal_base_pose"][0] + v_fwd * np.cos(memory["current_internal_base_pose"][2]) * dt
            # y_next = memory["current_internal_base_pose"][1] + v_fwd * np.sin(memory["current_internal_base_pose"][2]) * dt
            # theta_next = (memory["current_internal_base_pose"][2] + omega_fwd*dt + np.pi) % (2*np.pi) - np.pi
            if abs(omega) > 1e-6:
                R = v_fwd / omega_fwd
                theta_next = (memory["current_internal_base_pose"][2] + omega_fwd*dt + np.pi) % (2*np.pi) - np.pi
                x_next  = memory["current_internal_base_pose"][0] + R*(np.sin(theta_next) - np.sin(memory["current_internal_base_pose"][2]))
                y_next  = memory["current_internal_base_pose"][1] - R*(np.cos(theta_next) - np.cos(memory["current_internal_base_pose"][2]))
            else:
                theta_next = memory["current_internal_base_pose"][2]
                x_next  = memory["current_internal_base_pose"][0] + v_fwd*np.cos(memory["current_internal_base_pose"][2])*dt
                y_next  = memory["current_internal_base_pose"][1] + v_fwd*np.sin(memory["current_internal_base_pose"][2])*dt

            #Update memory:
            memory["current_internal_base_pose"] = (x_next, y_next, theta_next)

            # action.set_base_motion((v_cmd, omega), "velocity")
            return action

        # 3) Navigation (pure pursuit-like on lookahead, with gating)
        memory["control_phase"] = "NAVIGATION"

        # advance waypoint pointer if closer to next
        path_pointer = int(memory.get("path_pointer", 0))
        waypoints = memory["path"]
        while path_pointer + 1 < len(waypoints):
            next_x, next_y = waypoints[path_pointer + 1][:2]
            current_x, current_y = waypoints[path_pointer][:2]
            if (memory["current_internal_base_pose"][0] - next_x)**2 + (memory["current_internal_base_pose"][1] - next_y)**2 <\
                            (memory["current_internal_base_pose"][0] - current_x)**2 + (memory["current_internal_base_pose"][1] - current_y)**2:
                path_pointer += 1
            else:
                break
        memory["path_pointer"] = path_pointer

        # lookahead waypoint
        lookahead_idx = min(path_pointer + max(1, int(LOOKAHEAD/0.01)), len(waypoints) - 1)
        wx, wy = waypoints[lookahead_idx][:2]

        # heading to lookahead
        alpha_W = float((np.arctan2(wy - memory["current_internal_base_pose"][1], wx - memory["current_internal_base_pose"][0]) - \
                                                                memory["current_internal_base_pose"][2] + np.pi) % (2*np.pi) - np.pi)

        # curvature -> ω, plus velocity gating by heading and distance
        la = max(0.05, min(LOOKAHEAD, dist_to_goal))
        v_cmd = vel
        v_cmd *= float(np.exp(-0.8 * abs(alpha_W)))       # slow if misaligned
        v_cmd = float(min(v_cmd, 1.5 * dist_to_goal))     # don’t overdrive when close
        v_cmd = float(np.clip(v_cmd, 0.0, v_max))

        kappa = 2.0 * np.sin(alpha_W) / la
        omega = float(np.clip(kappa * v_cmd, -w_max, w_max))

        print(f"[STEP {memory['step_count']}] NAV - d={dist_to_goal:.3f} ptr={path_pointer} look={lookahead_idx} "
              f"αW={alpha_W:.2f} v={v_cmd:.2f} ω={omega:.2f}")

        # action.set_base_motion((v_cmd, omega), "velocity")

        # IK:
        omega_r = np.clip((2*v_cmd+omega*track_width)/(2*wheel_radius), -omega_max, omega_max)
        omega_l = np.clip((2*v_cmd-omega*track_width)/(2*wheel_radius), -omega_max, omega_max) 

        action.set_base_motion((omega_r, omega_l), "velocity")

        #Forward kinematics to compute internal base pose:
        v_fwd = (wheel_radius * 0.5) * (omega_r + omega_l)
        omega_fwd = (wheel_radius/track_width) * (omega_r - omega_l)
        # x_next = memory["current_internal_base_pose"][0] + v_fwd * np.cos(memory["current_internal_base_pose"][2]) * dt
        # y_next = memory["current_internal_base_pose"][1] + v_fwd * np.sin(memory["current_internal_base_pose"][2]) * dt
        # theta_next = (memory["current_internal_base_pose"][2] + omega_fwd*dt + np.pi) % (2*np.pi) - np.pi
        if abs(omega) > 1e-6:
            R = v_fwd / omega_fwd
            theta_next = (memory["current_internal_base_pose"][2] + omega_fwd*dt + np.pi) % (2*np.pi) - np.pi
            x_next  = memory["current_internal_base_pose"][0] + R*(np.sin(theta_next) - np.sin(memory["current_internal_base_pose"][2]))
            y_next  = memory["current_internal_base_pose"][1] - R*(np.cos(theta_next) - np.cos(memory["current_internal_base_pose"][2]))
        else:
            theta_next = memory["current_internal_base_pose"][2]
            x_next  = memory["current_internal_base_pose"][0] + v_fwd*np.cos(memory["current_internal_base_pose"][2])*dt
            y_next  = memory["current_internal_base_pose"][1] + v_fwd*np.sin(memory["current_internal_base_pose"][2])*dt

        #Update memory:
        memory["current_internal_base_pose"] = (x_next, y_next, theta_next)


        return action

    def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        # (x, y, th), _ = get_current_base_and_arm_pose(robot, state, objects, params)
        if "path" not in memory or "target_base_pose" not in memory or "path_pointer" not in memory or\
                                                                 "current_internal_base_pose" not in memory:
            return False
        gx, gy, gth = memory["target_base_pose"]
        pos_ok = np.linalg.norm([gx - memory["current_internal_base_pose"][0], gy - memory["current_internal_base_pose"][1]])\
                                                                                             <= memory["COMPLETE_STOP_DISTANCE"]
        yaw_ok = abs((gth - memory["current_internal_base_pose"][2] + np.pi) % (2*np.pi) - np.pi) <= orientation_tol
        if pos_ok and yaw_ok:
            x, y, th = memory['current_internal_base_pose']
            pos_err = np.hypot(gx - x, gy - y)
            yaw_err = abs(((gth - th + np.pi) % (2*np.pi)) - np.pi)
            print(f"[TERMINAL] SUCCESS - Position error: {pos_err:.4f}, Orientation error: {yaw_err:.4f}")
        return pos_ok and yaw_ok

    return ParameterizedOption(
        name=name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )


def create_base_reset_based_move_base_option(
    name: str,
    robot: MobileSingleArmPyBulletRobot,
    types: Sequence[Type],
    params_space: Box,
    get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
                                            Tuple[Pose, JointPositions]],
    home_orn: Sequence[float],
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object_id_at_start: Optional[int] = None,
    ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None,
    rng: Optional[np.random.Generator] = None,
    try_arm_only_first: bool = False,
    base_path_planner_max_tries: int = 30,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    final_finger_state: Optional[float] = None,
    ) -> ParameterizedOption:

    def _plan_and_cache_base_motion(robot: MobileSingleArmPyBulletRobot, state: State, objects: Sequence[Object],
                                   memory: Dict) -> None:
        
        filtered_collision_bodies = list(collision_bodies)
        if held_object_id_at_start is not None:
            # Exclude the held object from obstacle set
            filtered_collision_bodies = [b for b in filtered_collision_bodies if b != held_object_id_at_start]

        _, table = objects

        table_x, table_y, table_z = (state.get(table, "pose_x"),
                                     state.get(table, "pose_y"),
                                     state.get(table, "pose_z"))

        target_z = table_z+0.4

        # x_workspace = (table_x-0.125, table_x+0.125)
        # y_workspace = (table_y-0.2, table_y+0.2)

        # x_sample = np.random.uniform(*x_workspace, size=None)
        # y_sample = np.random.uniform(*y_workspace, size=None)

        target_ee_pose = Pose(position=(table_x, table_y, target_z), orientation=home_orn)
        # _, block = objects

        # print(f"Block to be picked:{block}.")
        # input()

        # block_x, block_y, block_z = (state.get(block, "pose_x"),
        #                              state.get(block, "pose_y"),
        #                              state.get(block, "pose_z"))

        # target_z = block_z + 0.2

        # target_ee_pose = Pose(position=(block_x, block_y, target_z), orientation=home_orn)


        # print(f"Sending base planning for target EE position: {target_ee_pose}.")
        # input()

        base_path_waypoints: List[Tuple[float, float, float]] = run_coordinated_motion_planning(
                                                                        robot=robot,
                                                                        target_ee_pose=target_ee_pose,
                                                                        collision_bodies=filtered_collision_bodies,
                                                                        seed=seed,
                                                                        physics_client_id=physics_client_id,
                                                                        try_arm_only_first=try_arm_only_first,
                                                                        base_path_planner_max_tries=base_path_planner_max_tries,
                                                                        workspace_bounds=workspace_bounds,
                                                                        rng=np.random.default_rng(seed),
                                                                        final_finger_state=final_finger_state,
                                                                        held_object_id_at_start=held_object_id_at_start,
                                                                        ee_to_held_object_transform_at_start=ee_to_held_object_transform_at_start,
                                                                    )

        if base_path_waypoints is None or len(base_path_waypoints) == 0:
            raise utils.OptionExecutionFailure(f"{name}: Base path planning failed or returned empty path.")

        # target_base_pose = base_path_waypoints[-1]
        current_arm_joints = robot.get_joints()
        base_path_waypoints = deque(base_path_waypoints)
        memory["path"] = base_path_waypoints
        memory["current_arm_joints"] = current_arm_joints

        return

        

    def _initiable(state: State, memory: dict, objs: Sequence[Object], params: Array) -> bool:
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        if "path" not in memory:
            _plan_and_cache_base_motion(robot, state, objects, memory)

        waypoint = memory["path"].popleft()

        action = Action(np.array(memory["current_arm_joints"]))
        action.set_base_motion(params=waypoint, mode="smooth_position")
        return action

    def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        if "path" not in memory:
            return False
        return len(memory["path"])==0

    return ParameterizedOption(
        name=name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )




def create_base_reset_based_move_base_to_pick_option(
    name: str,
    robot: MobileSingleArmPyBulletRobot,
    types: Sequence[Type],
    params_space: Box,
    get_current_base_and_arm_pose: Callable[[SingleArmPyBulletRobot, State, Sequence[Object], Array],
                                            Tuple[Pose, JointPositions]],
    home_orn: Sequence[float],
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object_id_at_start: Optional[int] = None,
    ee_to_held_object_transform_at_start: Optional[Tuple[NDArray, NDArray]] = None,
    rng: Optional[np.random.Generator] = None,
    try_arm_only_first: bool = False,
    base_path_planner_max_tries: int = 30,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    final_finger_state: Optional[float] = None,
    ) -> ParameterizedOption:

    def _plan_and_cache_base_motion(robot: MobileSingleArmPyBulletRobot, state: State, objects: Sequence[Object],
                                   memory: Dict) -> None:
        
        filtered_collision_bodies = list(collision_bodies)
        if held_object_id_at_start is not None:
            # Exclude the held object from obstacle set
            filtered_collision_bodies = [b for b in filtered_collision_bodies if b != held_object_id_at_start]

        _, block = objects

        print(f"Block to be picked:{block}.")
        input()

        block_x, block_y, block_z = (state.get(block, "pose_x"),
                                     state.get(block, "pose_y"),
                                     state.get(block, "pose_z"))

        target_z = block_z + 0.2

        target_ee_pose = Pose(position=(block_x, block_y, target_z), orientation=home_orn)

        # print(f"Target EE position for block {block.name}: {target_ee_pose}.")
        # input()
            

        base_path_waypoints: List[Tuple[float, float, float]] = run_coordinated_motion_planning(
                                                                        robot=robot,
                                                                        target_ee_pose=target_ee_pose,
                                                                        collision_bodies=filtered_collision_bodies,
                                                                        seed=seed,
                                                                        physics_client_id=physics_client_id,
                                                                        try_arm_only_first=try_arm_only_first,
                                                                        base_path_planner_max_tries=base_path_planner_max_tries,
                                                                        workspace_bounds=workspace_bounds,
                                                                        rng=np.random.default_rng(seed),
                                                                        final_finger_state=final_finger_state,
                                                                        held_object_id_at_start=held_object_id_at_start,
                                                                        ee_to_held_object_transform_at_start=ee_to_held_object_transform_at_start,
                                                                    )

        if base_path_waypoints is None or len(base_path_waypoints) == 0:
            raise utils.OptionExecutionFailure(f"{name}: Base path planning failed or returned empty path.")

        # target_base_pose = base_path_waypoints[-1]
        current_arm_joints = robot.get_joints()
        base_path_waypoints = deque(base_path_waypoints)
        memory["path"] = base_path_waypoints
        memory["current_arm_joints"] = current_arm_joints

        return

        

    def _initiable(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        if "path" not in memory:
            _plan_and_cache_base_motion(robot, state, objects, memory)
        
        waypoint = memory["path"].popleft()

        action = Action(np.array(memory["current_arm_joints"]))
        action.set_base_motion(params=waypoint, mode="smooth_position")
        return action

    def _terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
        if "path" not in memory:
            return False
        return len(memory["path"])==0

    return ParameterizedOption(
        name=name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )
