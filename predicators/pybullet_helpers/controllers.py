"""Generic controllers for the robots."""
#from typing import Callable, Dict, Sequence, Set, Tuple, cast, Optional, Any

from typing import Any, Callable, Collection, DefaultDict, Dict, Iterator, \
    List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import\
    MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.joint import JointInfo, JointPositions
from predicators.pybullet_helpers.motion_planning import run_coordinated_motion_planning
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type
from predicators.pybullet_helpers.link import get_link_state

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda", "fetch_mobile"}

#Constants for grasp/place offsets:
#Meters above object center for pre-grasp/ finsh grasp
PICK_PRE_GRASP_Z_OFFSET = 0.1
#Meters above target surface for release
PLACE_RELEASE_Z_OFFSET = 0.1


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
        #3. If the distance is greater than allowed,reduce the step size so that the end-effector
        # moves at most max_vel_norm meters in this time step. Done to keep the motion smooth , 
        # avoiding abrupt jumps.
        if ee_norm > max_vel_norm:
            ee_delta = ee_delta * max_vel_norm / ee_norm
        #4. Compute the new intermediate goal by adding the limited motion vector to current
        # position to get the next position.
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
                                                       validate=False,
                                                       set_joints=True)
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
        # Override the meaningless finger values in joint_action.
        # This is required because IK is only responsible for joints and would have 
        # given random values for fingers. Hence, they are controlled/assigned manually.
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
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

def execute_coordinated_path(
    robot: MobileSingleArmPyBulletRobot,
    base_path:List[Tuple[float, float, float]],
    arm_path: List[JointPositions],
    physics_client_id: int,
    ) -> List[Action]:
    
    """
    Generate a sequence of low-level Action objects to execute a coordinated
    base and arm path, executing the base path first, then the arm path.
    Intuitively, think of this function as the next step after you have 
    outputs from motion planning functions in motion_planning.py. The path generated in those functions
    are converted to Actions by this function.

    Args:
        robot: The mobile robot instance.
        base_path: List of (x,y,theta) waypoints for the base. First element should be current pose.
        arm_path: List of JointPositions waypoints for the arm.
        physics_client_id: PyBullet physics client ID.

    Returns:
        List of Action objects for execution by the environment.
    """

    actions: List[Action] = []
    
    # num_arm_finger_joints = len(robot.joint_lower_limits[:-2]) # This was arm joints only, excluding fingers.

    # Correctly get the number of arm + finger joints, which is the length of robot.get_joints()
    # robot.get_joints() returns positions for robot.arm_joints which *includes* fingers.
    num_controllable_arm_joints = len(robot.get_joints())


    #Get initial arm joint positions
    current_arm_joint_positions = robot.get_joints()
    assert len(current_arm_joint_positions) == num_controllable_arm_joints, \
                        "Mismatch in current arm joint dimensions."

    #First: Generate base movement actions
    #The arm will be held at its current_arm_joint_positions during base movement.

    ####### TODO: Remove all calls/cases for velocity based base motion. These are in files controller.py,###################
    ####### mobile_single_arm.py and inside class Action in structs.py. Remove this comment when done.#######################

    if base_path and len(base_path) > 0: # Ensure base_path is not empty
        current_robot_base_pose_xytheta = robot.get_base_pose(physics_client_id) # Gets (x,y,theta)
        # If base_path has only one point and it's effectively the current pose, no base actions needed.
        if len(base_path) == 1 and np.allclose(base_path[0], current_robot_base_pose_xytheta, atol=1e-3):
             print("Base path is just the current pose, no base movement actions generated.")
        else:
            # The first waypoint in base_path is assumed to be the current/starting pose.
            # We generate actions to move from base_path[i] to base_path[i+1].
            # So, we iterate up to len(base_path) - 1, using base_path[i+1] as the target.
            print(f"Generating {len(base_path) -1 } base movement actions from path of length {len(base_path)}...")
            for i in range(len(base_path) -1): # Iterate through segments
                target_x, target_y, target_theta = base_path[i+1] # Target the next waypoint
                action_arr = np.zeros_like(robot.action_space.low) 
                
                action_arr[:num_controllable_arm_joints] = current_arm_joint_positions

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

            action_arr = np.zeros_like(robot.action_space.low)
            action_arr[:num_controllable_arm_joints] = target_arm_joint_positions_waypoint

            action = Action(action_arr.copy())
            # If there's a final_finger_state implied by this option (e.g., Pick or Place),
            # it should have been incorporated into the arm_path waypoints by run_coordinated_motion_planning.
            # If direct finger commands are needed *per action step* (e.g. for continuous gripper control),
            # that would require changes in how arm_path and options are defined.
            # For now, assuming finger state is part of target_arm_joint_positions_waypoint.
            actions.append(action)

            # This update is mostly for conceptual clarity, as current_arm_joint_positions
            # is not used further in this loop for generating subsequent arm actions.
            current_arm_joint_positions = target_arm_joint_positions_waypoint

    print(f"Generated total {len(actions)} low-level actions.")
    return actions


#---------------------------Creating Options for Base motion-------------


def create_coordinated_motion_option(
    robot: MobileSingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_target_ee_pose: Callable[[State, Sequence[Object], Array], Pose],
    physics_client_id: int,
    try_arm_only_first: bool = True,
    final_finger_state_fn: Optional[Callable[[State, Sequence[Object], Array], Optional[float]]]=None,
    grabbable_object_type: Optional[Type] = None
    ) -> ParameterizedOption:
    
    """
    Create a ParameterizedOption for performing coordinated motion planning and execution.
    The policy plans the full motion on the first call and then executes one from the planned sequence calls.
    """

    def _policy(state:State, memory: Dict, objects:Sequence[Object], params: Array) -> Action:
        if "actions" not in memory or not memory["actions"] or memory.get("current_action_idx",0)==0:
            #Get target end-effector pose
            target_ee_pose = get_target_ee_pose(state, objects, params)

            #Determine final finger state for this specific option execution
            current_final_finger_state = None
            if final_finger_state_fn:
                current_final_finger_state = final_finger_state_fn(state, objects, params)

            #Get collision bodies (all bodies except robot)
            all_body_ids = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                            for i in range(p.getNumBodies(physicsClientId=physics_client_id))]

            collision_bodies = [body_id for body_id in all_body_ids if body_id !=robot.robot_id]

            # Synchronize robot state with current for planning.
            # Reset the PyBullet robot to reflect the joint positions and base pose in the symbolic state
            # for planning.

            # 1) Remember where the simulator really was before we overwrite it:
            current_sim_base_pose_before_sync = robot.get_base_pose(
                physics_client_id, mode="position"
            )
            # 2. Remember joint state too, if you want to restore that exactly:
            current_sim_joint_positions_before_sync = robot.get_joints()

            assert isinstance(state, utils.PyBulletState)
            #Set robot base pose from state if it's a PyBulletState with base_pose
            if hasattr(state, 'base_pose') and state.base_pose is not None:
                #Assuming state.base_pose is (x,y, theta)
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
                for obj_in_state in state:
                    if obj_in_state.is_instance(grabbable_object_type):
                        try:
                            if state.get(obj_in_state, "held") > 0.5:
                                symbolic_held_object = obj_in_state
                                break 
                        except (KeyError, ValueError): # Feature "held" might not exist for all grabbable types
                            pass
            
            # Now, try to get the pybullet ID using the robot's internal _held_obj_id,
            # which should be managed by the PyBulletEnv grasping logic.
            # We use the symbolic_held_object as a confirmation.
            if symbolic_held_object is not None:
                if hasattr(robot, '_held_obj_id') and isinstance(robot, PyBulletEnv) and robot._held_obj_id is not None:
                    # We have a symbolic held object, and the PyBulletEnv robot instance also reports a held_obj_id.
                    # So we have it for future motion planning use.
                    held_object_id_at_start = robot._held_obj_id 
                    # We trust and assume that if symbolic says held, and pybullet env says held, they are the same.
                    # We log this information.
                    print(f"Option '{name}': Symbolic state indicates '{symbolic_held_object.name}' is held. "
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
                    print(f"Warning: Option '{name}': Symbolic state says '{symbolic_held_object.name}' is held, "
                          "but robot instance (PyBulletEnv) does not report a _held_obj_id or is not a PyBulletEnv. "
                          "Cannot get PyBullet ID for held object planning.")
            elif name == "PlaceObject": # If it's a place action, something should ideally be held.
                 print(f"Warning: Option '{name}' called, but no symbolically held grabbable object found in state.")


            # Remove the identified held_object_id_at_start from the set of collision bodies.
            if held_object_id_at_start is not None:
                collision_bodies = [b for b in collision_bodies if b != held_object_id_at_start]

            planning_rng = np.random.default_rng(CFG.pybullet_rng_seed)

            # Store original sim state to restore after planning if needed
            # This is important because run_coordinated_motion_planning itself also modifies state
            # We want the robot to be in the `state` specified by the option call before planning.
            # The set_joints and move_base_to above achieve this.

            result = run_coordinated_motion_planning(
                        robot=robot,
                        target_ee_pose=target_ee_pose,
                        collision_bodies=collision_bodies, 
                        seed=planning_rng.integers(1000000), 
                        physics_client_id=physics_client_id,
                        try_arm_only_first=try_arm_only_first,
                        rng=planning_rng,
                        final_finger_state=current_final_finger_state,
                        held_object_id_at_start=held_object_id_at_start,
                        ee_to_held_object_transform_at_start=ee_to_held_object_transform_at_start
                        )
            
            # After planning, restore the robot to the state it was in when the option policy was first called,
            # as the execution of the *planned path* will occur from this state.
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
            return False # Should continue until planning is attempted or policy sets these
        # Terminal if all planned actions have been executed
        return memory["current_action_idx"] >= len(memory["actions"])


    return ParameterizedOption(
            name=name,
            types=types,
            params_space=params_space,
            policy=policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_terminal
            )


def create_pick_object_option(
    robot: MobileSingleArmPyBulletRobot,
    robot_type: Type,
    object_to_pick_type: Type,
    physics_client_id: int,
    grasp_height_offset: float = PICK_PRE_GRASP_Z_OFFSET,
    #Common grasp orientation: Pointing down -pi/2
    grasp_euler_orn: Tuple[float, float, float] = (0, -np.pi/2, 0),
    # grabbable_object_type parameter removed from here, will be passed internally
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

        #Target EE position for grasping (e.g.: slightly above the object's center)
        ee_grasp_pos = [obj_x, obj_y, obj_z + grasp_height_offset]
        ee_grasp_orn_quat = p.getQuaternionFromEuler(grasp_euler_orn)

        return Pose(ee_grasp_pos, ee_grasp_orn_quat)


    def _get_final_finger_state_for_pick(state: State, objects: Sequence[Object], params: Array) -> float:
        del state, objects, params
        return robot.closed_fingers


    return create_coordinated_motion_option(
            robot=robot,
            name="PickObject",
            types=[robot_type, object_to_pick_type],
            params_space=params_space,
            get_target_ee_pose=_get_target_ee_pose_for_pick,
            physics_client_id=physics_client_id,
            try_arm_only_first=True,
            final_finger_state_fn=_get_final_finger_state_for_pick,
            grabbable_object_type=object_to_pick_type
            )

def create_place_object_option(
    robot: MobileSingleArmPyBulletRobot,
    robot_type: Type,
    location_type: Type,
    object_to_place_type: Type,
    physics_client_id:int,
    place_height_offset: float = PLACE_RELEASE_Z_OFFSET,
    place_euler_orn: Tuple[float, float, float] = (0, -np.pi/2, 0),
    # This grabbable_object_type is crucial: it's the type of object that would be *held*
    # It should match the type of objects that PickObject can pick.
    ) -> ParameterizedOption:
    
    """
    Creates ParamterizedOption for placing the currently held object at a location.
    The location_obj is expected to have 'pose_x', 'pose_y', 'pose_z' feature.
    representing the target placement spot (e.g., center of a region, top of another object.)
    """

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
            grabbable_object_type=object_to_place_type
        )
    


