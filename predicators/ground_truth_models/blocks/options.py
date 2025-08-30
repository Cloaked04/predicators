"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple, Union

import ipdb
import random
import pybullet as p
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_multitable_blocks import PyBulletMultiTableBlocksEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option,\
    create_arm_motion_planning_option, create_disjoint_move_base_option,\
    create_base_reset_based_move_base_option, create_base_reset_based_move_base_to_pick_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointInfo, JointPositions
from predicators.pybullet_helpers.link import get_link_state, get_link_pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BlocksGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the (non-pybullet) blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks", "blocks_clear"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size

        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: []
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, block_type])

        Stack = utils.SingletonParameterizedOption(
            # variables: [robot, object on which to stack currently-held-object]
            # params: []
            "Stack",
            cls._create_stack_policy(action_space, block_size),
            types=[robot_type, block_type])

        PutOnTable = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable",
            cls._create_putontable_policy(action_space, block_size),
            types=[robot_type],
            params_space=Box(0, 1, (2, )))

        return {Pick, Stack, PutOnTable}

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, block = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            arr = np.r_[block_pose, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_stack_policy(cls, action_space: Box,
                             block_size: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, block = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            relative_grasp = np.array([
                0.,
                0.,
                block_size,
            ])
            arr = np.r_[block_pose + relative_grasp, 1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_putontable_policy(cls, action_space: Box,
                                  block_size: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            x = BlocksEnv.x_lb + (BlocksEnv.x_ub - BlocksEnv.x_lb) * x_norm
            y = BlocksEnv.y_lb + (BlocksEnv.y_ub - BlocksEnv.y_lb) * y_norm
            z = BlocksEnv.table_height + 0.5 * block_size
            arr = np.array([x, y, z, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy


class PyBulletBlocksGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_blocks environment."""

    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_blocks"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        _, pybullet_robot, _ = \
            PyBulletBlocksEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletBlocksEnv.fingers_state_to_joint(
                pybullet_robot, state.get(robot, "fingers"))

        def open_fingers_func(state: State, objects: Sequence[Object],
                              params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.open_fingers
            return current, target

        def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.closed_fingers
            return current, target

        # Pick
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move down to grasp.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda block_z: (block_z + cls._offset_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Close fingers.
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ])

        # Stack
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        Stack = utils.LinearChainParameterizedOption(
            "Stack",
            [
                # Move to above the block on which we will stack.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda block_z:
                    (block_z + block_size + cls._offset_z),
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: PyBulletBlocksEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ])

        # PutOnTable
        option_types = [robot_type]
        params_space = Box(0, 1, (2, ))
        place_z = PyBulletBlocksEnv.table_height + \
            block_size / 2 + cls._offset_z
        PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=PyBulletBlocksEnv.pick_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnTable",
                    z=place_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorBackUp",
                    z=PyBulletBlocksEnv.pick_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ])

        return {Pick, Stack, PutOnTable}

    @classmethod
    def _create_blocks_move_to_above_block_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """
        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, block = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            target_position = (state.get(block,
                                         "pose_x"), state.get(block, "pose_y"),
                               z_func(state.get(block, "pose_z")))
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _create_blocks_move_to_above_table_option(
            cls, name: str, z: float, finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table.

        The z position of the target pose must be provided.
        """
        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            robot, = objects
            current_position = (state.get(robot, "pose_x"),
                                state.get(robot, "pose_y"),
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            target_position = (
                PyBulletBlocksEnv.x_lb +
                (PyBulletBlocksEnv.x_ub - PyBulletBlocksEnv.x_lb) * x_norm,
                PyBulletBlocksEnv.y_lb +
                (PyBulletBlocksEnv.y_ub - PyBulletBlocksEnv.y_lb) * y_norm, z)
            target_pose = Pose(target_position, home_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)



class PyBulletMultiTableBlocksGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_blocks environment."""

    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01
    CFG.seed = random.randint(0,10000)

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_multitable_blocks"}

    
    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box, robot: MobileSingleArmPyBulletRobot,
                    env: BlocksEnv, physics_client_id: int) -> Set[ParameterizedOption]:

        #ipdb.set_trace()
        robot_type = types["robot"]
        block_type = types["block"]
        table_type = types["table"]
        block_size = CFG.blocks_block_size

        def get_current_fingers(state: State) -> float:
            symbolic_robot, = state.get_objects(robot_type)
            return PyBulletBlocksEnv.fingers_state_to_joint(
                robot, state.get(symbolic_robot, "fingers"))

        def open_fingers_func(state: State, objects: Sequence[Object],
                              params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = robot.open_fingers
            return current, target

        def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = robot.closed_fingers
            return current, target

        #Move
        option_types = [robot_type, table_type]
        params_space = Box(0, 1, (0, ))
        MoveTo = utils.LinearChainParameterizedOption(
                "MoveTo",
                [
                    cls._create_move_robot_base_option(
                        name="MoveRobot",
                        option_types=option_types,
                        params_space=params_space,
                        robot=robot,
                        env=env,
                        physics_client_id=physics_client_id),
                ])

        #MoveToPick
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0,))
        MoveToPick = utils.LinearChainParameterizedOption(
                        "MoveToPick",
                        [
                            cls._create_move_robot_base_to_pick_option(
                                name="MoveRobotToPick",
                                option_types=option_types,
                                params_space=params_space,
                                robot=robot,
                                env=env,
                                physics_client_id=physics_client_id),
                        ])


        # Pick
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                cls._create_move_arm_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda z: (z+0.2),
                    finger_status="open",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
                # Open fingers.
                create_change_fingers_option(
                    robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move down to grasp.
                cls._create_move_arm_to_above_block_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda z: (z + cls._offset_z),
                    finger_status="open",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
                # Close fingers.
                create_change_fingers_option(
                    robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_move_arm_to_above_block_option(
                    name="MoveEndEffectorBackUpPostGrasp",
                    z_func=lambda z: (z+0.2),
                    finger_status="closed",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
            ])

        # Stack
        option_types = [robot_type, block_type, table_type]
        params_space = Box(0, 1, (0, ))
        Stack = utils.LinearChainParameterizedOption(
            "Stack",
            [
                # Move to above the block on which we will stack.
                cls._create_move_arm_to_above_block_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda z: (z+0.2),
                    finger_status="closed",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
                # Move down to place.
                cls._create_move_arm_to_above_block_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda block_z:
                    (block_z + block_size + cls._offset_z),
                    finger_status="closed",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
                # Open fingers.
                create_change_fingers_option(
                    robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_move_arm_to_above_block_option(
                    name="MoveEndEffectorBackUpPostStack",
                    z_func=lambda z: (z+0.2),
                    finger_status="open",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
            ])

        # PutOnTable
        option_types = [robot_type, table_type]
        params_space = Box(0, 1, (2, ))
        place_z = PyBulletBlocksEnv.table_height + \
            block_size / 2 + cls._offset_z
        PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                cls._create_move_arm_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=0.3,
                    finger_status="closed",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
                # Move down to place.
                cls._create_move_arm_to_above_table_option(
                    name="MoveEndEffectorToPutOnTable",
                    z=place_z,
                    finger_status="closed",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
                # Open fingers.
                create_change_fingers_option(
                    robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBlocksEnv.grasp_tol),
                # Move back up.
                cls._create_move_arm_to_above_table_option(
                    name="MoveEndEffectorBackUpPostPutOnTable",
                    z=0.3,
                    finger_status="open",
                    robot=robot,
                    option_types=option_types,
                    params_space=params_space,
                    env=env,
                    physics_client_id=physics_client_id),
            ])

        return {MoveTo, MoveToPick, Pick, Stack, PutOnTable}


    @classmethod
    def _create_move_arm_to_above_block_option(cls, name: str, z_func: Union[Callable[[float], float], float], finger_status:str,
                                               robot: MobileSingleArmPyBulletRobot, option_types:List[Type], 
                                               params_space: Box, env: PyBulletMultiTableBlocksEnv, 
                                               physics_client_id: int) -> ParameterizedOption:

        """Compute/derive values required to initialize the arm motion option which first plans
        then executes the the arm motion.
        """

        initial_joint_position = robot.get_joints()
        held_obj_id = env._held_obj_id
        ee_link_to_held_object = None
        # if "Grasp" in name:
        #     assert held_obj_id is None, "Cannot be holding an item during Pick."
        # elif "Stack" in name:
        #     assert held_obj_id is not None, "Must be holding an item during Stack."

        if held_obj_id is not None:
            # 1. world -> base_link pose | It actually is World -> EE transform
            world_to_ee_pos, world_to_ee_orn = get_link_pose(
                                                        robot.robot_id,
                                                        robot.end_effector_id,
                                                        physics_client_id=physics_client_id
                                                    )

            # 2. base_link -> world
            ee_to_world_pos, ee_to_world_orn = p.invertTransform(
                                                        world_to_ee_pos, world_to_ee_orn
                                                    )
                                                    
            # 3. world -> object
            world_to_obj_pos, world_to_obj_orn = p.getBasePositionAndOrientation(
                                                        held_obj_id, physicsClientId=physics_client_id
                                                    )

            # 4. base_link -> object (chain transforms)
            ee_link_to_held_obj = p.multiplyTransforms(
                                                        ee_to_world_pos, ee_to_world_orn,
                                                        world_to_obj_pos, world_to_obj_orn
                                                        )
        

        all_bodies = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                for i in range(p.getNumBodies(physicsClientId=physics_client_id))]

        collision_bodies = [b for b in all_bodies if b!=robot.robot_id]

        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        return create_arm_motion_planning_option(name=name, robot=robot, types=option_types, params_space=params_space, 
                                physics_client_id=physics_client_id,  initial_joint_positions=initial_joint_position, 
                                z_func=z_func, home_orn=home_orn, collision_bodies=collision_bodies, seed=CFG.seed, 
                                held_obj_id=held_obj_id, base_link_to_held_object=ee_link_to_held_object)

    @classmethod
    def _create_move_robot_base_option(cls, name: str, robot: MobileSingleArmPyBulletRobot, 
                                        option_types: Sequence[Type],  params_space:Box, 
                                        env: PyBulletMultiTableBlocksEnv, physics_client_id: int) -> ParameterizedOption:

        """Compute/derive values required to initialize the base motion option which first plans
        then executes the base motion via differential drive.
        """

        def get_current_base_and_arm_pose(robot: MobileSingleArmPyBulletRobot, state:State, objects: Sequence[Object],
                                         params: Array) -> Tuple[Tuple[float, float, float], JointPositions]:

            current_base_pose = robot.get_base_pose(physics_client_id)
            current_joint_positions = robot.get_joints()

            return current_base_pose, current_joint_positions

        assert physics_client_id is not None
        all_bodies = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                for i in range(p.getNumBodies(physicsClientId=physics_client_id))]

        # ipdb.set_trace()

        collision_bodies = [b for b in all_bodies if b!=robot.robot_id and b!=0]

        held_obj_id_at_start = env._held_obj_id
        ee_link_to_held_obj = None
        if held_obj_id_at_start is not None:
            # 1. world -> base_link pose | It actually is World -> EE transform
            world_to_ee_pos, world_to_ee_orn = get_link_pose(
                                                        robot.robot_id,
                                                        robot.end_effector_id,
                                                        physics_client_id=physics_client_id
                                                    )

            # 2. base_link -> world
            ee_to_world_pos, ee_to_world_orn = p.invertTransform(
                                                        world_to_ee_pos, world_to_ee_orn
                                                    )
                                                    
            # 3. world -> object
            world_to_obj_pos, world_to_obj_orn = p.getBasePositionAndOrientation(
                                                        held_obj_id_at_start, physicsClientId=physics_client_id
                                                    )

            # 4. base_link -> object (chain transforms)
            ee_link_to_held_obj = p.multiplyTransforms(
                                                        ee_to_world_pos, ee_to_world_orn,
                                                        world_to_obj_pos, world_to_obj_orn
                                                        )
        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        return create_base_reset_based_move_base_option(name=name, robot=robot, types=option_types, params_space=params_space, 
            get_current_base_and_arm_pose=get_current_base_and_arm_pose, home_orn=home_orn, collision_bodies=collision_bodies, 
            seed=CFG.seed, physics_client_id=physics_client_id, held_object_id_at_start=held_obj_id_at_start, 
            ee_to_held_object_transform_at_start=ee_link_to_held_obj)



    @classmethod
    def _create_move_robot_base_to_pick_option(cls, name: str, robot: MobileSingleArmPyBulletRobot, 
                                        option_types: Sequence[Type],  params_space:Box, 
                                        env: PyBulletMultiTableBlocksEnv, physics_client_id: int) -> ParameterizedOption:

        """Compute/derive values required to initialize the base motion option which first plans
        then executes the base motion via differential drive.
        """

        def get_current_base_and_arm_pose(robot: MobileSingleArmPyBulletRobot, state:State, objects: Sequence[Object],
                                         params: Array) -> Tuple[Tuple[float, float, float], JointPositions]:

            current_base_pose = robot.get_base_pose(physics_client_id)
            current_joint_positions = robot.get_joints()

            return current_base_pose, current_joint_positions

        assert physics_client_id is not None
        all_bodies = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                for i in range(p.getNumBodies(physicsClientId=physics_client_id))]

        # ipdb.set_trace()

        collision_bodies = [b for b in all_bodies if b!=robot.robot_id and b!=0]

        held_obj_id_at_start = env._held_obj_id
        ee_link_to_held_obj = None
        if held_obj_id_at_start is not None:
            # 1. world -> base_link pose | It actually is World -> EE transform
            world_to_ee_pos, world_to_ee_orn = get_link_pose(
                                                        robot.robot_id,
                                                        robot.end_effector_id,
                                                        physics_client_id=physics_client_id
                                                    )

            # 2. base_link -> world
            ee_to_world_pos, ee_to_world_orn = p.invertTransform(
                                                        world_to_ee_pos, world_to_ee_orn
                                                    )
                                                    
            # 3. world -> object
            world_to_obj_pos, world_to_obj_orn = p.getBasePositionAndOrientation(
                                                        held_obj_id_at_start, physicsClientId=physics_client_id
                                                    )

            # 4. base_link -> object (chain transforms)
            ee_link_to_held_obj = p.multiplyTransforms(
                                                        ee_to_world_pos, ee_to_world_orn,
                                                        world_to_obj_pos, world_to_obj_orn
                                                        )
        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        return create_base_reset_based_move_base_to_pick_option(name=name, robot=robot, types=option_types, params_space=params_space, 
            get_current_base_and_arm_pose=get_current_base_and_arm_pose, home_orn=home_orn, collision_bodies=collision_bodies, 
            seed=CFG.seed, physics_client_id=physics_client_id, held_object_id_at_start=held_obj_id_at_start, 
            ee_to_held_object_transform_at_start=ee_link_to_held_obj)


    @classmethod
    def _create_move_arm_to_above_table_option(cls, name: str, z: float, finger_status:str, robot:MobileSingleArmPyBulletRobot, 
                                            option_types: Sequence[Type], params_space:Box, env: PyBulletMultiTableBlocksEnv, 
                                            physics_client_id: int) -> ParameterizedOption:

        """Compute/derive values required to initialize the arm motion planning option to
        place item on table."""

        initial_joint_position = robot.get_joints()
        held_obj_id = env._held_obj_id
        # assert held_obj_id is not None, "Held object id cannot be none in Place option."
        ee_link_to_held_object = None
        # 1. world -> base_link pose | It actually is World -> EE transform
        if held_obj_id is not None:
            world_to_ee_pos, world_to_ee_orn = get_link_pose(
                                                        robot.robot_id,
                                                        robot.end_effector_id,
                                                        physics_client_id=physics_client_id
                                                    )

            # 2. base_link -> world
            ee_to_world_pos, ee_to_world_orn = p.invertTransform(
                                                        world_to_ee_pos, world_to_ee_orn
                                                    )
                                                    
            # 3. world -> object
            world_to_obj_pos, world_to_obj_orn = p.getBasePositionAndOrientation(
                                                        held_obj_id, physicsClientId=physics_client_id
                                                    )

            # 4. base_link -> object (chain transforms)
            ee_link_to_held_obj = p.multiplyTransforms(
                                                        ee_to_world_pos, ee_to_world_orn,
                                                        world_to_obj_pos, world_to_obj_orn
                                                        )


        all_bodies = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                for i in range(p.getNumBodies(physicsClientId=physics_client_id))]

        collision_bodies = [b for b in all_bodies if b!=robot.robot_id]

        home_orn = PyBulletBlocksEnv.get_robot_ee_home_orn()

        #TODO: Figure out how to compute or pass the target_ee_pose for this option.

        return create_arm_motion_planning_option(name=name, robot=robot, types=option_types, params_space=params_space, 
                                physics_client_id=physics_client_id,  initial_joint_positions=initial_joint_position, 
                                z_func=z, home_orn=home_orn, collision_bodies=collision_bodies, seed=CFG.seed, 
                                held_obj_id=held_obj_id, base_link_to_held_object=ee_link_to_held_object)












