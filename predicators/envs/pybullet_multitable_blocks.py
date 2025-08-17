#Pybullet version of blocks that allows creating multiple table in the env.

import sys
import logging
from pathlib import Path
from typing import Any, Callable, Collection, DefaultDict, Dict, Iterator, \
    List, Optional, Sequence, Set, Tuple, TypeVar, Union, ClassVar, cast

import numpy as np
import pybullet as p
import random
from PIL import ImageColor

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, create_single_arm_pybullet_robot
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, Object, State, Type, Predicate, GroundAtom



class PyBulletMultiTableBlocksEnv(PyBulletEnv, BlocksEnv):
    """Pybullet Blocks env similar to PyBulletBlocks but with
    capability to initialize multiple tables.
    """

    #Table params:
    _default_table_poses: ClassVar[List[Pose3D]] = [
            (1.0, -0.5, 0.0),
            (-1.7, 1.5, 0.0),
            (2.35, 2.0, 0.0),
            ]

    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    #Workspace params for each table:
    _table_workspaces: ClassVar[List[Dict[str, float]]] = [
            {"x_lb":0.875, "xub": 1.125, "y_lb": -0.7, "y_ub": -0.3},
            {"x_lb":-1.825, "xub": -1.575, "y_lb": 1.3, "y_ub": 1.7},
            {"x_lb":2.225, "xub": 2.475, "y_lb": 1.8, "y_ub": 2.2},
            ]
    robo_x = sum(pose[0] for pose in _default_table_poses)/ len(_default_table_poses)
    robo_y = sum(pose[1] for pose in _default_table_poses)/ len(_default_table_poses)
    robo_z = 0.01


    def __init__(self, use_gui: bool = True,
                 num_tables: int = 3,
                 blocks_per_table: Optional[List[int]] = [2, 2, 2],
                 table_poses: Optional[List[Pose3D]] = None,
                 table_workspaces: Optional[List[Dict[str, float]]] = None,
                 ) -> None:

        #Store multi-table configuration
        self._num_tables = num_tables
        self._blocks_per_table = blocks_per_table
        self._table_poses = table_poses or self._default_table_poses[:num_tables]
        self._table_workspaces = table_workspaces or self._table_workspaces[:num_tables]

        #Validate configuration
        assert len(self._blocks_per_table) == num_tables, f"blocks_per_table_length ({len(self._blocks_per_table)})"
                                                                                f"must match num_tables ({num_tables})."
        assert len(self._table_poses) == num_tables, f"table_poses length ({len(self._table_poses)})"
                                                                f"must match num_tables ({num_tables})"
        assert len(self._table_workspaces) == num_tables, f"table_workspaces length({len(self._table_workspaces)})" 
                                                                            f"must match num_tables ({num_tables})"
          
        super().__init__(use_gui)
          
        # Create table type and objects  
        self._table_type = Type("table", ["pose_x", "pose_y", "pose_z", "id"])
        self._tables = [Object(f"table{i}", self._table_type) for i in range(num_tables)]
          
        # On predicate for multi-table; Eg: On(block1, table2)  
        self._OnTable = Predicate("OnTable", [self._block_type, self._table_type],  
                                        self._OnTable_holds)

        # RobotAt predicate for determining robot's location; Eg: At(robot, table1)
        self._At = Predicate("RobotAt", [self._robot_type, self._table_type], self._AtHolds)  
          
        # Track correspondence between PyBullet IDs and objects  
        self._block_id_to_block: Dict[int, Object] = {}  
        self._table_id_to_table: Dict[int, Object] = {}

        if not isinstance(self._pybullet_robot, MobileSingleArmPyBulletRobot):
            raise TypeError("PyBulletBlocksEnv with coordinated options "
                            "requires a MobileSingleArmPyBulletRobot.")


    @property
    def tables(self) -> List[Object]:
        """Expose the table objects."""
        return self._tables

    @property
    def predicates(self) -> set:
        """Over-ride to include table-specific predicates"""
        base_predicates = set(super().predicates)
        base_predicates.discard(BlocksEnv._OnTable)
        base_predicates.add(self._OnTable)
        base_predicates.add(self._At)
        return base_predicates

    def _OnTable_holds(self, state:State, objects:List[Object]) -> bool:
        """Check if a block is on a specific table
        """
        assert len(objects) == 2
        block, table = objects
        assert block.is_instance(self._block_type)
        assert table.is_instance(self._table_type)

        table_id = int(state.get(table, "id"))
        workspace = self._table_workspaces[table_id]

        bx = state.get(block, "pose_x")
        by = state.get(block, "pose_y")
        bz = state.get(block, "pose_z")

        #Check if block is within table workspace and at table height and not held by robot
        x_in_bounds = workspace["x_lb"] <=bx <= workspace["x_ub"]
        y_in_bounds = workspace["y_lb"] <=by <= workspace["y_ub"]
        z_on_table = abs(bz-(self.table_height+self._block_size/2))<self.on_tol
        not_held = state.get(block, "held")<0.5

        return x_in_bounds and y_in_bounds and z_on_table and not_held


    def _AtHolds(self, state: State, objects: List[Object]) -> bool:
        """Checks whether robot is at a table/loc.
        Currently returning true if robot is within 0.1 of the
        table.
        """
        assert len(objects) == 1, f"Only the table to be checked for must be passed as object."
        table = objects[0]
        assert table.is_instance(self._table_type)

        tx = state.get(table, "pose_x")
        ty = state.get(table, "pose_y")

        x_workspace = (tx-0.125, ty+0.125)
        y_workspace = (ty-0.2, ty+0.2)

        robot_x, robot_y, _ = self._pybullet_robot.get_base_pose(physics_client_id=self._physics_client_id)

        
        return (x_workspace[0]-0.1 <=robot_x <= x_workspace[1]+0.1 and y_workspace[0]-0.1<=robot_y<=y_workspace[1]+0.1)



    @classmethod
    def get_name(cls) -> str:
        return "pybullet_multitable_blocks"

    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool,
        num_tables: int = 3,
        table_poses: Optional[List[Pose3D]] = None
        ) -> Tuple[int, MobileSingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize pybullet with multiple tables"""

        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

        #Use provided poses or defaults
        poses = table_poses or cls._default_table_poses[:num_tables]

        #Create multiple tables
        table_ids = []
        for i, pose in enumerate(poses):
            table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                                  useFixedBase=True,
                                  physicsClientId=physics_client_id)
            p.resetBasePositionAndOrientation(table_id, pose, cls._table_orientation, physicsClientId=physics_client_id)
            table_ids.append(table_id)

        bodies["table_ids"] = table_ids

        #Debug visualization:
        if CFG.pybullet_draw_debug:  # pragma: no cover  
            assert using_gui, "using_gui must be True to use pybullet_draw_debug."  
          
            # Draw workspace for each table: draw debug lines on each table
            # similar to PyBulletBlocksEnv.
            for i, workspace in enumerate(cls._table_workspaces[:num_tables]):  
                color = [1.0, 0.0, 0.0] if i == 0 else [0.0, 1.0, 0.0] if i == 1 else [0.0, 0.0, 1.0]  
                  
                p.addUserDebugLine([workspace["x_lb"], workspace["y_lb"], cls.table_height],  
                                 [workspace["x_ub"], workspace["y_lb"], cls.table_height],  
                                 color, lineWidth=3.0, physicsClientId=physics_client_id)  
                p.addUserDebugLine([workspace["x_lb"], workspace["y_ub"], cls.table_height],  
                                 [workspace["x_ub"], workspace["y_ub"], cls.table_height],  
                                 color, lineWidth=3.0, physicsClientId=physics_client_id)  
                p.addUserDebugLine([workspace["x_lb"], workspace["y_lb"], cls.table_height],  
                                 [workspace["x_lb"], workspace["y_ub"], cls.table_height],  
                                 color, lineWidth=3.0, physicsClientId=physics_client_id)  
                p.addUserDebugLine([workspace["x_ub"], workspace["y_lb"], cls.table_height],  
                                 [workspace["x_ub"], workspace["y_ub"], cls.table_height],  
                                 color, lineWidth=3.0, physicsClientId=physics_client_id)  

        # Create blocks - total from all tables  
        total_blocks = sum(cls._blocks_per_table) if hasattr(cls, '_blocks_per_table') else \
                      max(max(CFG.blocks_num_blocks_train), max(CFG.blocks_num_blocks_test))  
          
        block_ids = []  
        block_size = CFG.blocks_block_size  
        for i in range(total_blocks):  
            color = cls._obj_colors[i % len(cls._obj_colors)]  
            half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)  
            block_ids.append(  
                create_pybullet_block(color, half_extents, cls._obj_mass,  
                                    cls._obj_friction, cls._default_orn,  
                                    physics_client_id))  
        bodies["block_ids"] = block_ids  
  
        return physics_client_id, pybullet_robot, bodies


    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_ids = pybullet_bodies["table_ids"]
        self._block_ids = pybullet_bodies["block_ids"]


    @classmethod
    def _create_pybullet_robot(cls, physics_client_id: int) -> MobileSingleArmPyBulletRobot:
        """Create robot positioned to reach all tables."""

        #Position robot at center of tables:
        avg_x = sum(pose[0] for pose in cls._default_table_poses)/ len(cls._default_table_poses)
        avg_y = sum(pose[1] for pose in cls._default_table_poses)/ len(cls._default_table_poses)
        base_pose = Pose(position=(avg_x, avg_y, 0.01), orientation=(0.0, 0.0, 0.0, 1.0))

        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((avg_x+0.5, avg_y, cls.pick_z), robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot, physics_client_id, ee_home, base_pose)

    def _extract_robot_state(self, state: State) -> Array:
        # The orientation is fixed in this environment.
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        f = self.fingers_state_to_joint(self._pybullet_robot,
                                        state.get(self._robot, "fingers"))
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"), qx, qy, qz, qw, f
                        ], dtype=np.float32)


    def _reset_state(self, state: State) -> None:
        """Resets state with blocks distributed across tables."""
        super()._reset_state(state)

        #Reset tables
        table_objs = state.get_objects(self._table_type)
        self._table_id_to_table = {}
        for i, table_obj in enumerate(table_objs):
            if i<len(self._table_ids):
                table_id = self._table_ids[i]
                self._table_id_to_table[table_id] = table_obj

        #Reset blocks: save a mapping from block id to block obj,
        #and update its pose and color.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            if i<len(self._block_ids):
                block_id = self._block_ids[i]
                self._block_id_to_block[block_id] = block_obj

                bx = state.get(block_obj, "pose_x")
                by = state.get(block_obj, "pose_y")
                bz = state.get(block_obj, "pose_z")

                p.resetBasePositionAndOrientation(block_id, [bx, by, bz], self._default_orn,
                                                    physicsClientId=self._physics_client_id)

                #Update block color
                r = state.get(block_obj, "color_r")
                g = state.get(block_obj, "color_g")
                b = state.get(block_obj, "color_b")
                color = (r, g, b, 1.0)
                p.changeVisualShape(block_id, linkIndex=-1, rgbaColor=color, physicsClientId=self._physics_client_id)

        #Handle held objects
        held_object = self._get_held_block(state)
        if held_object is not None:
            self._force_grasp_object(held_object)

        #Move unused blocks out of view
        h = self._block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(block_id, [oov_x, oov_y, i*h], self._default_orn,
                                            physicsClientId=self._physics_client_id)

        #Validate state reconstruction
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create state including tables and blocks."""
        state_dict = {}

        #Get robot state
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()  
        fingers = self._fingers_joint_to_state(rf)  
        state_dict[self._robot] = np.array([rx, ry, rz, fingers], dtype=np.float32)  
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0]
            r, g, b, _ = visual_data[7]
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([bx, by, bz, held, r, g, b],
                                         dtype=np.float32)

        #Get table states
        for table_id, table in self._table_id_to_table.items():
            (tx, ty, tz), _ = p.getBasePositionAndOrientation(
                table_id, physicsClientId=self._physics_client_id)
            idx = self._table_ids.index(table_id)
            state_dict[table] = np.array([tx, ty, tz, idx], dtype=np.float32)

        #Get base pose for mobile robot:
        if isinstance(self._pybullet_robot, MobileSingleArmPyBulletRobot):
            base_pose = self._pybullet_robot.get_base_pose(self._physics_client_id)
        else:
            base_pose = None

        state = utils.PyBulletState(state_dict, simulator_state=joint_positions, base_pose=base_pose)

        assert set(state) == self._current_state, \
                (f"Reconstructed state has objects {set(state)}, but "
                f"self._current_state has objects {set(self._current_state)}.")

        return state


    #Below function only valid for Single table block envs.
    #If you wish for this to work define the full structure
    #similar to BaseEnv, BlocksEnv, PyBulletBlocksEnv.
    #Defining reset_env() below to set this env.
    # def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
    #            rng: np.random.Generator) -> List[EnvironmentTask]:
    #     tasks = super()._get_tasks(num_tasks, possible_num_blocks, rng)
    #     return self._add_pybullet_state_to_tasks(tasks)


    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch_mobile":
            #Same as "fetch"
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }


    #Currently, copied as is from the PyBulletBlocksEnv.
    #TODO: Check if this needs to be updated for use with multiple
    #tables.
    def _force_grasp_object(self, block: Object) -> None:
        block_to_block_id = {b: i for i, b in self._block_id_to_block.items()}
        block_id = block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()


    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)

    def sample_initial_pile_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]],
            table_idx: int) -> Tuple[float, float]:
        """
        Randomly sample a new (x, y) position for a pile of blocks on the table, making sure it doesn’t overlap
        too closely with any existing piles.
        """
        table_workspace = _table_workspaces[table_idx]

        while True:
            x = rng.uniform(table_workspace['x_lb'], table_workspae['x_ub'])
            y = rng.uniform(table_workspace['y_lb'], table_workspace['y_ub'])
            if table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    def table_xy_is_clear(self, x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        """
        Determine whether a newly sampled (x, y) location is sufficiently far from all existing pile positions
        to avoid a collision.
        """
        if all(
                abs(x - other_x) > self.collision_padding * self._block_size
                for other_x, _ in existing_xys):
            return True
        if all(
                abs(y - other_y) > self.collision_padding * self._block_size
                for _, other_y in existing_xys):
            return True
        return False


    def set_table(self, table_idx: int, exact_state: Dict[str, Any], setup: str = 'pile',
                                            params: Optional[Union[int, List[Any]]] = None) -> State:
        """Take in params and define an initial state for multiple tables
        env.

        mode: 'pile', 'exact_grid', 'exact_scattered' ; pile gives a random pile, exact states allow
        defining exact state with colors of blocks etc.
        other modes can be added to extend the setup,trivially;

        params: integer value for number of piles(max: 3; no reason for this choice);
                integer value for number of rows in grid: fixed to 3 or 4;used for the
                random pile generator.

        exact_state: for exact_pile, currently, the code below assumes that it's in the format
        {"pile1": ["block_color_1", "block_color_2", ..., "block_color_N"], "pile2":[...], ...}
        """

        assert table_idx is not None, f"Table idx not provided."

        if setup == 'pile':
            if params is None:
                num_piles = 3
                #Setting all piles to have the same number of blocks for now.
                num_block_per_pile = random.choice([1,2,3,4])
            else:
                assert isinstance(params, list), f"Params must be a set for mode = pile."
                num_piles, num_blocks_per_pile = params[0], params[1]

            #Create list of piles; each pile constains list of 
            #block objects.
            piles: List[List[Object]] = []
            for pile in range(num_piles):
                piles.append([])
                for block_num in range(num_blocks_per_pile):
                    block = Object(f"block{block_num}", self._block_type)
                    # Add block to pile
                    piles[-1].append(block)

            data: Dict[Object, Array] = {}
            # Create a block to pile index:
            block_to_pile_idx = {}
            for i, pile in enumerate(piles):
                for j, block in enumerate(pile):
                    assert block not in block_to_pile_idx
                    block_to_pile_idx[block] = (i, j)
            
            # Sample pile (x, y)s
            pile_to_xy: Dict[int, Tuple[float, float]] = {}
            #Sample (x,y) for each pile within table workspace,
            #such that they don't overlap and are outside a certain 
            #tolerance level.
            rng = np.random.default_rng()
            for i in range(len(piles)):
                pile_to_xy[i] = sample_initial_pile_xy(
                    rng, set(pile_to_xy.values()), table_idx)

            # Create block states
            for block, pile_idx in block_to_pile_idx.items():
                pile_i, pile_j = pile_idx
                x, y = pile_to_xy[pile_i]
                #z corresponds to the center of the block.
                z = self.table_height + self._block_size * (0.5 + pile_j)
                r, g, b = rng.uniform(size=3)
                if "clear" in self._block_type.feature_names:
                    # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b,
                    # clear]
                    # Block is clear iff it is at the top of a pile
                    clear = pile_j == len(piles[pile_i]) - 1
                    data[block] = np.array([x, y, z, 0.0, r, g, b, clear])
                else:
                    # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
                    data[block] = np.array([x, y, z, 0.0, r, g, b])
            # [pose_x, pose_y, pose_z, fingers]
            # Note: the robot poses are not used in this environment (they are
            # constant), but they change and get used in the PyBullet subclass.
            #TODO: This will create a problem. Need to figure out how to initialize
            #       this correctly for this environment.
            rx, ry, rz = self.robo_x, self.robo_y, self.robo_z
            rf = 1.0  # fingers start out open
            data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
            return State(data)

        if setup == 'exact_pile':
            assert exact_state is not None, f"Must provide a state description for exact state."

            piles: List[List[Object]] = []
            block_to_params_dict: Dict[Object, Any] = {}
            block_count = 0
            for pile in exact_state.keys():
                piles.append([])
                for i in range(len(exact_state[pile])):
                    block = Object(f"block{block_count}", self._block_type)
                    piles[-1].append(block)
                    block_to_params_dict[block] = exact_state[pile][i]
                    block_count+=1

            data: Dict[Object, Array] = {}
            # Create a block to pile index:
            block_to_pile_idx = {}
            for i, pile in enumerate(piles):
                for j, block in enumerate(pile):
                    assert block not in block_to_pile_idx
                    block_to_pile_idx[block] = (i, j)

            # Sample pile (x, y)s
            pile_to_xy: Dict[int, Tuple[float, float]] = {}
            #Sample (x,y) for each pile within table workspace,
            #such that they don't overlap and are outside a certain 
            #tolerance level.
            rng = np.random.default_rng()
            for i in range(len(piles)):
                pile_to_xy[i] = sample_initial_pile_xy(
                    rng, set(pile_to_xy.values()), table_idx)


            # Create block states
            for block, pile_idx in block_to_pile_idx.items():
                pile_i, pile_j = pile_idx
                x, y = pile_to_xy[pile_i]
                #z corresponds to the center of the block.
                z = self.table_height + self._block_size * (0.5 + pile_j)
                r, g, b = ImageColor.getrgb(block_to_params_dict[block])
                if "clear" in self._block_type.feature_names:
                    # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b,
                    # clear]
                    # Block is clear iff it is at the top of a pile
                    clear = pile_j == len(piles[pile_i]) - 1
                    data[block] = np.array([x, y, z, 0.0, r, g, b, clear])
                else:
                    # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
                    data[block] = np.array([x, y, z, 0.0, r, g, b])
            # [pose_x, pose_y, pose_z, fingers]
            # Note: the robot poses are not used in this environment (they are
            # constant), but they change and get used in the PyBullet subclass.
            #TODO: This will create a problem. Need to figure out how to initialize
            #       this correctly for this environment.
            rx, ry, rz = self.robo_x, self.robo_y, self.robo_z
            rf = 1.0  # fingers start out open
            data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
            return State(data)

        if setup == "exact_scattered":
            assert exact_state is not None, f"Must provide a state description for exact state."
            assert isinstance(exact_state, list), f"State description must be provided in a list format for\
                                                    mode exact_scattered."

            table_workspace = _table_workspaces[table_idx]
            #Compute the max number of blocks that can be on the table;
            #maximum allowed blocks is a constant number below that.
            #This constant can be updated if required.
            table_width = np.abs(table_workspace['x_ub'] - table_workspace['x_lb'])
            table_length = np.abs(table_workspace['y_ub'] - table_workspace['y_lb'])
            max_blocks_on_table = np.floor(((table_length*table_width)/(self._block_size*self._block_size))\
                                                        *(1 - self.collision_padding))
            max_num_block_limit = max_blocks_on_table+5

            exact_state = exact_state[:max_num_block_limit]

            #Each block is a new pile, scattered on the table.
            piles: List[List[Object]] = []
            for i in range(len(exact_state)):
                piles.append([])    
                block = Object(f"block{i}", self._block_type)
                piles[-1].append(block)


            data: Dict[Object, Array] = {}
            # Sample pile (x, y)s
            pile_to_xy: Dict[int, Tuple[float, float]] = {}
            #Sample (x,y) for each pile within table workspace,
            #such that they don't overlap and are outside a certain 
            #tolerance level.
            rng = np.random.default_rng()
            for i in range(len(piles)):
                pile_to_xy[i] = sample_initial_pile_xy(
                    rng, set(pile_to_xy.values()), table_idx)

            # Create block states
            for i, block in enumerate(piles):
                # pile_i, pile_j = pile_idx
                x, y = pile_to_xy[i]
                #z corresponds to the center of the block.
                z = self.table_height + self._block_size * 0.5
                r, g, b = ImageColor.getrgb(exact_state[i])
                if "clear" in self._block_type.feature_names:
                    # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b,
                    # clear]
                    # Block is clear iff it is at the top of a pile
                    clear = pile_j == len(piles[pile_i]) - 1
                    data[block[0]] = np.array([x, y, z, 0.0, r, g, b, clear])
                else:
                    # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
                    data[block[0]] = np.array([x, y, z, 0.0, r, g, b])
            # [pose_x, pose_y, pose_z, fingers]
            # Note: the robot poses are not used in this environment (they are
            # constant), but they change and get used in the PyBullet subclass.
            #TODO: This will create a problem. Need to figure out how to initialize
            #       this correctly for this environment.
            rx, ry, rz = self.robo_x, self.robo_y, self.robot_z
            rf = 1.0  # fingers start out open
            data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
            return State(data)







            






        






