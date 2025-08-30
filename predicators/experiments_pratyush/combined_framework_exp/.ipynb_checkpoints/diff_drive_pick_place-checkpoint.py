import sys
import numpy as np
import pybullet as p
import time
import logging
from typing import List, Tuple, Optional, Sequence, Collection, Dict, Any, cast
import random
import json
import ipdb

from predicators.structs import Action, Array, GroundAtom, Object, State, Type, ParameterizedOption
from predicators import utils
from predicators.settings import CFG
from gym.spaces import Box

#Import core environment methods, robot function etc.

from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions, get_joint_infos, get_joint_positions
from predicators.pybullet_helpers.link import get_link_state, get_link_pose

#Import the functions that are to be tested:

from predicators.pybullet_helpers.motion_planning import run_motion_planning, run_base_motion_planning,\
                                                            run_coordinated_motion_planning
#The pick/place options to be tested are accessed via the env instance
from predicators.pybullet_helpers.controllers import execute_coordinated_path, create_move_end_effector_to_pose_option,\
                                                    create_change_fingers_option, create_move_base_option
#Configure logging for better debugging outputs:
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(
    level=logging.WARNING,                    
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#Defining test configuration, and overriding some default ones:
CFG.pybullet_robot = "fetch_mobile"
CFG.use_gui = True
#Draws helpful debug lines in the workspace.
#NOT SURE WHETHER TO USE THIS. WILL DECIDE AFTER A COUPLE RUNS.
#CFG.pybullet_draw_debug = True
#Initializing with standard size of blocks.
CFG.blocks_block_size = 0.05
CFG.pybullet_birrt_num_iters = 50
CFG.pybullet_birrt_num_attempts = 10
CFG.pybullet_birrt_smooth_amt = 20
CFG.seed = random.randint(0,10000)
#CFG.seed = 12
#Num of PyBullet physics steps per high-level Action in visualize_action_sequence
CFG.pybullet_sim_steps_per_action = 30

#Function to reset robot to a known pose
def reset_robot_fetch_mobile(robot: MobileSingleArmPyBulletRobot,
                             physics_client_id:int,
                             base_pose: Tuple[float, float, float] = (1.35, 0.75, 0.0), # (x,y,theta)
                             arm_joint_angle: Optional[List[float]]=None):
    """
    Resets the robot's base/ teleports it and arm to specified poses.

    """

    robot.move_base_to(base_pose, physics_client_id)
    if arm_joint_angle:
        #Set arm joints only
        robot.set_joints(arm_joint_angle)
    else:
        #robot.initial_joint_positions includes arm and finger joints
        robot.set_joints(robot.initial_joint_positions)
    #Step simulation a bit to allow PyBullet to settle the state.
    for _ in range(10):
        p.stepSimulation(physicsClientId=physics_client_id)


#Fn to create blocks in the env.
def create_test_block(env: PyBulletEnv,
                      pose: Tuple[float, float, float],
                      color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
                      name_suffix: str = "test") -> int:
    """
    Creates a single block at a specified pose for testing and returns its PyBullet ID.
    """

    #Use the fn defined in utils to create block
    block_id = create_pybullet_block(
        color,
        (CFG.blocks_block_size/2,)*3,
        env._obj_mass,
        env._obj_friction,
        env._default_orn,
        env._physics_client_id
    )
    #Place the block at desired pose.
    p.resetBasePositionAndOrientation(block_id, pose, env._default_orn, physicsClientId=env._physics_client_id)

    return block_id


#Get the list of all bodies except the robot.
#TODO: Need to add logic that saves object/body name
#      which can be used for better debugging with collision.
def get_all_non_robot_bodies(robot_id: int, physics_client_id:int) -> List[int]:
    """
    Gets all PyBullet body IDs in the simulation except for the robot itself.
    These are typically used as collision obstacles.
    """
    all_bodies = [p.getBodyUniqueId(i, physicsClientId=physics_client_id)
                    for i in range(p.getNumBodies(physicsClientId=physics_client_id))]


    return [b for b in all_bodies if b!=robot_id]


#Setup Env.
#Initialize the PyBulletBlocksEnv which sets up PyBullet,
#loads the robot, tables etc.

env = PyBulletBlocksEnv(use_gui=CFG.use_gui)

# Resets the environment to a specific task, getting an initial symbolic state.
# While initial_state_from_env is fetched, the option tests will create their own
# more specific symbolic states.
initial_state = env.reset("train", 0)

# The robot instance from the environment
robot = env._pybullet_robot
# The PyBullet physics client ID
physics_client_id = env._physics_client_id
print(f"Physics Client Id: {physics_client_id}.")


if not isinstance(robot, MobileSingleArmPyBulletRobot):
    logging.error("This test script is designed for a MobileSingleArmPyBulletRobot.")

# dyn = p.getDynamicsInfo(robot.robot_id, -1, physicsClientId=env._physics_client_id)
# print(f"\nMass, inertialFrame…{dyn}")
# input()


logging.info(f"Using robot: {robot.get_name()}")


#Store the robot's default arm and finger joint positions.
home_arm_joints = robot.initial_joint_positions

robot_obj = initial_state.get_objects(env._robot_type)[0]

#Store permament, fixed bodies
static_collision_bodies = get_all_non_robot_bodies(robot.robot_id, physics_client_id)

#print(f"Static_collision_bodies:{static_collision_bodies}")

#sys.exit(0)

#Define a rectangular workspace for base motion planning tests.
#(min_x, min_y, max_x, max_y)
workspace_bounds = (1.0, 0.2, 1.7, 1.3)

initial_base_pose = (0.4, 1.6, -np.pi/2)


#Resetting robot's state:
robot.move_base_to(initial_base_pose, physics_client_id)
robot.set_joints(home_arm_joints)

#Step simulation a bit to allow PyBullet to settle the state.
for _ in range(20):
    p.stepSimulation(physicsClientId=physics_client_id)

home_orn = env.get_robot_ee_home_orn()
# Keeping the z a bit high to avoid collision:
z = env.table_height + CFG.blocks_block_size/2 + 0.1
#logging.critical(f"Value of z: {z}.")
orn = (0, 0.7071, 0, 0.7071)

reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose, arm_joint_angle=home_arm_joints)

# Create a block to be picked.
block_to_pick_pose_world = (1.5, 0.75, CFG.blocks_block_size / 2 + env.table_height)
logging.critical(f"World coords of block to pick:{block_to_pick_pose_world}.")
#mark_pos = (1.5, 0.75, (CFG.blocks_block_size / 2) + env.table_height+0.05)


block_to_pick_id = create_test_block(env, pose=block_to_pick_pose_world, name_suffix="pick_target")

# Make a unique name for the symbolic object
symbolic_block_name = f"block{block_to_pick_id}" 
block_to_pick_obj_sym = Object(symbolic_block_name, env._block_type)


# Update the environment's physical-to-symbolic map.
# This tells _get_state() that the new physical block ID now corresponds
# to a new symbolic object.
env._block_id_to_block[block_to_pick_id] = block_to_pick_obj_sym

logging.debug(f"\nBlocks in the env:{env._block_id_to_block}.")

# Get a handle to the official state object, which is mutable.
state_obj_to_modify = env._current_observation
#logging.debug(f"\nCurrent environment: {state_obj_to_modify}.")
assert isinstance(state_obj_to_modify, utils.PyBulletState), \
    f"Expected env._current_observation to be a PyBulletState, got {type(state_obj_to_modify)}"

#"Prime" the state object with a placeholder for the new block.
# This is ONLY to satisfy the assertion inside _get_state().
state_obj_to_modify.data[block_to_pick_obj_sym] = np.zeros(len(env._block_type.feature_names))

# Now, create a fresh state. The assertion will pass because the object
# returned by the _current_state property (which is state_obj_to_modify)
# now has the same objects as the newly created state.
fresh_state = env._get_state()

# Update the official state object IN-PLACE with the fresh data.
# We are not assigning to _current_state; we are modifying the object it points to.
#print(f"[DEBUG] Attributes of state_obj_to_modify: {dir(state_obj_to_modify)}")
state_obj_to_modify.data = fresh_state.data
state_obj_to_modify.simulator_state = fresh_state.simulator_state 
state_obj_to_modify.base_pose = fresh_state.base_pose

print(f"\n State with block:{state_obj_to_modify}")

# Crucial: ensure robot starts not holding anything. PyBulletEnv uses this.
env._held_obj_id = None
# The symbolic robot object, already in env.types and env._get_state()
robot_obj_sym = env._robot

#Simple check to affirm that the block is part of the sim/env.
bodies_in_sim = get_all_non_robot_bodies(robot.robot_id, physics_client_id)
assert block_to_pick_id in bodies_in_sim, f"\n Block to be picked not part of sim."

#Currently chaining the things together.
logging.info(f"Testing PickObject option for block ID {block_to_pick_id} ('{block_to_pick_obj_sym.name}') at {block_to_pick_pose_world}")
# pick_option = env._pick_block_option # Get the option instance from the environment

target_ee_pose = Pose(position=(1.5 , 0.75, z), 
                         orientation=home_orn)

#Moving to the target block + moving arm to around the block:
coordinated_path = run_coordinated_motion_planning(
    robot,
    target_ee_pose=target_ee_pose,
    collision_bodies=static_collision_bodies,
    seed=CFG.seed,
    physics_client_id=physics_client_id,
    try_arm_only_first=True # Planner will try arm-only, fail, then try base+arm
)

# base_path_waypoints, arm_path_waypoints = coordinated_path
base_path_waypoints = coordinated_path


def get_current_base_and_arm_pose(robot, state:State, objects: Sequence[Object], params: Array):

    current_base_pose = robot.get_base_pose(physics_client_id)
    current_joint_positions = robot.get_joints()

    return current_base_pose, current_joint_positions


target_base_pose = base_path_waypoints[-1]

move_option_memory = {}
param_space = Box(low=np.array([], dtype=np.float32),
                  high=np.array([], dtype=np.float32), dtype=np.float32)

move_option = create_move_base_option(robot, name="diff-drive",types=[env._robot_type], params_space=param_space,
                                    get_current_base_and_arm_pose=get_current_base_and_arm_pose, base_path=base_path_waypoints,
                                    target_base_pose = target_base_pose)


grounded_move = move_option.ground([robot_obj], np.array([], dtype=np.float32))

assert grounded_move.initiable(move_option_memory)

print(grounded_move)

state = state_obj_to_modify

# ipdb.set_trace()


while not grounded_move.terminal(move_option_memory):
    action = grounded_move.policy(move_option_memory)
    assert action is not None, "Action for base motion can't be None."
    #logging.warning(f"\nNext action to be simulated:{action}.")
    state = env.simulate(state, action)

    # omega_r, omega_l = action.base_motion['params']

    # robot.set_wheel_motors(omega_r, omega_l, physics_client_id)

    # fixed_arm_pos = action.arr
    # robot.set_motors(fixed_arm_pos)

    # for _ in range(30):
    #     p.stepSimulation(physicsClientId=physics_client_id)
    # time.sleep(0.15)


print(f"\n Robot at {robot.get_base_pose(physics_client_id)} after executing differential drive.")
input()


'''
Now chain arm motion planning with this:
    - convert run_motion_planning to a singleParameterizedOption
    - connect to diff drive
'''

# for waypoint in arm_path_waypoints:
#     action = Action(np.zeros(len(robot.action_space.low), dtype=float))
#     action._arr = waypoint
#     action.set_base_motion(params=(0.0, 0.0), mode="velocity")

#     state = env.simulate(state, action)

    # robot.set_motors(action.arr)
    # robot.set_wheel_motors(robot, 0.0, 0.0, physics_client_id)

    # for _ in range(150):
    #     p.stepSimulation(physicsClientId=physics_client_id)
    # time.sleep(0.05)


"""Chain finger closing to pick block:
"""

# type_dict = {t.name: t for t in env.types}

# robot_type = type_dict["robot"]
# block_type = type_dict["block"]

# def get_current_fingers(state: State) -> float:
#     robot, = state.get_objects(robot_type)
#     return PyBulletBlocksEnv.fingers_state_to_joint(
#         env._pybullet_robot, state.get(robot, "fingers"))

# def close_fingers_func(state: State, objects: Sequence[Object],
#                    params: Array) -> Tuple[float, float]:
#     del objects, params  # unused
#     current = get_current_fingers(state)
#     target = env._pybullet_robot.closed_fingers
#     return current, target

# option_types = [robot_type, block_type]
# params_space = Box(0, 1, (0, ))

# # Close fingers.
# close_finger_option = create_change_fingers_option(
#     robot, "CloseFingers", option_types, params_space,
#     close_fingers_func, CFG.pybullet_max_vel_norm,
#     PyBulletBlocksEnv._finger_action_tol)



# grounded_close_finger_option = close_finger_option.ground([robot_obj_sym, block_to_pick_obj_sym], params=np.array([], dtype=np.float32))

# finger_action_count = 0

# sim_state = state

# try:
#     while not grounded_close_finger_option.terminal(env._current_state):
#         finger_action_count+=1
#         state_1 = cast(utils.PyBulletState, env._current_state)
#         target_1 = np.array(state_1.joint_positions, dtype=np.float32)
#         print(f"\n Joint positions from env._current_state before finger action:{target_1}. ")
#         finger_action = grounded_close_finger_option.policy(env._current_state)

#         sim_state = env.simulate(sim_state, finger_action)
# except utils.OptionExecutionFailure as e:
#     logging.error(f"\nExecuting close_finger_option failed.")

# # ipdb.set_trace()

# assert env._held_obj_id is not None, "Object is not held."
# assert env._held_constraint_id is not None, "Held constraint not created."

# logging.critical(f"Arm is now holding object {block_to_pick_obj_sym.name}.")


# #Slightly move arm up to check if the arm motion is possible without dropping held obj:

# current_x, current_y, current_z = robot.get_state()[:3]
# target_z = current_z+0.5
# ee_position_with_held = (current_x, current_y, target_z)
# ee_pose_with_held_orn = robot.get_state()[3:-1]
# ee_pose_with_held = Pose(position=ee_position_with_held, orientation=ee_pose_with_held_orn)

# initial_joint_positions_with_held = robot.get_joints()

# assert len(initial_joint_positions_with_held) == 9, "Initial joint positions must of lenght 9."

# initial_left_finger_val = initial_joint_positions_with_held[robot.left_finger_joint_idx]
# initial_right_finger_val = initial_joint_positions_with_held[robot.right_finger_joint_idx]

# target_joint_positions_with_held = robot.inverse_kinematics(ee_pose_with_held, validate=True, set_joints=False)

# target_joint_positions_with_held[robot.left_finger_joint_idx] = initial_left_finger_val
# target_joint_positions_with_held[robot.right_finger_joint_idx] = initial_right_finger_val

# bodies_in_sim = [body for body in bodies_in_sim if body != 9 and body != 0]

# print(f"\n Collision bodies: {bodies_in_sim}.")
# # ipdb.set_trace()

# # 1. world -> base_link pose | It actually is World -> EE transform
# world_to_ee_pos, world_to_ee_orn = get_link_pose(
#                                             robot.robot_id,
#                                             robot.end_effector_id,
#                                             physics_client_id=physics_client_id
#                                         )

# # 2. base_link -> world
# ee_to_world_pos, ee_to_world_orn = p.invertTransform(
#                                             world_to_ee_pos, world_to_ee_orn
#                                         )
                                        
# # 3. world -> object
# world_to_obj_pos, world_to_obj_orn = p.getBasePositionAndOrientation(
#                                             env._held_obj_id, physicsClientId=physics_client_id
#                                         )

# # 4. base_link -> object (chain transforms)
# ee_link_to_held_obj = p.multiplyTransforms(
#                                             ee_to_world_pos, ee_to_world_orn,
#                                             world_to_obj_pos, world_to_obj_orn
#                                             )

# arm_waypoints_with_held = run_motion_planning(robot,
#                                               initial_joint_positions_with_held,
#                                               target_joint_positions_with_held,
#                                               bodies_in_sim,
#                                               CFG.seed,
#                                               physics_client_id,
#                                               held_object=env._held_obj_id,
#                                               base_link_to_held_object=ee_link_to_held_obj)

# assert arm_waypoints_with_held is not None, "Arm planning with obj held failed."

# for waypoint in arm_waypoints_with_held:
#     assert len(waypoint) == 9, "Waypoints must be of length 9."

#     action = Action(np.array(waypoint))
#     sim_state = env.simulate(sim_state, action)

# assert env._held_obj_id is not None, "Object is not held after moving arm up."
# assert env._held_constraint_id is not None, "Held constraint did not remain intact after moving arm up."
# # print(f"\nMoving arm up with object held complete.")
# input("\nMoving arm up with object held complete.")

# #Trying full-body IK with held obj:

# home_orn = env.get_robot_ee_home_orn()
# target_ee_pos_with_held = Pose(position=(-2.5, 0.75, 0.3), orientation=home_orn)

# #Repeating same code again; TODO: this needs to be a function:
# # 1. world -> base_link pose | It actually is World -> EE transform
# world_to_ee_pos, world_to_ee_orn = get_link_pose(
#                                             robot.robot_id,
#                                             robot.end_effector_id,
#                                             physics_client_id=physics_client_id
#                                         )

# # 2. base_link -> world
# ee_to_world_pos, ee_to_world_orn = p.invertTransform(
#                                             world_to_ee_pos, world_to_ee_orn
#                                         )
                                        
# # 3. world -> object
# world_to_obj_pos, world_to_obj_orn = p.getBasePositionAndOrientation(
#                                             env._held_obj_id, physicsClientId=physics_client_id
#                                         )

# # 4. base_link -> object (chain transforms)
# ee_link_to_held_obj = p.multiplyTransforms(
#                                             ee_to_world_pos, ee_to_world_orn,
#                                             world_to_obj_pos, world_to_obj_orn
#                                             )

# coordinated_path_with_held = run_coordinated_motion_planning(robot,
#                                                              target_ee_pos_with_held,
#                                                              bodies_in_sim,
#                                                              CFG.seed,
#                                                              physics_client_id,
#                                                              try_arm_only_first=False,
#                                                              final_finger_state=robot.closed_fingers,
#                                                              held_object_id_at_start=env._held_obj_id,
#                                                              ee_to_held_object_transform_at_start=ee_link_to_held_obj
#                                                             )

# assert env._held_obj_id is not None, "Object is not held after base motion planning with held obj."
# assert env._held_constraint_id is not None, "Held constraint did not remain intact after base motion with held obj."

# input("Coordinated planning successful. Continue to simulate:")

# base_path_with_held, arm_path_with_held = coordinated_path_with_held

# sim_state = env._current_observation

# for waypoint in base_path_with_held:
#     action = Action(np.zeros(shape=len(robot.arm_joints),dtype=float))
#     action.set_base_motion(params=waypoint, mode='smooth_position')
#     sim_state = env.simulate(sim_state, action)

# assert env._held_obj_id is not None, "Object is not held after moving base."
# assert env._held_constraint_id is not None, "Held constraint did not remain intact after moving base."
# print(f"\nMoving base with object held complete.")  

# for waypoint in arm_path_with_held:
#     action = Action(np.array(waypoint))
#     sim_state = env.simulate(sim_state, action)

# assert env._held_obj_id is not None, "Object is not held after moving arm after moving base."
# assert env._held_constraint_id is not None, "Held constraint did not remain intact after moving arm after base."
# print(f"\nMoving base and arm with object held complete.")            



input("Test 5 Finished. Press Enter to continue...")