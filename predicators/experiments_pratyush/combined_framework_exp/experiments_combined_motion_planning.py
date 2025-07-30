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
CFG.pybullet_sim_steps_per_action = 200

#TODO: Add color to types of logging.

#***************************************************************#

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


#*******************************************************************#

#Temporarily overriding _ee_home_pose from BlockEnv to check whether IKFast
#fails even for the simpler poses:

# simpler_ee_orientation = (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)

# if "pybullet_blocks" not in CFG.pybullet_robot_ee_orns:
#         CFG.pybullet_robot_ee_orns["pybullet_blocks"] = {}
# CFG.pybullet_robot_ee_orns["pybullet_blocks"]["fetch_mobile"] = simpler_ee_orientation

# logging.info(f"Temporarily set fetch_mobile EE home orientation in CFG to: {simpler_ee_orientation}")


#*******************************************************************#
def main_test_script():
    """
    Define tests cases for testing different methods/cases.
    """

    # Temporarily set the number of blocks to 0 for a clean test environment.
    # logging.info("Disabling default block creation for focused testing.")
    # CFG.blocks_num_blocks_train = [0]
    # CFG.blocks_num_blocks_test = [0]

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

    if not isinstance(robot, MobileSingleArmPyBulletRobot):
        logging.error("This test script is designed for a MobileSingleArmPyBulletRobot.")
        return

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

    # initial_base_pose = (0.4, 1.6, -np.pi/2)
    # target_base_pose = (0.75, 0.7441, np.pi / 4) # Target base pose
    # logging.info(f"Testing base-only motion from {initial_base_pose} to {target_base_pose}...")

    # #Resetting robot's state:
    # robot.move_base_to(initial_base_pose, physics_client_id)
    # robot.set_joints(home_arm_joints)

    # #Step simulation a bit to allow PyBullet to settle the state.
    # for _ in range(20):
    #     p.stepSimulation(physicsClientId=physics_client_id)

    # # Call the base motion planner for set of waypoints:
    # # base_path_waypoints = run_base_motion_planning(
    # #     robot,
    # #     target_pose=target_base_pose,
    # #     collision_bodies=static_collision_bodies,
    # #     current_arm_positions=home_arm_joints, # Arm joints to maintain during base planning collision checks
    # #     seed=CFG.seed,
    # #     physics_client_id=physics_client_id,
    # #     workspace_bounds=workspace_bounds
    # # )

    # # assert base_path_waypoints is not None, "No collision free base paths found."
    # # assert len(base_path_waypoints) != 0, "No waypoints in returned base path."

    # home_orn = env.get_robot_ee_home_orn()
    # # Keeping the z a bit high to avoid collision:
    # z = env.table_height + CFG.blocks_block_size/2 + 0.01
    # #logging.critical(f"Value of z: {z}.")
    # orn = (0, 0.7071, 0, 0.7071)

    # target_ee_pose = Pose(position=(1.5 , 0.75, z), 
    #                          orientation=home_orn)

    # coordinated_path = run_coordinated_motion_planning(
    #     robot,
    #     target_ee_pose=target_ee_pose,
    #     collision_bodies=static_collision_bodies,
    #     seed=CFG.seed,
    #     physics_client_id=physics_client_id,
    #     try_arm_only_first=True # Planner will try arm-only, fail, then try base+arm
    # )

    # base_path_waypoints, arm_path_waypoints = coordinated_path

    # def get_current_base_and_arm_pose(robot, state:State, objects: Sequence[Object], params: Array):

    #     current_base_pose = robot.get_base_pose(physics_client_id)
    #     current_joint_positions = robot.get_joints()

    #     return current_base_pose, current_joint_positions


    # move_option_memory = {}
    # param_space = Box(low=np.array([], dtype=np.float32),
    #                   high=np.array([], dtype=np.float32), dtype=np.float32)

    # move_option = create_move_base_option(robot, name="diff-drive",types=[env._robot_type], params_space=param_space,
    #                                     get_current_base_and_arm_pose=get_current_base_and_arm_pose, base_path=base_path_waypoints,
    #                                     target_base_pose = target_base_pose)


    # #The empty nd array is the empty param space that this option takes in as all
    # #the values are passed in to the option creation function. If this were to change,
    # #values will be passed in this array, and assigned to appropricate vars in _initialble
    # #and the param_space Box size will be changed accordingly.
    # grounded_move = move_option.ground([robot_obj], np.array([], dtype=np.float32))

    # assert grounded_move.initiable(move_option_memory)

    # print(grounded_move)

    # state = initial_state

    # while not grounded_move.terminal(move_option_memory):
    #     action = grounded_move.policy(move_option_memory)
    #     #logging.warning(f"\nNext action to be simulated:{action}.")
    #     #state = env.simulate(initial_state, action)

    #     #omega_r, omega_l = action.base_motion
    #     vel, omega = action.base_motion['params']

    #     # print(f"\nWheel velocities: {omega_r, omega_l}.")
    #     # input()

    #     #robot.set_wheel_motors(omega_r, omega_l, physics_client_id)
    #     # X, Y, THETA = robot.get_base_pose(physics_client_id)
    #     # vx, vy = 0.3 * np.cos(THETA), 0.3 * np.sin(THETA)
    #     # p.resetBaseVelocity(
    #     #     robot.robot_id,
    #     #     linearVelocity  = [vx, vy, 0.0],
    #     #     angularVelocity = [0.0, 0.0, omega],
    #     #     physicsClientId = physics_client_id)

    #     robot.set_wheel_motors(robot, vel, omega, physics_client_id)

    #     fixed_arm_pos = action.arr
    #     robot.set_motors(fixed_arm_pos)

    #     for _ in range(30):
    #         p.stepSimulation(physicsClientId=physics_client_id)
    #     time.sleep(0.05)

    #     '''
    #     Now chain arm motion planning with this:
    #         - convert run_motion_planning to a singleParameterizedOption
    #         - connect to diff drive
    #     '''

    #     for waypoint in arm_path_waypoints:
    #         action = Action(np.zeros(len(robot.action_space.low), dtype=float))
    #         action._arr = waypoint

    #         robot.set_motors(action.arr)
    #         robot.set_wheel_motors(robot, 0.0, 0.0, physics_client_id)



    # print(f"\n Robot at {robot.get_base_pose(physics_client_id)} after executing differential drive.")




    # input()




















    ##########################
    # Visualization function #---------------------------
    ##########################

    def visualize_action_sequence(actions: List[Action], test_name: str):
        """
        Executes a sequence of planned actions for visualization.
        Simplified version of PyBulletEnv.step() focused on moving the robot
        according to action's component. It does the following for each Action:
         1. If Action has base motion, tell's robot base to move using smooth approach(linear interpolation).
            Currently, direct teleportation is also implemented but will be removed.
         2. If Action's arr attribute has target arm joint values, it tells the robot's arm motors
            to move to those target joint positions.
         3. If Action has specific gripper commands, translate those into 'open' or 'close' values
            and send to finger motors.
         4. Advance simulation by stepping simulation forward in time so the changes are visible.
        """

        logging.info(f"Visualizing {len(actions)} actions for {test_name}...")

        mark_pos = (1.5, 0.75, 0.325)

        p.addUserDebugText(
            "*",                          
            mark_pos,                     
            textColorRGB=[0, 0, 0],       
            textSize=1,                 
            lifeTime=0,                   
            physicsClientId=physics_client_id)


        # Iterate through each Action object in the provided list.
        for act_idx, act in enumerate(actions):
            logging.debug(f" Executing action {act_idx+1}/{len(actions)}")

            #Debug prints before the taking action:
            base_pos, base_orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=physics_client_id)
            #print(f"[DEBUG] Before action {act_idx}: base_pos = {base_pos}")
            #-----Handle base motion if present in the action
            #----TODO: NEED TO REMOVE ALL OTHER MOTION MODES AND ONLY KEEP SMOOTH MOTION(linear interpolation).
            #1.Check if the current Action object contains a base movement command.
            if act.has_base_motion:
                bm = act.base_motion
                logging.debug(f"    Base motion: mode={bm['mode']}, params={bm['params']}")
                #2. Execute the base motion based on its mode:
                if bm['mode'] == "smooth_position":
                     robot.move_base_smoothly(bm['params'], physics_client_id, time_step=0.01)
                elif bm['mode'] == "position": # Direct teleport
                     robot.move_base_to(bm['params'], physics_client_id)
                else:
                    logging.warning(f"    Unsupported base motion mode in test viz: {bm['mode']}")


            # Handle arm motion if present;action.arr contains arm joint targets.
            # The act.arr attribute of an Action object holds the target joint positions for the arm (and fingers).
            # act.arr has a length greater than 0, it means there's an arm component to this action.
            if len(act.arr) > 0 :
                logging.debug(f"    Arm joints (first 3 of {len(act.arr)}): {act.arr.tolist()[:3]}")
                #Commands motors to move arm joints
                #print(f"[DEBUG] Action idx {act_idx}: len(act.arr) = {len(act.arr)}, expected = {len(robot.arm_joints)}")
                #print(f"[DEBUG] act.arr = {act.arr}")

                arr = act.arr.tolist()

                if len(arr) > len(robot.arm_joints):
                    #print(f"[DEBUG] Truncating action array from {len(arr)} to {len(robot.arm_joints)}")
                    arr = arr[:len(robot.arm_joints)]

                # for idx, (name, val) in enumerate(zip(robot.arm_joint_names, arr)):
                #     print(f"Joint {idx}: {name}, Value: {val}")

                robot.set_motors(arr)

            #------- Handle gripper commands
            if act.has_gripper_command:
                # Typically 0 for close, 1 for open
                cmd = act.gripper_command 
                # Convert the 0/1 command to the actual joint values for open/closed fingers.
                target_finger_val = robot.open_fingers if cmd > 0.5 else robot.closed_fingers
                logging.debug(f"    Gripper command: {cmd} -> finger_val: {target_finger_val}")
                #Get current full arm joint state, then override just the finger joints
                #First, gotta make it mutable
                current_arm_js = list(robot.get_joints()) 
                # Ensure the finger joint indices are valid for the current_arm_js array.
                if robot.left_finger_joint_idx < len(current_arm_js) and \
                   robot.right_finger_joint_idx < len(current_arm_js):
                   #Override the finger joint values.
                   current_arm_js[robot.left_finger_joint_idx] = target_finger_val
                   current_arm_js[robot.right_finger_joint_idx] = target_finger_val
                   # Command motors with updated full joint stte
                   robot.set_joints(current_arm_js) 
                else:
                    logging.warning("    Finger joint indices out of bounds for gripper command in viz.")


            # Step simulation to see the effect of the motors/teleport
            for _ in range(30):
                 p.stepSimulation(physicsClientId=physics_client_id)
            time.sleep(0.08) # Pause briefly for smoother viewing

            #Debug prints after taking action
            base_pos, base_orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=physics_client_id)
            last_joint_positions = robot.get_joints()

            # print(f"\n***********************************************************************************************")
            # print(f"\nRobot's Base before reset: {base_pos}.")
            # print(f"\nRobot's Orientation before reset: {base_orn}.")
            # print(f"\nRobot's EE position before reset: {robot.get_state()[:3]}.")
            # print(f"\n-----------------------------------------------------------------------------------------------")
            # new_base_pos, new_base_orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=physics_client_id)
            # print(f"\nRobot's Base after reset: {new_base_pos}.")
            # print(f"\nRobot's Orientation after reset: {new_base_orn}.")
            # print(f"\nRobot's EE position after reset: {robot.get_state()[:3]}.")
            # print(f"\n***********************************************************************************************")
            # input("Press ENTER to continue...")
        logging.info(f"{test_name} visualization complete.")

        new_base_pos, new_base_orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=physics_client_id)
        print(f"\nRobot's Base at the end of simulation: {new_base_pos}.")
        print(f"\nRobot's Orientation at the end of simulation: {new_base_orn}.")
        print(f"\nRobot's last joint positions at the end of simulation: {last_joint_positions}.")
        print(f"\nRobot's EE position at the end of simulation: {robot.get_state()[:3]}.")

        # link_info = get_link_state(robot.robot_id, robot.wrist_roll_link_id, physics_client_id)
        # wrist_roll_link_pose: Pose = link_info.com_pose

        #logging.critical(f"\nFinal Wrist roll link position is: {wrist_roll_link_pose}.")

        #logging.critical(f"\n**********************************************")

        # link_info = get_link_state(robot.robot_id, robot.tool_link_id, physics_client_id)
        # ee_pose: Pose = link_info.com_pose

        #logging.critical(f"\nFinal EE position is: {ee_pose}.")
        # logging.debug(f"\nEE should around:{block_to_pick_pose_world}.")

        return

        #sys.exit(0)



    #-----------------------------------------------------------
    ##### Test 1: test the run_motion_planning function in motion_planning.py (arm only).
    #-----------------------------------------------------------

    # logging.info("\n----Test 1: run_motion_planning (Arm Only)---")

    # #Reset robot to known starting place
    # reset_robot_fetch_mobile(robot, physics_client_id, base_pose=(0.80, 0.70, 0.0), arm_joint_angle=home_arm_joints)

    # # Create an obstacle for the arm to navigate around.
    # #obstacle_block_id_1 = create_test_block(env, pose=(1.45, 0.75, CFG.blocks_block_size / 2 + env.table_height + 0.1))
    # current_collision_bodies_1 = get_all_non_robot_bodies(robot.robot_id, physics_client_id)

    # # Define a target arm joint configuration (slightly different from home).
    # # target_arm_joints_1 = list(home_arm_joints)
    # target_ee_pose_1 = Pose(position=(1.5, 0.75, CFG.blocks_block_size / 2 + env.table_height + 0.08),\
    #                                                                          orientation=env.get_robot_ee_home_orn())

    # p.addUserDebugText(
    #         "*",                          
    #         target_ee_pose_1.position,                     
    #         textColorRGB=[1, 0, 1],       
    #         textSize=2,                 
    #         lifeTime=0,                   
    #         physicsClientId=physics_client_id)

    # logging.warning(f"\nTarget ee pose for test 1: {target_ee_pose_1}.")
    # target_arm_joints_1 = robot.inverse_kinematics(target_ee_pose_1, validate=True, set_joints=False)
    # # Ensure there are arm joints to modify
    # if len(target_arm_joints_1) > 0: 
    #     #change the first arm joint
    #     target_arm_joints_1[0] += 0.5 
    #     # Clip to ensure target is within valid joint limits (excluding base control parts of action_space)
    #     arm_low = robot.action_space.low[:-2] if isinstance(robot, MobileSingleArmPyBulletRobot) else robot.action_space.low
    #     arm_high = robot.action_space.high[:-2] if isinstance(robot, MobileSingleArmPyBulletRobot) else robot.action_space.high
    #     target_arm_joints_1 = np.clip(target_arm_joints_1, arm_low, arm_high).tolist()
    # else:
    #     target_arm_joints_1 = [] # Should not happen for fetch

    # # Proceed if target is valid
    # if target_arm_joints_1: 
    #     logging.info("Testing arm-only motion to a new joint configuration...")
    #     # Call the arm motion planner.
    #     # initial_positions and target_positions are for arm+finger joints.
    #     arm_path_1 = run_motion_planning(
    #         robot,
    #         initial_positions=home_arm_joints, # Starting arm joint values
    #         target_positions=target_arm_joints_1, # Target arm joint values
    #         collision_bodies=current_collision_bodies_1, # Obstacles to avoid
    #         seed=CFG.seed,
    #         physics_client_id=physics_client_id
    #     )

    #     if arm_path_1:
    #         logging.info(f"Arm-only path found with {len(arm_path_1)} waypoints. Visualizing...")
    #         # Visualize the path by setting joint states directly.
    #         for joints_waypoint in arm_path_1:
    #             robot.set_joints(joints_waypoint)
    #             time.sleep(0.05)
    #     else:
    #         logging.warning("Arm-only path planning failed for Test 1.")

    # p.removeBody(obstacle_block_id_1) # Clean up the obstacle
    # input("Test 1 Finished. Press Enter to continue...")


    # #-----------------------------------------------------------
    # ##### Test 2: test the run_base_motion_planning function in motion_planning.py (base only).
    # #-----------------------------------------------------------

    # logging.info("\n--- Test 2: Testing differential drive ---")
    # initial_base_pose_2 = (0.2, 0.6, -np.pi/2) # Starting base pose
    # reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose_2, arm_joint_angle=home_arm_joints)

    # # Create an obstacle for the base to navigate around.
    # obstacle_block_id_2 = create_test_block(env, pose=(1.4, 0.7, CFG.blocks_block_size / 2 + env.table_height))
    # current_collision_bodies_2 = get_all_non_robot_bodies(robot.robot_id, physics_client_id)

    # target_base_pose_2 = (0.75, 0.7441, np.pi / 4) # Target base pose
    # logging.info(f"Testing base-only motion from {initial_base_pose_2} to {target_base_pose_2}...")

    # # Call the base motion planner.
    # base_path_2 = run_base_motion_planning(
    #     robot,
    #     target_pose=target_base_pose_2,
    #     collision_bodies=current_collision_bodies_2,
    #     current_arm_positions=home_arm_joints, # Arm joints to maintain during base planning collision checks
    #     seed=CFG.seed,
    #     physics_client_id=physics_client_id,
    #     workspace_bounds=workspace_bounds
    # )

    # if base_path_2:
    #     logging.info(f"Base-only path found with {len(base_path_2)} waypoints. Visualizing...")
    #     # Visualize by teleporting base and resetting arm.
    #     for pose_waypoint in base_path_2:
    #         robot.move_base_to(pose_waypoint, physics_client_id)
    #         robot.set_motors(home_arm_joints) # Keep arm static
    #         time.sleep(0.05)
    # else:
    #     logging.warning("Base-only path planning failed for Test 2.")
    # p.removeBody(obstacle_block_id_2) # Clean up
    # input("Test 2 Finished. Press Enter to continue...")


    # #-----------------------------------------------------------
    # ##### Test 3: test the run_coordinated_motion_planning function in motion_planning.py (arm only).
    # #-----------------------------------------------------------

    # logging.info("\n--- Test 3: Coordinated Motion (Arm-Only) ---")
    # initial_base_pose_3 = (-1.40, 0.60, 0.0) # Robot is well-positioned
    # reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose_3, arm_joint_angle=home_arm_joints)

    # current_ee_pose_3 = robot.forward_kinematics(home_arm_joints)
    # target_ee_pos_3_list = list(current_ee_pose_3.position)
    # target_ee_pos_3_list[0] += 0.2 # Small EE movement in x, likely reachable by arm
    # target_ee_pose_3 = Pose(tuple(target_ee_pos_3_list), current_ee_pose_3.orientation) # Keep orientation same

    # logging.info(f"Testing coordinated motion (expect arm-only success) to EE pose: {target_ee_pose_3}")
    # # Call coordinated planner with try_arm_only_first=True.
    # coord_path_3_result = run_coordinated_motion_planning(
    #     robot,
    #     target_ee_pose=target_ee_pose_3,
    #     collision_bodies=static_collision_bodies,
    #     seed=CFG.seed,
    #     physics_client_id=physics_client_id,
    #     try_arm_only_first=True # This is key for this test case as we only move the EE slightly.
    # )

    # if coord_path_3_result:
    #     base_path_3, arm_path_3 = coord_path_3_result
    #     logging.info(f"Coordinated path (arm-only) found: Base waypoints: {len(base_path_3)}, Arm waypoints: {len(arm_path_3)}")
    #     # Expect base_path_3 to contain only the initial_base_pose_3.
    #     # Convert planned paths to executable actions.
    #     actions_3 = execute_coordinated_path(robot, base_path_3, arm_path_3, physics_client_id)
    #     visualize_action_sequence(actions_3, "Test 3 Coordinated (Arm-Only)")
    # else:
    #     logging.warning("Coordinated motion (arm-only) planning failed for Test 3.")
    # input("Test 3 Finished. Press Enter to continue...")


    #-----------------------------------------------------------
    ##### Test 4: test the run_coordinated_motion_planning function in motion_planning.py (base+arm only).
    #-----------------------------------------------------------

    logging.info("\n--- Test 4: Coordinated Motion (Base + Arm) ---")
    # Start further away and rotated, likely needing base movement
    initial_base_pose_4 = (0.4, 0.6, -np.pi/2) 
    reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose_4, arm_joint_angle=home_arm_joints)
    # print(f"[DEBUG] Home arm joints:{home_arm_joints}.")
    # print(f"[DEBUG] Robot's joints after reset in test 4:{robot.get_joints()}.")
    # print(f"[DEBUG] Symbolic state joints (STALE):    {env._current_state.simulator_state}")
    #sys.exit(0)
    home_orn = env.get_robot_ee_home_orn()
    # Define a target EE pose that's likely out of reach for arm-only from initial_base_pose_4.
    # A point on the table 1.35, 0.6, 0.2
    z = env.table_height + CFG.blocks_block_size/2 + 0.1
    #logging.critical(f"Value of z: {z}.")
    orn = (0, 0.7071, 0, 0.7071)

    target_ee_pose_4 = Pose(position=(1.5 , 0.75, z), 
                             orientation=home_orn)

    # --- create a tiny sphere once -------------------------------------------
    # vis_id = p.createVisualShape(
    #             shapeType=p.GEOM_SPHERE,
    #             radius=0.05,                 # 1 cm sphere
    #             rgbaColor=[1, 0, 0, 1])      # bright red

    # sphere_id = p.createMultiBody(
    #                baseMass=0,               # 0 ⇒ purely visual
    #                baseVisualShapeIndex=vis_id,
    #                basePosition=[0,0,0],
    #                baseOrientation=[0,0,0,1])

    # # --- every sim tick, move it to the gripper_link --------------------------

    # pos, orn = p.getLinkState(robot.robot_id,
    #                           robot.tool_link_id,
    #                           computeForwardKinematics=True)[:2]
    # print(f"\nPose and orientation for tool link, i.e., gripper link: {Pose(pos,orn)}.")

    # # teleport the sphere
    # p.resetBasePositionAndOrientation(sphere_id, pos, orn)
    # input()


    # p.addUserDebugText(
    #     "+",                          
    #     wrist_in_world.position,                     
    #     textColorRGB=[0, 0, 0],       
    #     textSize=1,                 
    #     lifeTime=0,                   
    #     physicsClientId=physics_client_id)

    # p.addUserDebugText(
    #     "*",                          
    #     (1.5, 0.75, 0.225),                     
    #     textColorRGB=[0, 1, 1],       
    #     textSize=1,                 
    #     lifeTime=0,                   
    #     physicsClientId=physics_client_id)

    # p.addUserDebugText(
    #     "*",                          
    #     (1.5, 0.75, 0.325),                     
    #     textColorRGB=[1, 0, 1],       
    #     textSize=1,                 
    #     lifeTime=0,                   
    #     physicsClientId=physics_client_id)


    # logging.warning(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # logging.warning(f"Testing coordinated motion (expect base + arm) to EE pose: {target_ee_pose_4}")
    # logging.warning(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # coord_path_4_result = run_coordinated_motion_planning(
    #     robot,
    #     target_ee_pose=target_ee_pose_4,
    #     collision_bodies=static_collision_bodies,
    #     seed=CFG.seed,
    #     physics_client_id=physics_client_id,
    #     try_arm_only_first=True # Planner will try arm-only, fail, then try base+arm
    # )

    # if coord_path_4_result:
    #     base_path_4, arm_path_4 = coord_path_4_result
    #     logging.info(f"Coordinated path (base+arm) found: Base waypoints: {len(base_path_4)}; Arm waypoints:{len(arm_path_4)}.")
    #     actions_4 = execute_coordinated_path(robot, base_path_4, arm_path_4, physics_client_id)
    #     # print(f"\n Last base position in base path: {base_path_4[-1]}")
    #     # print(f"\n Last full length action in the path: {actions_4[-1].arr}.")
    #     # print(f"\n Number of arm joints: {len(robot.arm_joints)}.")
    #     # print(f"\n Length of joint solution in action.arr: {len(actions_4[-1].arr)}")
    #     current_joint_positions = robot.get_joints()
    #     robot.move_base_to(base_path_4[-1], physics_client_id=physics_client_id)
    #     robot.set_joints(actions_4[-1].arr)
    #     ee_position_from_solution = robot.get_state()[:3]
    #     print(f"\nFinal EE position from last action returned by controller: {ee_position_from_solution}.")
    #     robot.move_base_to(initial_base_pose_4, physics_client_id=physics_client_id)
    #     robot.set_joints(current_joint_positions)
    #     input("Press Enter to proceed...")
    #     visualize_action_sequence(actions_4, "Test 4 Coordinated (Base+Arm)")
    #     time.sleep(0.1)

    #     #final_ee_cache.append(new_ee)
    # else:
    #     logging.warning("Coordinated motion base planning failed for Test 4.")

    # print(f"Target EE pose: {target_ee_pose_4}.")

    # input("Test 4 Finished. Press Enter to continue...")


    # filename = "ee_cache.json"
    # with open(filename, "w", encoding="utf-8") as f:
    #     json.dump(final_ee_cache, f, indent=2) 
    # input("Test 4 Finished. Press Enter to continue...")



    # def get_current_and_target_pose_and_finger_status(
    #         state: State, objects: Sequence[Object],
    #         params: Array) -> Tuple[Pose, Pose, str]:
    #     assert not params
    #     robot, block = objects
    #     current_position = (state.get(robot, "pose_x"),
    #                         state.get(robot, "pose_y"),
    #                         state.get(robot, "pose_z"))
    #     current_pose = Pose(current_position, home_orn)
    #     target_pose = target_ee_pose_4
    #     finger_status = "closed"
    #     return current_pose, target_pose, finger_status
    # env._current_state.simulator_state = list(robot.get_joints())
    # # print(f"[DEBUG] Robot's joints after base motion:{robot.get_joints()}.")
    # # print(f"[DEBUG] Symbolic state joints after base motion:    {env._current_state.simulator_state}")
    # logging.warning(f"\nDone with base motion. Moving on to arm motion.")

    # #Now combine this with create_end_effector_to_pose_option:
    # type_dict = {t.name: t for t in env.types} 

    # robot_type = type_dict["robot"]
    # block_type = type_dict["block"]
    # robot_obj, = env._current_state.get_objects(robot_type)
    # block_obj = block_obj  = next(b for b in env._current_state.get_objects(block_type)
    #               if b.name == "block0")
    # option_types = [robot_type, block_type]
    # params_space = Box(0, 1, (0, ))
    # move_to_pose_tol = 1e-4
    # finger_action_nudge_magnitude = 1e-3

    # #block = Object("block0", env._block_type)

    # move_arm = create_move_end_effector_to_pose_option(
    #         robot, "move_arm_test_4", option_types, params_space,
    #         get_current_and_target_pose_and_finger_status,
    #         move_to_pose_tol, CFG.pybullet_max_vel_norm,
    #         finger_action_nudge_magnitude)

    # params = np.array([], dtype=np.float32)

    # move_inst = move_arm.ground([robot_obj, block_obj], params)

    # #move_arm.ground([robot_obj, block_obj], params=np.array([], dtype=np.float32))

    # actions_move_arm = []
    # memory_move_arm = {}


    # try:
    #     while not move_inst.terminal(env._current_state):
    #         act = move_inst.policy(env._current_state)
    #         actions_move_arm.append(act)

    #     logging.info(f"Move arm option {len(actions_move_arm)} actions.")
    #     visualize_action_sequence(actions_move_arm, "Test 4 move arm")

    # except utils.OptionExecutionFailure as e:
    #     logging.error(f"Move arm execution failed: {e}")



    #-----------------------------------------------------------
    ##### Test 5: test PickObject option
    #-----------------------------------------------------------

    # Test the high-level PickObject option. This involves planning to a pre-grasp pose
    # (potentially moving base and arm), then closing fingers. 



    logging.info("\n--- Test 5: PickObject Option ---")
    initial_base_pose_5 = (0.4, 0.6, -np.pi/2)
    reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose_5, arm_joint_angle=home_arm_joints)
    #reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose_5)

    #Try lowering torso
    # try:
    #     torso_joint_id = robot.joint_from_name("torso_lift_joint")
    #     # Get joint limits for the torso
    #     torso_joint_info = p.getJointInfo(robot.robot_id, torso_joint_id, physics_client_id)
    #     torso_lower_limit = torso_joint_info[8]
    #     # Set the torso to its lowest position.
    #     p.resetJointState(robot.robot_id,
    #                       torso_joint_id,
    #                       targetValue=torso_lower_limit,
    #                       physicsClientId=physics_client_id)
    #     logging.info(f"Torso lowered to {torso_lower_limit:.3f}m.")
    # except ValueError:
    #     logging.warning("Could not find torso_lift_joint. Proceeding without lowering torso.")


    # ADD THIS DEBUG BLOCK
    # print("\n[DEBUG] --- Right after Test 5 reset ---")
    # print(f"\n[DEBUG] Home arm joints:{home_arm_joints}.")
    # print(f"[DEBUG] Actual robot joints from PyBullet: {robot.get_joints()}")
    # print(f"[DEBUG] Symbolic state joints:    {env._current_state.simulator_state}")
    # END DEBUG BLOCK

    #sys.exit(0)

    # Create a block to be picked.
    block_to_pick_pose_world = (1.5, 0.75, CFG.blocks_block_size / 2 + env.table_height)
    logging.critical(f"World coords of block to pick:{block_to_pick_pose_world}.")
    mark_pos = (1.5, 0.75, CFG.blocks_block_size / 2 + env.table_height+0.05)

    p.addUserDebugText(
        "*",                          
        mark_pos,                     
        textColorRGB=[0, 0, 0],       
        textSize=1,                 
        lifeTime=0,                   
        physicsClientId=physics_client_id)
    #sys.exit(0)
    block_to_pick_id = create_test_block(env, pose=block_to_pick_pose_world, name_suffix="pick_target")

    #logging.debug(f"Block id for block to pick:{block_to_pick_id}.")


    # --- Symbolic State Setup for Option ---
    # For option testing, we need a symbolic State that the option's policy can understand.
    # In a real run, PyBulletEnv.step() would update env._current_state. Here, we mock it.
    
    # Make a unique name for the symbolic object
    symbolic_block_name = f"block{block_to_pick_id}" 
    block_to_pick_obj_sym = Object(symbolic_block_name, env._block_type)

    #logging.debug(f"Block to pick object:{block_to_pick_obj_sym}.")

    # Step 1: Update the environment's physical-to-symbolic map.
    # This tells _get_state() that the new physical block ID now corresponds
    # to a new symbolic object.
    env._block_id_to_block[block_to_pick_id] = block_to_pick_obj_sym

    #logging.debug(f"\nBlocks in the env:{env._block_id_to_block}.")

    # Step 2: Get a handle to the official state object, which is mutable.
    state_obj_to_modify = env._current_observation
    #logging.debug(f"\nCurrent environment: {state_obj_to_modify}.")
    assert isinstance(state_obj_to_modify, utils.PyBulletState), \
        f"Expected env._current_observation to be a PyBulletState, got {type(state_obj_to_modify)}"


    """
    NEED TO UNDERSTAND WHY EXACTLY THIS WHOLE BLOCK OF CODE IS NECESSARY.

    I know that it is used to add the env with the new block to the state.
    But what exactly is the difference between env._current_observation and 
    env._get_state().
    """

    # Step 3: "Prime" the state object with a placeholder for the new block.
    # This is ONLY to satisfy the assertion inside _get_state().
    state_obj_to_modify.data[block_to_pick_obj_sym] = np.zeros(len(env._block_type.feature_names))

    # Step 4: Now, create a fresh state. The assertion will pass because the object
    # returned by the _current_state property (which is state_obj_to_modify)
    # now has the same objects as the newly created state.
    fresh_state = env._get_state()
    #logging.debug(f"\nFresh state returned by env._get_state: {fresh_state}.")

    # Step 5: Update the official state object IN-PLACE with the fresh data.
    # We are not assigning to _current_state; we are modifying the object it points to.
    #print(f"[DEBUG] Attributes of state_obj_to_modify: {dir(state_obj_to_modify)}")
    state_obj_to_modify.data = fresh_state.data
    state_obj_to_modify.simulator_state = fresh_state.simulator_state 
    state_obj_to_modify.base_pose = fresh_state.base_pose

    # logging.debug(f"\nAfter updating state_obj_to_modify: {state_obj_to_modify}.")
    # logging.debug(f"\n Actually calling env._current_observation gives:{env._current_observation}.")
    # sys.exit(0)

    # Crucial: ensure robot starts not holding anything. PyBulletEnv uses this.
    env._held_obj_id = None
    # The symbolic robot object, already in env.types and env._get_state()
    robot_obj_sym = env._robot

    #Simple check to affirm that the block is part of the sim/env.
    bodies_in_sim = get_all_non_robot_bodies(robot.robot_id, physics_client_id)
    assert block_to_pick_id in bodies_in_sim, f"\n Block to be picked not part of sim."


    logging.info(f"Testing PickObject option for block ID {block_to_pick_id} ('{block_to_pick_obj_sym.name}') at {block_to_pick_pose_world}")
    pick_option = env._pick_block_option # Get the option instance from the environment

    # Check if the option thinks it can start.
    if not pick_option.initiable(env._current_state, {}, [robot_obj_sym, block_to_pick_obj_sym], np.array([])):
        logging.error("PickOption is not initiable. Check state/predicates (e.g., GripperOpen, Clear).")
    else:
        logging.info("PickOption is initiable. Executing policy (will plan and generate actions)...")
        option_memory_pick: Dict = {} # Option policies can use memory
        actions_for_pick: List[Action] = []

        sim_state = env._get_state()

        try:
            # The option's policy, when called, will:
            # 1. Internally call get_target_ee_pose_for_pick.
            # 2. Call run_coordinated_motion_planning to plan base/arm moves to pre-grasp.
            # 3. Call execute_coordinated_path to get a list of low-level Actions.
            #    This list also includes actions for the final finger closing.
            # 4. Store these actions in option_memory_pick["actions"].
            # 5. Return the first action from this list.
            # We simulate multiple calls to the policy to get all actions.
            while not pick_option.terminal(sim_state, option_memory_pick, [robot_obj_sym, block_to_pick_obj_sym], np.array([])):
                 act = pick_option.policy(sim_state, option_memory_pick, [robot_obj_sym, block_to_pick_obj_sym], np.array([]))
                 # actions_for_pick.append(act)
                 sim_state = env.simulate(sim_state, act)
                 # In a real scenario, env._current_state would be updated after each env.step(act)
                 # For this test visualization, we execute all planned actions sequentially from the initial state.
                 # The terminal condition checks memory, so it will become true after all planned actions are retrieved.

            logging.critical(f"PickOption planned {len(actions_for_pick)} actions.")
            #visualize_action_sequence(actions_for_pick, "Test 5 PickOption")

            type_dict = {t.name: t for t in env.types}

            robot_type = type_dict["robot"]
            block_type = type_dict["block"]

            def get_current_fingers(state: State) -> float:
                robot, = state.get_objects(robot_type)
                return PyBulletBlocksEnv.fingers_state_to_joint(
                    env._pybullet_robot, state.get(robot, "fingers"))

            def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
                del objects, params  # unused
                current = get_current_fingers(state)
                target = env._pybullet_robot.closed_fingers
                return current, target

            option_types = [robot_type, block_type]
            params_space = Box(0, 1, (0, ))

            # Close fingers.
            close_finger_option = create_change_fingers_option(
                robot, "CloseFingers", option_types, params_space,
                close_fingers_func, CFG.pybullet_max_vel_norm,
                PyBulletBlocksEnv._finger_action_tol)

            

            grounded_close_finger_option = close_finger_option.ground([robot_obj_sym, block_to_pick_obj_sym], params=np.array([], dtype=np.float32))

            finger_action_count = 0

            try:
                while not grounded_close_finger_option.terminal(env._current_state):
                    finger_action_count+=1
                    state_1 = cast(utils.PyBulletState, env._current_state)
                    target_1 = np.array(state_1.joint_positions, dtype=np.float32)
                    print(f"\n Joint positions from env._current_state before finger action:{target_1}. ")
                    finger_action = grounded_close_finger_option.policy(env._current_state)

                    sim_state = env.simulate(sim_state, finger_action)
            except utils.OptionExecutionFailure as e:
                logging.error(f"\nExecuting close_finger_option failed.")

            # ipdb.set_trace()

            assert env._held_obj_id is not None, "Object is not held."
            assert env._held_constraint_id is not None, "Held constraint not created."

            logging.critical(f"Arm is now holding object {block_to_pick_obj_sym.name}.")


            #Slightly move arm up to check if the arm motion is possible without dropping held obj:

            current_x, current_y, current_z = robot.get_state()[:3]
            target_z = current_z+0.5
            ee_position_with_held = (current_x, current_y, target_z)
            ee_pose_with_held_orn = robot.get_state()[3:-1]
            ee_pose_with_held = Pose(position=ee_position_with_held, orientation=ee_pose_with_held_orn)

            initial_joint_positions_with_held = robot.get_joints()

            assert len(initial_joint_positions_with_held) == 9, "Initial joint positions must of lenght 9."

            initial_left_finger_val = initial_joint_positions_with_held[robot.left_finger_joint_idx]
            initial_right_finger_val = initial_joint_positions_with_held[robot.right_finger_joint_idx]

            target_joint_positions_with_held = robot.inverse_kinematics(ee_pose_with_held, validate=True, set_joints=False)

            target_joint_positions_with_held[robot.left_finger_joint_idx] = initial_left_finger_val
            target_joint_positions_with_held[robot.right_finger_joint_idx] = initial_right_finger_val

            bodies_in_sim = [body for body in bodies_in_sim if body != 9 and body != 0]

            print(f"\n Collision bodies: {bodies_in_sim}.")
            # ipdb.set_trace()

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
                                                        env._held_obj_id, physicsClientId=physics_client_id
                                                    )
            
            # 4. base_link -> object (chain transforms)
            ee_link_to_held_obj = p.multiplyTransforms(
                                                        ee_to_world_pos, ee_to_world_orn,
                                                        world_to_obj_pos, world_to_obj_orn
                                                        )

            arm_waypoints_with_held = run_motion_planning(robot,
                                                          initial_joint_positions_with_held,
                                                          target_joint_positions_with_held,
                                                          bodies_in_sim,
                                                          CFG.seed,
                                                          physics_client_id,
                                                          held_object=env._held_obj_id,
                                                          base_link_to_held_object=ee_link_to_held_obj)


            assert arm_waypoints_with_held is not None, "Arm planning with obj held failed."

            for waypoint in arm_waypoints_with_held:
                assert len(waypoint) == 9, "Waypoints must be of length 9."

                action = Action(np.array(waypoint))
                sim_state = env.simulate(sim_state, action)

            assert env._held_obj_id is not None, "Object is not held after moving arm up."
            assert env._held_constraint_id is not None, "Held constraint did not remain intact after moving arm up."
            print(f"\nMoving arm up with object held complete.")

            #Trying full-body IK with held obj:

            home_orn = env.get_robot_ee_home_orn()
            target_ee_pos_with_held = Pose(position=(-2.5, 0.75, 0.3), orientation=home_orn)

            #Repeating same code again;this needs to be a function:
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
                                                        env._held_obj_id, physicsClientId=physics_client_id
                                                    )
            
            # 4. base_link -> object (chain transforms)
            ee_link_to_held_obj = p.multiplyTransforms(
                                                        ee_to_world_pos, ee_to_world_orn,
                                                        world_to_obj_pos, world_to_obj_orn
                                                        )

            coordinated_path_with_held = run_coordinated_motion_planning(robot,
                                                                         target_ee_pos_with_held,
                                                                         bodies_in_sim,
                                                                         CFG.seed,
                                                                         physics_client_id,
                                                                         try_arm_only_first=False,
                                                                         final_finger_state=robot.closed_fingers,
                                                                         held_object_id_at_start=env._held_obj_id,
                                                                         ee_to_held_object_transform_at_start=ee_link_to_held_obj
                                                                        )

            #input("Coordinated planning successful. Continue to simulate:")

            base_path_with_held, arm_path_with_held = coordinated_path_with_held

            for waypoint in base_path_with_held:
                action = Action(np.zeros(shape=len(robot.arm_joints),dtype=float))
                action.set_base_motion(params=waypoint, mode='smooth_position')
                sim_state = env.simulate(sim_state, action)

            assert env._held_obj_id is not None, "Object is not held after moving base."
            assert env._held_constraint_id is not None, "Held constraint did not remain intact after moving base."
            print(f"\nMoving base with object held complete.")  

            for waypoint in arm_path_with_held:
                action = Action(np.array(waypoint))
                sim_state = env.simulate(sim_state, action)

            assert env._held_obj_id is not None, "Object is not held after moving arm after moving base."
            assert env._held_constraint_id is not None, "Held constraint did not remain intact after moving arm after base."
            print(f"\nMoving base and arm with object held complete.")            



        except utils.OptionExecutionFailure as e:
            logging.error(f"PickOption execution failed: {e}")

    input("Test 5 Finished. Press Enter to continue...")
    # block_to_pick_id and block_to_pick_obj_sym are kept if pick was successful, for the Place test.


    #-----------------------------------------------------------
    ##### Test 6: test PlaceObject option
    #-----------------------------------------------------------
    # Test the high-level PlaceObject option. This assumes PickObject was successful.
    # Involves planning to move the HELD object to a target location and releasing fingers.

    logging.info("\n--- Test 6: PlaceObject Option ---")
    if env._held_obj_id != block_to_pick_id: # Check if PickObject test set this up correctly
        logging.error("Cannot run PlaceObject test because PickObject failed or block was not recorded as held.")
        # Clean up the block if it exists and pick failed.
        if p.getBodyInfo(block_to_pick_id, physicsClientId=physics_client_id) is not None:
             p.removeBody(block_to_pick_id, physicsClientId=physics_client_id)
    else:
        logging.info(f"Robot is holding block {block_to_pick_id} ('{block_to_pick_obj_sym.name}'). Proceeding with PlaceObject test.")
        place_target_world_pos = (1.3, 0.85, CFG.blocks_block_size / 2 + env.table_height) # Target placement on table

        # --- Symbolic State Setup for Place Option ---
        # Create a "dummy" symbolic object for the placement location.
        # The PlaceObject option is defined to take a `location_type` object.
        # For placing on the table, this location_obj_sym's pose features define the target spot.
        location_obj_sym = Object("place_location_dummy", env._block_type) # Using block_type for convenience as it has pose features

        # Get current symbolic state (robot is holding block_to_pick_id).
        current_symbolic_state_6 = env._get_state()
        # Ensure the HELD block's symbolic state correctly reflects "held = 1.0".
        if block_to_pick_obj_sym in current_symbolic_state_6.data:
            current_symbolic_state_6.set(block_to_pick_obj_sym, "held", 1.0)
        else: # Should not happen if _get_state() is working after pick
            logging.warning(f"Held block {block_to_pick_obj_sym.name} not found in symbolic state for Place test. Manually adding.")
            current_symbolic_state_6.data[block_to_pick_obj_sym] = np.array([0,0,0, 1.0, 0.8,0.2,0.2]) # Dummy pose, held=1
        # Add the dummy location object to the symbolic state with its target pose features.
        current_symbolic_state_6.data[location_obj_sym] = np.array(
            [place_target_world_pos[0], place_target_world_pos[1], place_target_world_pos[2], # pose_x,y,z
             0.0, # held (location is not held)
             0.2, 0.8, 0.2], dtype=np.float32) # color_r,g,b

        logging.info(f"Testing PlaceObject option to place held block at {place_target_world_pos}")
        place_option = env._place_block_option # Get the option instance

        # Parameters for PlaceObject: [robot_obj_sym, location_obj_sym]
        if not place_option.initiable(current_symbolic_state_6, {}, [robot_obj_sym, location_obj_sym], np.array([])):
            logging.error("PlaceOption is not initiable. Check state/predicates (e.g., robot must be holding something).")
        else:
            logging.info("PlaceOption is initiable. Executing policy...")
            option_memory_place: Dict = {}
            actions_for_place: List[Action] = []
            try:
                # Similar to Pick, call policy repeatedly to get all planned actions.
                while not place_option.terminal(current_symbolic_state_6, option_memory_place, [robot_obj_sym, location_obj_sym], np.array([])):
                    act = place_option.policy(current_symbolic_state_6, option_memory_place, [robot_obj_sym, location_obj_sym], np.array([]))
                    actions_for_place.append(act)

                logging.info(f"PlaceOption planned {len(actions_for_place)} actions.")
                visualize_action_sequence(actions_for_place, "Test 6 PlaceOption")

                # --- Verification of Place ---
                # After actions, fingers should be open, and constraint removed.
                # This is normally handled by PyBulletEnv.step() processing finger opening actions.
                # For test, manually update env's grasp state.
                if env._held_constraint_id is not None:
                    p.removeConstraint(env._held_constraint_id, physics_client_id)
                    env._held_constraint_id = None
                env._held_obj_id = None # Robot should no longer be holding

                if env._held_obj_id is None:
                    logging.info("Place successful! Robot is no longer holding an object.")
                    # Check if the block is now at the target physical location.
                    final_block_pos, _ = p.getBasePositionAndOrientation(block_to_pick_id, physics_client_id)
                    if np.allclose(final_block_pos, place_target_world_pos, atol=1e-5): # Tolerance for placement
                        logging.info(f"Block correctly placed at {final_block_pos}")
                    else:
                        logging.warning(f"Block ended up at {final_block_pos}, expected near {place_target_world_pos}")
                else:
                    logging.error(f"Place appears to have failed. Robot still holding PyBullet ID: {env._held_obj_id}")

            except utils.OptionExecutionFailure as e:
                logging.error(f"PlaceOption execution failed: {e}")

        # Clean up the picked block if it still exists.
        if p.getBodyInfo(block_to_pick_id, physicsClientId=physics_client_id) is not None:
            p.removeBody(block_to_pick_id, physicsClientId=physics_client_id)
        # Remove the dummy location from _block_id_to_block if it was added for symbolic state
        if location_obj_sym in env._block_id_to_block.values():
            # Find key by value and remove (this is a bit hacky for test cleanup)
            key_to_del = [k for k, v in env._block_id_to_block.items() if v == location_obj_sym]
            if key_to_del:
                del env._block_id_to_block[key_to_del[0]]


    logging.info("\n--- All Tests Completed ---")
    input("Press Enter to close PyBullet and exit.")
    p.disconnect(physicsClientId=physics_client_id)


if __name__ == "__main__":
    main_test_script()





















