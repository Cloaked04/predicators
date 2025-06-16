import sys
import numpy as np
import pybullet as p
import time
import logging
from typing import List, Tuple, Optional, Sequence, Collection, Dict, Any
import random

from predicators.structs import Action, Array, GroundAtom, Object, State, Type, ParameterizedOption
from predicators import utils
from predicators.settings import CFG

#Import core environment methods, robot function etc.

from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_single_arm import MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions, get_joint_infos, get_joint_positions
from predicators.pybullet_helpers.link import get_link_state

#Import the functions that are to be tested:

from predicators.pybullet_helpers.motion_planning import run_motion_planning, run_base_motion_planning,\
                                                            run_coordinated_motion_planning
#The pick/place options to be tested are accessed via the env instance
from predicators.pybullet_helpers.controllers import execute_coordinated_path

#Configure logging for better debugging outputs:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
CFG.pybullet_sim_steps_per_action = 10

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
    initial_state_from_env = env.reset("train", 0)

    # The robot instance from the environment
    robot = env._pybullet_robot
    # The PyBullet physics client ID
    physics_client_id = env._physics_client_id

    if not isinstance(robot, MobileSingleArmPyBulletRobot):
        logging.error("This test script is designed for a MobileSingleArmPyBulletRobot.")
        return


    logging.info(f"Using robot: {robot.get_name()}")

    #Test variables/initial parametrs of the env:

    #Store the robot's default arm and finger joint positions.
    home_arm_joints = robot.initial_joint_positions

    #Store permament, fixed bodies
    static_collision_bodies = get_all_non_robot_bodies(robot.robot_id, physics_client_id)

    print(f"Static_collision_bodies:{static_collision_bodies}")

    #sys.exit(0)

    #Define a rectangular workspace for base motion planning tests.
    #(min_x, min_y, max_x, max_y)
    workspace_bounds = (1.0, 0.2, 1.7, 1.3)



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

        # Iterate through each Action object in the provided list.
        for act_idx, act in enumerate(actions):
            logging.debug(f" Executing action {act_idx+1}/{len(actions)}")

            #Debug prints before the taking action:
            base_pos, base_orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=physics_client_id)
            print(f"[DEBUG] Before action {act_idx}: base_pos = {base_pos}")
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
                   robot.set_motors(current_arm_js) 
                else:
                    logging.warning("    Finger joint indices out of bounds for gripper command in viz.")


            # Step simulation to see the effect of the motors/teleport
            for _ in range(CFG.pybullet_sim_steps_per_action):
                 p.stepSimulation(physicsClientId=physics_client_id)
            time.sleep(0.08) # Pause briefly for smoother viewing

            #Debug prints after taking action
            base_pos, base_orn = p.getBasePositionAndOrientation(robot.robot_id, physicsClientId=physics_client_id)
            # Extract only the x, y, and yaw, discarding unstable roll/pitch/z.
            x, y, _ = base_pos
            _, _, yaw = p.getEulerFromQuaternion(base_orn)

            # Define a stable pose with a fixed z-height and zero roll/pitch.
            stable_pos = (x, y, 0.01) # Keep robot on the floor (z=0). Adjust if your floor is different.
            stable_orn = p.getQuaternionFromEuler([0, 0, yaw])

            #print(f"[DEBUG] After action {act_idx}: base_pos = {base_pos}")
            # fixed_z = 0.05
            # fixed_base_pos = (base_pos[0], base_pos[1], fixed_z)
            p.resetBasePositionAndOrientation(robot.robot_id, stable_pos, stable_orn, physicsClientId=physics_client_id)
        logging.info(f"{test_name} visualization complete.")



    #-----------------------------------------------------------
    ##### Test 1: test the run_motion_planning function in motion_planning.py (arm only).
    #-----------------------------------------------------------

    # logging.info("\n----Test 1: run_motion_planning (Arm Only)---")

    # #Reset robot to known starting place
    # reset_robot_fetch_mobile(robot, physics_client_id, base_pose=(0.80, 0.70, 0.0), arm_joint_angle=home_arm_joints)

    # # Create an obstacle for the arm to navigate around.
    # obstacle_block_id_1 = create_test_block(env, pose=(1.45, 0.75, CFG.blocks_block_size / 2 + env.table_height + 0.1))
    # current_collision_bodies_1 = get_all_non_robot_bodies(robot.robot_id, physics_client_id)

    # # Define a target arm joint configuration (slightly different from home).
    # target_arm_joints_1 = list(home_arm_joints)
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

    # logging.info("\n--- Test 2: run_base_motion_planning (Base Only) ---")
    # initial_base_pose_2 = (-2.40, 0.30, 0.0) # Starting base pose
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
    #         robot.set_joints(home_arm_joints) # Keep arm static
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


    # #-----------------------------------------------------------
    # ##### Test 4: test the run_coordinated_motion_planning function in motion_planning.py (base+arm only).
    # #-----------------------------------------------------------

    # logging.info("\n--- Test 4: Coordinated Motion (Base + Arm) ---")
    # # Start further away and rotated, likely needing base movement
    # initial_base_pose_4 = (0.4, 0.6, -np.pi/2) 
    # reset_robot_fetch_mobile(robot, physics_client_id, base_pose=initial_base_pose_4, arm_joint_angle=home_arm_joints)
    # print(f"[DEBUG] Home arm joints:{home_arm_joints}.")
    # print(f"[DEBUG] Robot's joints after reset in test 4:{robot.get_joints()}.")
    # print(f"[DEBUG] Symbolic state joints (STALE):    {env._current_state.simulator_state}")
    # #sys.exit(0)

    # # Define a target EE pose that's likely out of reach for arm-only from initial_base_pose_4.
    # # A point on the table 1.35, 0.6, 0.2
    # z = env.table_height + CFG.blocks_block_size/2 + 0.10
    # orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])
    # target_ee_pose_4 = Pose(position=(1.5, 0.75, 0.3), 
    #                          orientation=robot._ee_home_pose.orientation)

    # logging.info(f"Testing coordinated motion (expect base + arm) to EE pose: {target_ee_pose_4}")
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
    #     logging.info(f"Coordinated path (base+arm) found: Base waypoints: {len(base_path_4)}, Arm waypoints: {len(arm_path_4)}")
    #     # Expect base_path_4 to have multiple waypoints.
    #     actions_4 = execute_coordinated_path(robot, base_path_4, arm_path_4, physics_client_id)
    #     visualize_action_sequence(actions_4, "Test 4 Coordinated (Base+Arm)")
    # else:
    #     logging.warning("Coordinated motion (base+arm) planning failed for Test 4.")


    # input("Test 4 Finished. Press Enter to continue...")

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
    try:
        torso_joint_id = robot.joint_from_name("torso_lift_joint")
        # Get joint limits for the torso
        torso_joint_info = p.getJointInfo(robot.robot_id, torso_joint_id, physics_client_id)
        torso_lower_limit = torso_joint_info[8]
        # Set the torso to its lowest position.
        p.resetJointState(robot.robot_id,
                          torso_joint_id,
                          targetValue=torso_lower_limit,
                          physicsClientId=physics_client_id)
        logging.info(f"Torso lowered to {torso_lower_limit:.3f}m.")
    except ValueError:
        logging.warning("Could not find torso_lift_joint. Proceeding without lowering torso.")


    # ADD THIS DEBUG BLOCK
    print("\n[DEBUG] --- Right after Test 5 reset ---")
    print(f"\n[DEBUG] Home arm joints:{home_arm_joints}.")
    print(f"[DEBUG] Actual robot joints from PyBullet: {robot.get_joints()}")
    print(f"[DEBUG] Symbolic state joints:    {env._current_state.simulator_state}")
    # END DEBUG BLOCK

    #sys.exit(0)

    # Create a block to be picked.
    block_to_pick_pose_world = (1.5, 0.75, CFG.blocks_block_size / 2 + env.table_height)
    print(f"[DEBUG] World coords of block to pick:{block_to_pick_pose_world}.")
    #sys.exit(0)
    block_to_pick_id = create_test_block(env, pose=block_to_pick_pose_world, name_suffix="pick_target")


    # --- Symbolic State Setup for Option ---
    # For option testing, we need a symbolic State that the option's policy can understand.
    # In a real run, PyBulletEnv.step() would update env._current_state. Here, we mock it.
    
    # Make a unique name for the symbolic object
    symbolic_block_name = f"block{block_to_pick_id}" 
    block_to_pick_obj_sym = Object(symbolic_block_name, env._block_type)

    # Step 1: Update the environment's physical-to-symbolic map.
    # This tells _get_state() that the new physical block ID now corresponds
    # to a new symbolic object.
    env._block_id_to_block[block_to_pick_id] = block_to_pick_obj_sym

    # Step 2: Get a handle to the official state object, which is mutable.
    state_obj_to_modify = env._current_observation
    assert isinstance(state_obj_to_modify, utils.PyBulletState), \
        f"Expected env._current_observation to be a PyBulletState, got {type(state_obj_to_modify)}"

    # Step 3: "Prime" the state object with a placeholder for the new block.
    # This is ONLY to satisfy the assertion inside _get_state().
    state_obj_to_modify.data[block_to_pick_obj_sym] = np.zeros(len(env._block_type.feature_names))

    # Step 4: Now, create a fresh state. The assertion will pass because the object
    # returned by the _current_state property (which is state_obj_to_modify)
    # now has the same objects as the newly created state.
    fresh_state = env._get_state()

    # Step 5: Update the official state object IN-PLACE with the fresh data.
    # We are not assigning to _current_state; we are modifying the object it points to.
    #print(f"[DEBUG] Attributes of state_obj_to_modify: {dir(state_obj_to_modify)}")
    state_obj_to_modify.data = fresh_state.data
    state_obj_to_modify.simulator_state = fresh_state.simulator_state 
    state_obj_to_modify.base_pose = fresh_state.base_pose
    #sys.exit(0)

    # Crucial: ensure robot starts not holding anything. PyBulletEnv uses this.
    env._held_obj_id = None
    # The symbolic robot object, already in env.types and env._get_state()
    robot_obj_sym = env._robot 

    logging.info(f"Testing PickObject option for block ID {block_to_pick_id} ('{block_to_pick_obj_sym.name}') at {block_to_pick_pose_world}")
    pick_option = env._pick_block_option # Get the option instance from the environment

    # Check if the option thinks it can start.
    if not pick_option.initiable(env._current_state, {}, [robot_obj_sym, block_to_pick_obj_sym], np.array([])):
        logging.error("PickOption is not initiable. Check state/predicates (e.g., GripperOpen, Clear).")
    else:
        logging.info("PickOption is initiable. Executing policy (will plan and generate actions)...")
        option_memory_pick: Dict = {} # Option policies can use memory
        actions_for_pick: List[Action] = []

        try:
            # The option's policy, when called, will:
            # 1. Internally call get_target_ee_pose_for_pick.
            # 2. Call run_coordinated_motion_planning to plan base/arm moves to pre-grasp.
            # 3. Call execute_coordinated_path to get a list of low-level Actions.
            #    This list also includes actions for the final finger closing.
            # 4. Store these actions in option_memory_pick["actions"].
            # 5. Return the first action from this list.
            # We simulate multiple calls to the policy to get all actions.
            while not pick_option.terminal(env._current_state, option_memory_pick, [robot_obj_sym, block_to_pick_obj_sym], np.array([])):
                 act = pick_option.policy(env._current_state, option_memory_pick, [robot_obj_sym, block_to_pick_obj_sym], np.array([]))
                 actions_for_pick.append(act)
                 # In a real scenario, env._current_state would be updated after each env.step(act)
                 # For this test visualization, we execute all planned actions sequentially from the initial state.
                 # The terminal condition checks memory, so it will become true after all planned actions are retrieved.

            logging.info(f"PickOption planned {len(actions_for_pick)} actions.")
            visualize_action_sequence(actions_for_pick, "Test 5 PickOption")

            # --- Verification of Pick ---
            # After actions, the environment's internal state (_held_obj_id, constraints)
            # should reflect the grasp. This is normally handled by PyBulletEnv.step().
            # For this test, we manually trigger the grasp detection and constraint creation
            # as if the finger-closing actions were processed by PyBulletEnv.step().

            # Check if fingers are now around the object
            env._held_obj_id = env._detect_held_object() 
            if env._held_obj_id == block_to_pick_id:
                env._create_grasp_constraint() # Create the PyBullet fixed constraint
                logging.info(f"Pick successful! Robot is now holding block {env._held_obj_id} ('{block_to_pick_obj_sym.name}').")
            else:
                logging.error(f"Pick appears to have failed. Detected held PyBullet ID: {env._held_obj_id}, expected {block_to_pick_id}.")

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
                    if np.allclose(final_block_pos, place_target_world_pos, atol=0.05): # Tolerance for placement
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





















