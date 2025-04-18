import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.structs import Action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.motion_planning import run_motion_planning

import os
print(os.getcwd())

# Set up the configuration for the simulation
CFG.pybullet_control_mode = "position"  # Use position control for smooth motion
CFG.blocks_block_size = 0.05  # Set the block size
CFG.seed = 123

# Initialize the environment
env = PyBulletBlocksEnv(use_gui=True)


# Get references to important objects

physics_client_id = env._physics_client_id

#Saving a video of the simulation
video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "basic_motion_test.mp4", physicsClientId=physics_client_id)

# Reset the environment to get the initial state
state = env.reset("train", 0)

robot = env._pybullet_robot


# Get initial joint positions and robot state
initial_joints = robot.get_joints()
initial_state = robot.get_state()

# Print position only to check that the value is not empty
print(f"Initial robot state: {initial_state[:3]}")  

# Define a sequence of target positions to move to
target_positions = [
    # Above starting position
    [initial_state[0], initial_state[1], initial_state[2] + 0.1],
    # Move right
    [initial_state[0] + 0.1, initial_state[1], initial_state[2] + 0.1],
    # Move forward
    [initial_state[0] + 0.1, initial_state[1] + 0.1, initial_state[2] + 0.1],
    # Move back home but higher
    [initial_state[0], initial_state[1], initial_state[2] + 0.1],
    # Back to starting position
    [initial_state[0], initial_state[1], initial_state[2]],
]

# Get a list of collision objects to avoid
num_bodies = p.getNumBodies(physics_client_id)
collision_objects = [i for i in range(num_bodies) if i != robot.robot_id]
print(f"Planning to avoid collision with {len(collision_objects)} objects")

# Test motion to each target position
for i, target_pos in enumerate(target_positions):
    print(f"\nMoving to target {i+1}: {target_pos}")
    
    # Use current orientation
    target_orn = initial_state[3:7]
    target_pose = Pose(target_pos, target_orn)
    
    try:
        # Compute inverse kinematics to get target joint positions
        target_joints = robot.inverse_kinematics(target_pose, validate=True, set_joints=False)
        print(f"Found IK solution: {[round(j, 2) for j in target_joints]}")
        
        # Plan a collision-free path
        print("Planning motion path...")
        path = run_motion_planning(
            robot=robot,
            initial_positions=robot.get_joints(),
            target_positions=target_joints,
            collision_bodies=collision_objects,
            seed=CFG.seed,
            physics_client_id=physics_client_id
        )
        
        if path is None:
            print("Motion planning failed, skipping to next target")
            continue
            
        print(f"Motion plan found with {len(path)} waypoints")
        
        # Execute the planned path
        for j, waypoint in enumerate(path):
            # Create an Action from the waypoint
            action = Action(np.array(waypoint, dtype=np.float32))
            
            # Step the environment with the action
            state = env.step(action)
            
            # Render and add a small delay
            env.render()
            time.sleep(0.05)
            
        print(f"Reached target {i+1}")
        time.sleep(0.5)  # Pause at each target
        
    except Exception as e:
        print(f"Error during motion: {e}")
        continue

# Test gripper control - open and close
print("\nTesting gripper control...")

# Open gripper
open_joints = robot.get_joints()
open_joints[robot.left_finger_joint_idx] = robot.open_fingers
open_joints[robot.right_finger_joint_idx] = robot.open_fingers
open_action = Action(np.array(open_joints, dtype=np.float32))
state = env.step(open_action)
print("Gripper opened")
env.render()
time.sleep(1.0)

# Close gripper
close_joints = robot.get_joints()
close_joints[robot.left_finger_joint_idx] = robot.closed_fingers
close_joints[robot.right_finger_joint_idx] = robot.closed_fingers
close_action = Action(np.array(close_joints, dtype=np.float32))
state = env.step(close_action)
print("Gripper closed")
env.render()
time.sleep(1.0)

print("\nMotion testing complete!")

print("Finish recording video start")
p.stopStateLogging(video_log_id)
print("Finish recording video end.")


# Keep the window open
while True:
    time.sleep(0.1)