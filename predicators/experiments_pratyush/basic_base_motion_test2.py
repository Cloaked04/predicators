import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.structs import Action
from predicators.pybullet_helpers.geometry import Pose

import sys

# Set up the configuration for the simulation
#CFG.pybullet_control_mode = "position"
CFG.blocks_block_size = 0.05
CFG.seed = 123
CFG.pybullet_robot = "fetch"

# Initialize the environment
env = PyBulletBlocksEnv(use_gui=True)

# Reset the environment to get the initial state
state = env.reset("train", 0)

# Get reference to the robot and environment
robot = env._pybullet_robot
physics_client_id = env._physics_client_id
table_id = env._table_id
block_ids = env._block_ids

table_position, table_orientation = p.getBasePositionAndOrientation(table_id, physicsClientId=physics_client_id)


# getAABB(linkIndex=-1) returns ((minX,minY,minZ),(maxX,maxY,maxZ))
# aabb_min, aabb_max = p.getAABB(table_id,
#                                linkIndex=-1,
#                                physicsClientId=physics_client_id)

# # extract the 2D rectangle corners in the table plane
# corners_2d = np.array([
#     (aabb_min[0], aabb_min[1]),
#     (aabb_min[0], aabb_max[1]),
#     (aabb_max[0], aabb_max[1]),
#     (aabb_max[0], aabb_min[1]),
# ], dtype=np.float32)


# print("Table footprint corners (XY):")
# for x, y in corners_2d:
#     print(f"  ({x:.3f}, {y:.3f})")

shapes = p.getCollisionShapeData(table_id, linkIndex=-1, physicsClientId=physics_client_id)





print("Printing ids for items in the env:")
print("-----------------------------------")
print(f"Robot id:{robot.robot_id}")
print(f"Table id:{table_id}")
print(f"Block ids:{block_ids}")
print("-----------------------------------")
print(f"Table's position and orientation:{table_position, table_orientation}")
print(f"Shape:{shapes}")

#sys.exit()


# Function to check if moving to a position would cause collision
def check_base_collision(robot_id, position, orientation, physics_client_id):
    """Check if moving the robot to this position would cause a collision"""
    # Save current position
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)

    print(f"Current position and orientation while program in check_base_collision:{current_pos, current_orn}")
    
    # Temporarily move robot to check collision
    p.resetBasePositionAndOrientation(
        robot_id, position, orientation, physicsClientId=physics_client_id)

    print(f"Current position and orientation after temporary move:{p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)}")

    #print("Moved Body temporarily.")
    
    # Perform collision detection
    p.performCollisionDetection(physicsClientId=physics_client_id)
    #print("Performed Collision Detection")
    
    # Check for collisions with all other objects
    collision = False
    num_bodies = p.getNumBodies(physicsClientId=physics_client_id)
    for i in range(num_bodies):
        #print(i)
        if i != robot_id:
            contact_points = p.getContactPoints(
                bodyA=robot_id, bodyB=i, physicsClientId=physics_client_id)
            if len(contact_points) > 0:
                collision = True
                print(f"Collision detected with body {i}")
                break
    
    # Restore original position
    p.resetBasePositionAndOrientation(
        robot_id, current_pos, current_orn, physicsClientId=physics_client_id)
    
    return collision

# Function to move the robot base with collision checking
def move_robot_base_safely(robot_id, target_position, target_orientation, physics_client_id):
    """Move the robot's base to a new position and orientation, avoiding collisions"""
    # Get current position and orientation
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Calculate a path from current to target (simple linear interpolation)
    num_steps = 10
    path = []
    
    for i in range(1, num_steps + 1):
        # Linear interpolation between current and target position
        alpha = i / num_steps
        interp_pos = [
            current_pos[0] * (1 - alpha) + target_position[0] * alpha,
            current_pos[1] * (1 - alpha) + target_position[1] * alpha,
            target_position[2]  # Keep z constant
        ]
        path.append((interp_pos, target_orientation))
    
    print(path)

    # Follow the path, checking for collisions
    for step_pos, step_orn in path:
        if check_base_collision(robot_id, step_pos, step_orn, physics_client_id):
            print(f"Cannot move to {step_pos} due to collision")
            return False
        
        # Move to this waypoint
        #print('Moving to waypoint')
        p.resetBasePositionAndOrientation(
            robot_id, step_pos, step_orn, physicsClientId=physics_client_id)
        
        # Allow time for physics to settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=physics_client_id)
        
        # Render
        env.render()
        time.sleep(0.1)
    
    return True

# Get initial robot state
initial_pos, initial_orn = p.getBasePositionAndOrientation(
    robot.robot_id, physicsClientId=physics_client_id)
print(f"Initial robot base position: {initial_pos}")

# Important: Determine the correct Z height for the robot base
# This should be the height that positions the robot properly on top of the table
# For Fetch, this is likely the height of its wheels
robot_base_z = initial_pos[2]  # Keep the initial z-height

# Define base movement waypoints (keeping z-height constant)
base_waypoints = [
    # x, y, z (z stays constant)
    (1.4, 0.85, robot_base_z),
    (1.5, 0.85, robot_base_z),
    (1.5, 0.65, robot_base_z),
    (1.3, 0.65, robot_base_z),
    (1.3, 0.75, robot_base_z),
]

# Test moving the base to different positions
print("\nTesting base movement with collision avoidance...")
for i, base_pos in enumerate(base_waypoints):
    print(f"\nMoving base to position {i+1}: {base_pos}")
    
    # Try to move the robot base safely
    success = move_robot_base_safely(
        robot.robot_id, base_pos, initial_orn, physics_client_id)
    
    if success:
        print(f"Successfully moved to position {i+1}")
        # Demonstrate arm movement at this base position
        print("Moving arm at this base position...")
        
        # Get current joint positions
        current_joints = robot.get_joints()
        
        # Move arm slightly
        target_joints = current_joints.copy()
        arm_joint_indices = [0, 1, 2, 3]  # Adjust based on your robot
        for idx in arm_joint_indices:
            target_joints[idx] += np.random.uniform(-0.1, 0.1)
        
        # Create an action from the target joints
        action = Action(np.array(target_joints, dtype=np.float32))
        
        # Step the environment with the action
        state = env.step(action)
        
        # Render and add a small delay
        env.render()
        time.sleep(1.0)
    else:
        print(f"Failed to move to position {i+1} due to collisions")

print("\nBase movement testing complete!")

# Keep the window open
while True:
    time.sleep(0.1)