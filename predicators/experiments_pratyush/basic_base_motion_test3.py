import numpy as np
import pybullet as p
import time
import math
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.structs import Action
from predicators.pybullet_helpers.geometry import Pose

# Set up the configuration for the simulation
CFG.pybullet_control_mode = "position"
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

# Function to visualize a point for debugging purposes
def visualize_point(position, color=(1, 0, 0), size=0.05, lifetime=0):
    return p.addUserDebugPoints(
        [position], 
        [color], 
        pointSize=size,
        lifeTime=lifetime,
        physicsClientId=physics_client_id
    )

# Function to visualize a line for debugging
def visualize_line(start, end, color=(0, 1, 0), width=2, lifetime=0):
    return p.addUserDebugLine(
        start, 
        end, 
        color,
        width,
        lifeTime=lifetime,
        physicsClientId=physics_client_id
    )

# Improved collision detection with minimum distance check
def check_base_collision(robot_id, position, orientation, physics_client_id, min_distance=0.05):
    """
    Check if moving the robot to this position would cause a collision
    or be too close to obstacles
    """
    # Save current position
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Temporarily move robot to check collision
    p.resetBasePositionAndOrientation(
        robot_id, position, orientation, physicsClientId=physics_client_id)
    
    # Perform collision detection
    p.performCollisionDetection(physicsClientId=physics_client_id)
    
    # Check for collisions or close contacts with all other objects
    collision = False
    num_bodies = p.getNumBodies(physicsClientId=physics_client_id)
    
    # Get all contact points for more detailed information
    for i in range(num_bodies):
        if i != robot_id:
            # Get closest points between robot and other bodies
            closest_points = p.getClosestPoints(
                bodyA=robot_id, 
                bodyB=i, 
                distance=min_distance,
                physicsClientId=physics_client_id
            )
            
            # Check if any points are too close
            for point in closest_points:
                distance = point[8]  # Distance between objects
                if distance < min_distance:
                    linkA = point[3]  # Link index on robot
                    linkB = point[4]  # Link index on other body
                    posA = point[5]   # Contact position on robot
                    posB = point[6]   # Contact position on other body
                    
                    # Visualize collision points
                    visualize_point(posA, color=(1, 0, 0), size=10, lifetime=1)
                    visualize_point(posB, color=(0, 0, 1), size=10, lifetime=1)
                    
                    print(f"Too close to object {i}, link {linkB}, distance: {distance}")
                    collision = True
                    break
            
            # Also check actual collisions
            contact_points = p.getContactPoints(
                bodyA=robot_id, 
                bodyB=i, 
                physicsClientId=physics_client_id
            )
            
            if len(contact_points) > 0:
                collision = True
                print(f"Collision detected with body {i}")
                break
    
    # Restore original position
    p.resetBasePositionAndOrientation(
        robot_id, current_pos, current_orn, physicsClientId=physics_client_id)
    
    return collision

# Improved function to move the robot base with better collision checking
def move_robot_base_safely(robot_id, target_position, target_orientation, physics_client_id):
    """Move the robot's base to a new position and orientation, avoiding collisions"""
    # Get current position and orientation
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Calculate a path from current to target (simple linear interpolation)
    distance = np.linalg.norm(np.array(target_position[:2]) - np.array(current_pos[:2]))
    num_steps = max(10, int(distance * 20))  # More steps for longer distances
    
    print(f"Planning path with {num_steps} steps over distance {distance:.2f}")
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
    
    # Visualize the planned path
    for i in range(len(path) - 1):
        visualize_line(path[i][0], path[i+1][0], color=(0, 1, 0), width=2, lifetime=5)
    
    # Follow the path, checking for collisions
    for step_pos, step_orn in path:
        if check_base_collision(robot_id, step_pos, step_orn, physics_client_id):
            print(f"Cannot move to {step_pos} due to collision")
            return False
        
        # Move to this waypoint
        p.resetBasePositionAndOrientation(
            robot_id, step_pos, step_orn, physicsClientId=physics_client_id)
        
        # Allow time for physics to settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=physics_client_id)
        
        # Render
        env.render()
        time.sleep(0.05)  # Faster movement
    
    return True

# Get initial robot state
initial_pos, initial_orn = p.getBasePositionAndOrientation(
    robot.robot_id, physicsClientId=physics_client_id)
print(f"Initial robot base position: {initial_pos}")

# Get room dimensions by examining the environment
# This assumes the environment has objects that define its boundaries
room_min = [float('inf'), float('inf'), float('inf')]
room_max = [float('-inf'), float('-inf'), float('-inf')]

num_bodies = p.getNumBodies(physicsClientId=physics_client_id)
for i in range(num_bodies):
    if i != robot.robot_id:
        aabb_min, aabb_max = p.getAABB(i, physicsClientId=physics_client_id)
        room_min = [min(room_min[j], aabb_min[j]) for j in range(3)]
        room_max = [max(room_max[j], aabb_max[j]) for j in range(3)]

print(f"Environment boundaries: Min {room_min}, Max {room_max}")

# Define a safe area for robot movement
# Add margins to stay away from obstacles
margin = 0.3
safe_min = [room_min[0] + margin, room_min[1] + margin, initial_pos[2]]
safe_max = [room_max[0] - margin, room_max[1] - margin, initial_pos[2]]

# Visualize the safe area
visualize_line([safe_min[0], safe_min[1], 0], [safe_max[0], safe_min[1], 0], color=(0, 0, 1), width=2)
visualize_line([safe_max[0], safe_min[1], 0], [safe_max[0], safe_max[1], 0], color=(0, 0, 1), width=2)
visualize_line([safe_max[0], safe_max[1], 0], [safe_min[0], safe_max[1], 0], color=(0, 0, 1), width=2)
visualize_line([safe_min[0], safe_max[1], 0], [safe_min[0], safe_min[1], 0], color=(0, 0, 1), width=2)

# Important: Use the correct Z height for the robot base
robot_base_z = initial_pos[2]  # Keep the initial z-height

# Define more diverse base movement waypoints in the safe area
base_waypoints = [
    # Move in a rectangular pattern
    (initial_pos[0] - 0.2, initial_pos[1], robot_base_z),      # Move backward
    (initial_pos[0] - 0.2, initial_pos[1] + 0.2, robot_base_z), # Move right
    (initial_pos[0], initial_pos[1] + 0.2, robot_base_z),      # Move forward
    (initial_pos[0], initial_pos[1] - 0.2, robot_base_z),      # Move left
    (initial_pos[0] - 0.2, initial_pos[1] - 0.2, robot_base_z), # Move diagonal
    (initial_pos[0], initial_pos[1], robot_base_z),            # Return to start
]

# Test moving the base to different positions
print("\nTesting base movement with improved collision avoidance...")
for i, base_pos in enumerate(base_waypoints):
    print(f"\nMoving base to position {i+1}: {base_pos}")
    
    # Visualize the target position
    visualize_point(base_pos, color=(0, 1, 0), size=10)
    
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
        time.sleep(0.5)
    else:
        print(f"Failed to move to position {i+1} due to collisions")

print("\nBase movement testing complete!")

# Keep the window open
while True:
    time.sleep(0.1)