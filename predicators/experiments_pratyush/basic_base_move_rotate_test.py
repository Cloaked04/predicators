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

# Function to visualize the robot's forward direction
def visualize_robot_direction(robot_id, length=0.3, physics_client_id=0):
    pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)
    
    # Calculate forward direction vector from quaternion
    # This creates a unit vector pointing in the robot's forward direction
    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward_x = rot_matrix[0]
    forward_y = rot_matrix[3]
    forward_z = 0  # Keep it flat on the ground
    
    # Normalize the vector to have unit length
    magnitude = math.sqrt(forward_x*forward_x + forward_y*forward_y + forward_z*forward_z)
    if magnitude > 0:
        forward_x /= magnitude
        forward_y /= magnitude
        forward_z /= magnitude
    
    # Calculate end point
    end_point = [
        pos[0] + forward_x * length,
        pos[1] + forward_y * length,
        pos[2] + forward_z * length
    ]
    
    # Draw line from robot position to end point
    return visualize_line(pos, end_point, color=(0, 0, 1), width=4, lifetime=0.1)

# Function to check if position is in bounds
def is_position_in_bounds(position, min_bounds, max_bounds):
    """Check if a position is within the specified bounds"""
    return (position[0] >= min_bounds[0] and position[0] <= max_bounds[0] and
            position[1] >= min_bounds[1] and position[1] <= max_bounds[1])

# Disable collision detection for specific object pairs
def disable_collisions_for_robot(robot_id, physics_client_id):
    """Disable self-collisions for the robot to allow free movement"""
    num_bodies = p.getNumBodies(physicsClientId=physics_client_id)
    for i in range(num_bodies):
        if i != robot_id:
            # If this is a block or small object, disable collisions with the robot base
            # This allows the robot to move freely around the environment
            aabb_min, aabb_max = p.getAABB(i, physicsClientId=physics_client_id)
            size = [aabb_max[j] - aabb_min[j] for j in range(3)]
            
            # If it's not the ground (which should have collision enabled)
            if size[2] > 0.01:  # Not completely flat
                p.setCollisionFilterPair(robot_id, i, -1, -1, 0, physicsClientId=physics_client_id)
                print(f"Disabled collisions between robot and object {i}")

# Function to rotate the robot's base by a specific angle
def rotate_robot_base(robot_id, angle_degrees, physics_client_id, steps=20):
    """
    Rotate the robot's base by the specified angle in degrees.
    Positive angle = counter-clockwise, negative angle = clockwise.
    """
    angle_radians = math.radians(angle_degrees)
    
    # Get current position and orientation
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Convert quaternion to Euler angles
    current_euler = p.getEulerFromQuaternion(current_orn)
    
    # Calculate target yaw (rotation around z-axis)
    target_yaw = current_euler[2] + angle_radians
    
    # Keep the same roll and pitch
    target_euler = (current_euler[0], current_euler[1], target_yaw)
    
    # Convert back to quaternion
    target_orn = p.getQuaternionFromEuler(target_euler)
    
    # Rotate in small steps
    for i in range(1, steps + 1):
        # Interpolate between current and target orientation
        alpha = i / steps
        
        # Spherical linear interpolation (SLERP) between orientations
        interp_orn = p.getQuaternionSlerp(current_orn, target_orn, alpha)
        
        # Apply the rotation
        p.resetBasePositionAndOrientation(
            robot_id, current_pos, interp_orn, physicsClientId=physics_client_id)
        
        # Visualize the robot's direction
        visualize_robot_direction(robot_id, physics_client_id=physics_client_id)
        
        # Allow time for physics to settle
        for _ in range(2):
            p.stepSimulation(physicsClientId=physics_client_id)
        
        # Render
        env.render()
        time.sleep(0.05)
    
    print(f"Rotated by {angle_degrees} degrees")
    return True

# Function to move and rotate the robot's base
def move_and_rotate_base(robot_id, target_position, target_angle_degrees, physics_client_id):
    """Move the robot to a position and then rotate it"""
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Calculate path to target position
    distance = np.linalg.norm(np.array(target_position[:2]) - np.array(current_pos[:2]))
    num_steps = max(10, int(distance * 20))
    
    print(f"Moving to position {target_position}, distance {distance:.2f}")
    
    # Move in steps
    for i in range(1, num_steps + 1):
        alpha = i / num_steps
        interp_pos = [
            current_pos[0] * (1 - alpha) + target_position[0] * alpha,
            current_pos[1] * (1 - alpha) + target_position[1] * alpha,
            target_position[2]
        ]
        
        # Move the robot
        p.resetBasePositionAndOrientation(
            robot_id, interp_pos, current_orn, physicsClientId=physics_client_id)
        
        # Visualize direction
        visualize_robot_direction(robot_id, physics_client_id=physics_client_id)
        
        # Step simulation
        for _ in range(2):
            p.stepSimulation(physicsClientId=physics_client_id)
        
        # Render
        env.render()
        time.sleep(0.02)
    
    print("Reached target position, now rotating...")
    time.sleep(0.5)
    
    # Now rotate to the target orientation
    rotate_robot_base(robot_id, target_angle_degrees, physics_client_id)
    
    return True

# Define a safe operation area for the robot
def get_safe_operation_area():
    """Define a safe area for robot movement based on the environment"""
    # Define a default safe area
    min_bounds = [0.0, 0.0, 0.0]
    max_bounds = [2.0, 2.0, 0.0]
    
    # Scan for objects and adjust area
    num_bodies = p.getNumBodies(physicsClientId=physics_client_id)
    for i in range(num_bodies):
        if i != robot.robot_id:
            aabb_min, aabb_max = p.getAABB(i, physicsClientId=physics_client_id)
            # If this is a large object (like the table)
            size = [aabb_max[j] - aabb_min[j] for j in range(3)]
            if size[2] > 0.2:  # Taller than 20cm (probably the table)
                # Expand the safe area to cover outside the table
                min_bounds[0] = min(min_bounds[0], aabb_min[0] - 0.5)
                min_bounds[1] = min(min_bounds[1], aabb_min[1] - 0.5)
                max_bounds[0] = max(max_bounds[0], aabb_max[0] + 0.5)
                max_bounds[1] = max(max_bounds[1], aabb_max[1] + 0.5)
    
    return min_bounds, max_bounds

# Get initial robot state
initial_pos, initial_orn = p.getBasePositionAndOrientation(
    robot.robot_id, physicsClientId=physics_client_id)
print(f"Initial robot base position: {initial_pos}")
print(f"Initial robot orientation (quaternion): {initial_orn}")

# Convert initial orientation to Euler angles for better understanding
initial_euler = p.getEulerFromQuaternion(initial_orn)
print(f"Initial robot orientation (Euler angles): {[math.degrees(angle) for angle in initial_euler]}")

# Important: Use the correct Z height for the robot base
robot_base_z = initial_pos[2]  # Keep the initial z-height

# Disable collisions to allow free movement
print("\nDisabling unnecessary collisions...")
disable_collisions_for_robot(robot.robot_id, physics_client_id)

# Get safe area for operation
min_bounds, max_bounds = get_safe_operation_area()
print(f"Safe operation area: Min {min_bounds}, Max {max_bounds}")

# Visualize the safe operation area
visualize_line([min_bounds[0], min_bounds[1], 0.01], [max_bounds[0], min_bounds[1], 0.01], color=(0, 1, 1), width=2)
visualize_line([max_bounds[0], min_bounds[1], 0.01], [max_bounds[0], max_bounds[1], 0.01], color=(0, 1, 1), width=2)
visualize_line([max_bounds[0], max_bounds[1], 0.01], [min_bounds[0], max_bounds[1], 0.01], color=(0, 1, 1), width=2)
visualize_line([min_bounds[0], max_bounds[1], 0.01], [min_bounds[0], min_bounds[1], 0.01], color=(0, 1, 1), width=2)

# Define a sequence of moves and rotations to demonstrate robot mobility
# Ensure all positions are within the safe area
movement_sequence = [
    # (x, y, z, rotation_angle)
    (min(max_bounds[0], initial_pos[0] + 0.2), initial_pos[1], robot_base_z, 90),
    (min(max_bounds[0], initial_pos[0] + 0.2), min(max_bounds[1], initial_pos[1] + 0.2), robot_base_z, 90),
    (initial_pos[0], min(max_bounds[1], initial_pos[1] + 0.2), robot_base_z, 90),
    (initial_pos[0], initial_pos[1], robot_base_z, 90),
]

# Test the sequence of movements and rotations
print("\nDemonstrating base movements and rotations...")
for i, (x, y, z, angle) in enumerate(movement_sequence):
    print(f"\nMovement {i+1}: Move to ({x}, {y}, {z}) and rotate {angle}°")
    
    # Ensure position is valid
    if not is_position_in_bounds([x, y, z], min_bounds, max_bounds):
        print(f"Position ({x}, {y}, {z}) is out of bounds, skipping.")
        continue
    
    # Visualize target position
    visualize_point([x, y, z], color=(0, 1, 0), size=8)
    
    # Move and rotate
    move_and_rotate_base(robot.robot_id, [x, y, z], angle, physics_client_id)
    
    # Short pause between movements
    time.sleep(1.0)

# Demonstrate continuous rotation (robot spinning in place)
print("\nDemonstrating continuous rotation (spinning in place)...")
for _ in range(4):  # Spin 4 quarters = full 360°
    rotate_robot_base(robot.robot_id, 90, physics_client_id, steps=15)
    time.sleep(0.2)

print("\nRotation demonstration complete!")

# Keep the window open
while True:
    time.sleep(0.1)