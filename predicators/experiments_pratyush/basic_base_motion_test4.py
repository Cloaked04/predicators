import numpy as np
import pybullet as p
import time
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

# Function to get all objects in the environment
def get_all_objects_info():
    """Get information about all objects in the environment"""
    num_bodies = p.getNumBodies(physicsClientId=physics_client_id)
    objects_info = []
    
    for i in range(num_bodies):
        # Get object info
        aabb_min, aabb_max = p.getAABB(i, physicsClientId=physics_client_id)
        pos, orn = p.getBasePositionAndOrientation(i, physicsClientId=physics_client_id)
        
        # Determine object type (floor, robot, table, etc.)
        object_type = "unknown"
        if i == robot.robot_id:
            object_type = "robot"
        elif aabb_max[2] - aabb_min[2] < 0.1:  # Very flat object is likely the floor
            object_type = "floor"
        elif aabb_max[2] - aabb_min[2] > 0.2 and aabb_max[2] < 0.5:  # Medium height object is likely the table
            object_type = "table"
        
        objects_info.append({
            "id": i,
            "position": pos,
            "orientation": orn,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "size": [aabb_max[j] - aabb_min[j] for j in range(3)],
            "center": [(aabb_max[j] + aabb_min[j])/2 for j in range(3)],
            "type": object_type
        })
        
        print(f"Object {i}: Type={object_type}, AABB=[{aabb_min}, {aabb_max}], Size={[aabb_max[j] - aabb_min[j] for j in range(3)]}")
    
    return objects_info

# Improved collision detection with minimum distance check
def check_base_collision(robot_id, position, orientation, physics_client_id, min_distance=0.1):
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

# Get information about all objects in the environment
print("\nScanning environment...")
time.sleep(1)  # Give time for objects to settle
objects_info = get_all_objects_info()

# Find the table in the environment
table_info = None
for obj in objects_info:
    if obj["type"] == "table":
        table_info = obj
        break

if table_info is None:
    print("No table found in environment. Using default values.")
    # These are estimated values based on the typical PyBullet environment
    table_center = [1.65, 0.75, 0.3]
    table_size = [0.6, 0.4, 0.3]
else:
    table_center = table_info["center"]
    table_size = table_info["size"]
    print(f"Found table at {table_center} with size {table_size}")

# Visualize the table edges
table_min = [table_center[i] - table_size[i]/2 for i in range(3)]
table_max = [table_center[i] + table_size[i]/2 for i in range(3)]

visualize_line([table_min[0], table_min[1], 0.01], [table_max[0], table_min[1], 0.01], color=(1, 0, 0), width=3)
visualize_line([table_max[0], table_min[1], 0.01], [table_max[0], table_max[1], 0.01], color=(1, 0, 0), width=3)
visualize_line([table_max[0], table_max[1], 0.01], [table_min[0], table_max[1], 0.01], color=(1, 0, 0), width=3)
visualize_line([table_min[0], table_max[1], 0.01], [table_min[0], table_min[1], 0.01], color=(1, 0, 0), width=3)

# Important: Use the correct Z height for the robot base
robot_base_z = initial_pos[2]  # Keep the initial z-height

# Calculate safety margin (distance to keep from the table)
safety_margin = 0.15

# Determine which side of the table to go around based on initial position
# If robot is closer to the left side of the table, go left; otherwise go right
go_left = initial_pos[0] < table_center[0]

# Create a much simpler path with fewer waypoints
waypoints = []

if go_left:
    print("Planning path around the left side of the table")
    
    # Step 1: Move slightly to the left to clear the table
    left_pos = [table_min[0] - safety_margin, initial_pos[1], robot_base_z]
    waypoints.append(left_pos)
    
    # Step 2: Move forward past the table
    forward_pos = [table_min[0] - safety_margin, table_max[1] + safety_margin, robot_base_z]
    waypoints.append(forward_pos)
    
    # Step 3: Move right to the front center of the table
    front_pos = [table_center[0], table_max[1] + safety_margin, robot_base_z]
    waypoints.append(front_pos)
else:
    print("Planning path around the right side of the table")
    
    # Step 1: Move slightly to the right to clear the table
    right_pos = [table_max[0] + safety_margin, initial_pos[1], robot_base_z]
    waypoints.append(right_pos)
    
    # Step 2: Move forward past the table
    forward_pos = [table_max[0] + safety_margin, table_max[1] + safety_margin, robot_base_z]
    waypoints.append(forward_pos)
    
    # Step 3: Move left to the front center of the table
    front_pos = [table_center[0], table_max[1] + safety_margin, robot_base_z]
    waypoints.append(front_pos)

# Visualize all waypoints
for i, waypoint in enumerate(waypoints):
    visualize_point(waypoint, color=(0, 1, 0), size=8)
    p.addUserDebugText(
        str(i+1), 
        [waypoint[0], waypoint[1], waypoint[2]+0.1], 
        textColorRGB=(0, 1, 0),
        textSize=1.5,
        physicsClientId=physics_client_id
    )

# Test moving the base to different positions
print("\nNavigating around the table...")
for i, waypoint in enumerate(waypoints):
    print(f"\nMoving to waypoint {i+1}: {waypoint}")
    
    # Try to move the robot base safely
    success = move_robot_base_safely(
        robot.robot_id, waypoint, initial_orn, physics_client_id)
    
    if success:
        print(f"Successfully moved to waypoint {i+1}")
        # Add a small delay at each waypoint for visibility
        time.sleep(0.5)
    else:
        print(f"Failed to move to waypoint {i+1} due to collisions")
        
        # If a waypoint fails, try adjusting it slightly and retry
        adjusted_waypoint = list(waypoint)
        
        # Try moving the waypoint slightly further away from the table
        if go_left:
            adjusted_waypoint[0] -= 0.05  # Move more to the left
        else:
            adjusted_waypoint[0] += 0.05  # Move more to the right
            
        print(f"Trying adjusted waypoint: {adjusted_waypoint}")
        success = move_robot_base_safely(
            robot.robot_id, adjusted_waypoint, initial_orn, physics_client_id)
            
        if not success:
            print("Still cannot reach waypoint, continuing to next one")
            continue

print("\nNavigation complete!")

# Keep the window open
while True:
    time.sleep(0.1)