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
table_id = env._table_id
block_ids = env._blocks_ids

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
    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward_x = rot_matrix[0]
    forward_y = rot_matrix[3]
    forward_z = 0  # Keep it flat on the ground
    
    # Normalize the vector
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

# Function to check if a path is clear using ray casting
def is_path_clear(start, end, physics_client_id, ray_length=0.1):
    """Check if there's a clear path from start to end using ray casting"""
    # Calculate direction and distance
    direction = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:
        return True  # Start and end are the same
    
    # Normalize direction
    direction = direction / distance
    
    # Number of steps based on distance
    num_steps = max(10, int(distance / ray_length))
    
    # Cast rays along the path
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        ray_start = [
            start[0] + alpha * direction[0] * distance,
            start[1] + alpha * direction[1] * distance,
            start[2] + alpha * direction[2] * distance
        ]
        ray_end = [
            ray_start[0] + ray_length * direction[0],
            ray_start[1] + ray_length * direction[1],
            ray_start[2] + ray_length * direction[2]
        ]
        
        # Check for hits
        ray_results = p.rayTest(ray_start, ray_end, physicsClientId=physics_client_id)
        if ray_results[0][0] != -1:  # If we hit something
            # Visualize the hit point in red
            hit_pos = ray_results[0][3]
            visualize_point(hit_pos, color=(1, 0, 0), size=5, lifetime=5)
            print(f"Obstacle detected at {hit_pos}, object ID: {ray_results[0][0]}")
            return False
    
    return True

# Function to build an occupancy grid of the environment
def build_occupancy_grid(min_bounds, max_bounds, cell_size=0.1, height=0.1, physics_client_id=0):
    """Build a 2D occupancy grid of the environment"""
    grid_width = int((max_bounds[0] - min_bounds[0]) / cell_size) + 1
    grid_length = int((max_bounds[1] - min_bounds[1]) / cell_size) + 1
    
    # Initialize grid (0 = free, 1 = occupied)
    grid = np.zeros((grid_width, grid_length))
    
    # ray cast from above each cell to detect obstacles
    for i in range(grid_width):
        for j in range(grid_length):
            x = min_bounds[0] + i * cell_size
            y = min_bounds[1] + j * cell_size
            
            # Cast ray from above
            ray_start = [x, y, 1.0]  # 1 meter above
            ray_end = [x, y, -0.1]  # Slightly below ground
            
            ray_results = p.rayTest(ray_start, ray_end, physicsClientId=physics_client_id)
            
            # Check if we hit something at around table height
            if ray_results[0][0] != -1:
                hit_height = ray_results[0][3][2]
                # Check if it's likely a table (not the floor or something very high)
                if hit_height > 0.1 and hit_height < 0.5:
                    grid[i, j] = 1
                    # Visualize occupied cells
                    visualize_point([x, y, hit_height], color=(1, 0, 0), size=2, lifetime=0)
    
    print(f"Built occupancy grid with shape {grid.shape}")
    return grid, min_bounds, cell_size

# A* pathfinding algorithm
def find_path_astar(start, goal, grid, min_bounds, cell_size):
    """Find a path from start to goal using A* algorithm"""
    # Convert start and goal to grid coordinates
    start_grid = (
        int((start[0] - min_bounds[0]) / cell_size),
        int((start[1] - min_bounds[1]) / cell_size)
    )
    goal_grid = (
        int((goal[0] - min_bounds[0]) / cell_size),
        int((goal[1] - min_bounds[1]) / cell_size)
    )
    
    # Check if start or goal is in an obstacle
    if start_grid[0] < 0 or start_grid[0] >= grid.shape[0] or start_grid[1] < 0 or start_grid[1] >= grid.shape[1]:
        print("Start position is outside the grid")
        return None
    
    if goal_grid[0] < 0 or goal_grid[0] >= grid.shape[0] or goal_grid[1] < 0 or goal_grid[1] >= grid.shape[1]:
        print("Goal position is outside the grid")
        return None
    
    if grid[start_grid] == 1:
        print("Start position is in an obstacle")
        return None
    
    if grid[goal_grid] == 1:
        print("Goal position is in an obstacle")
        return None
    
    # A* algorithm
    open_set = {start_grid}
    closed_set = set()
    came_from = {}
    
    g_score = {start_grid: 0}
    f_score = {start_grid: manhattan_distance(start_grid, goal_grid)}
    
    while open_set:
        # Find node in open_set with lowest f_score
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        
        if current == goal_grid:
            # Reconstruct path
            path = reconstruct_path(came_from, current, min_bounds, cell_size)
            return path
        
        open_set.remove(current)
        closed_set.add(current)
        
        # Check neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if valid neighbor
            if (neighbor[0] < 0 or neighbor[0] >= grid.shape[0] or
                neighbor[1] < 0 or neighbor[1] >= grid.shape[1]):
                continue
            
            # Check if obstacle or already visited
            if grid[neighbor] == 1 or neighbor in closed_set:
                continue
            
            # Distance between current and neighbor
            # Use 1 for orthogonal moves, sqrt(2) for diagonal
            if dx == 0 or dy == 0:
                tentative_g_score = g_score[current] + 1
            else:
                tentative_g_score = g_score[current] + 1.414  # sqrt(2)
            
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            
            # This path is better
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal_grid)
    
    # No path found
    print("No path found")
    return None

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current, min_bounds, cell_size):
    """Reconstruct path from came_from dictionary"""
    path = []
    while current in came_from:
        # Convert grid coordinates to world coordinates
        x = min_bounds[0] + current[0] * cell_size
        y = min_bounds[1] + current[1] * cell_size
        path.append([x, y])
        current = came_from[current]
    
    # Add the start position
    x = min_bounds[0] + current[0] * cell_size
    y = min_bounds[1] + current[1] * cell_size
    path.append([x, y])
    
    # Reverse to get start-to-goal order
    path.reverse()
    return path

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
        time.sleep(0.02)
    
    print(f"Rotated by {angle_degrees} degrees")
    return True

# Function to move the robot's base safely along a path
def move_robot_along_path(robot_id, path, robot_base_z, physics_client_id):
    """Move the robot along a given path"""
    if not path or len(path) < 2:
        print("Path is too short or empty")
        return False
    
    # Get starting position and orientation
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Visualize the path
    for i in range(len(path) - 1):
        p1 = [path[i][0], path[i][1], robot_base_z + 0.05]
        p2 = [path[i+1][0], path[i+1][1], robot_base_z + 0.05]
        visualize_line(p1, p2, color=(0, 1, 0), width=2, lifetime=0)
    
    # Follow each segment of the path
    for i in range(1, len(path)):
        # Get current waypoint
        waypoint = [path[i][0], path[i][1], robot_base_z]
        
        # Get current position
        current_pos, current_orn = p.getBasePositionAndOrientation(
            robot_id, physicsClientId=physics_client_id)
        
        # Calculate distance to waypoint
        distance = np.linalg.norm(np.array(waypoint[:2]) - np.array(current_pos[:2]))
        num_steps = max(5, int(distance * 20))
        
        print(f"Moving to waypoint {i}/{len(path)-1}, distance: {distance:.2f}")
        
        # Move in steps
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            interp_pos = [
                current_pos[0] * (1 - alpha) + waypoint[0] * alpha,
                current_pos[1] * (1 - alpha) + waypoint[1] * alpha,
                robot_base_z
            ]
            
            # Move the robot
            p.resetBasePositionAndOrientation(
                robot_id, interp_pos, current_orn, physicsClientId=physics_client_id)
            
            # Visualize direction
            visualize_robot_direction(robot_id, physics_client_id=physics_client_id)
            
            # Step simulation
            for _ in range(1):
                p.stepSimulation(physicsClientId=physics_client_id)
            
            # Render
            env.render()
            time.sleep(0.01)
    
    print("Path following complete")
    return True

# Function to move and rotate the robot's base using A* path planning
def move_and_rotate_base_astar(robot_id, target_position, target_angle_degrees, grid_info, physics_client_id):
    """Move the robot to a position using A* path planning and then rotate it"""
    grid, min_bounds, cell_size = grid_info
    
    # Get current position
    current_pos, current_orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=physics_client_id)
    
    # Find path using A*
    path = find_path_astar(
        [current_pos[0], current_pos[1]],
        [target_position[0], target_position[1]],
        grid, min_bounds, cell_size
    )
    
    if path is None:
        print("Could not find path to target")
        return False
    
    # Add z-coordinate to path
    path_3d = [[p[0], p[1], target_position[2]] for p in path]
    
    # Follow the path
    success = move_robot_along_path(robot_id, path_3d, target_position[2], physics_client_id)
    
    if success:
        print("Reached target position, now rotating...")
        time.sleep(0.5)
        
        # Now rotate to the target orientation
        rotate_robot_base(robot_id, target_angle_degrees, physics_client_id)
        return True
    else:
        print("Failed to reach target position, skipping rotation")
        return False

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

# Define the environment bounds
min_bounds = [-0.5, -0.5]  # Minimum x, y coordinates
max_bounds = [3.0, 3.0]    # Maximum x, y coordinates
cell_size = 0.05           # Size of each grid cell

# Build occupancy grid of the environment
print("Building occupancy grid of the environment...")
grid_info = build_occupancy_grid(min_bounds, max_bounds, cell_size, robot_base_z, physics_client_id)

# Define a sequence of target positions
# These should be interesting positions that show the robot navigating around obstacles
positions = [
    # Front of table
    [1.65, 1.2, robot_base_z],
    # Left side of table
    [1.0, 0.75, robot_base_z],
    # Back of table
    [1.65, 0.3, robot_base_z],
    # Right side of table
    [2.3, 0.75, robot_base_z],
    # Back to initial
    [initial_pos[0], initial_pos[1], robot_base_z]
]

# Test moving to each position and rotating
print("\nDemonstrating base movements with A* path planning...")
for i, position in enumerate(positions):
    print(f"\nMovement {i+1}: Move to position {position}")
    
    # Visualize target position
    visualize_point(position, color=(0, 1, 0), size=8)
    
    # Move and rotate
    move_and_rotate_base_astar(robot.robot_id, position, 90, grid_info, physics_client_id)
    
    # Short pause between movements
    time.sleep(1.0)

# Demonstrate continuous rotation
print("\nDemonstrating continuous rotation (spinning in place)...")
for _ in range(4):  # Spin 4 quarters = full 360°
    rotate_robot_base(robot.robot_id, 90, physics_client_id, steps=15)
    time.sleep(0.2)

print("\nRotation demonstration complete!")

# Keep the window open
while True:
    time.sleep(0.1)