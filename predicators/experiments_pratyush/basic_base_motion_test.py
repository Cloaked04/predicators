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
# Set to "fetch" to use a robot with a mobile base
CFG.pybullet_robot = "fetch"

# Initialize the environment
env = PyBulletBlocksEnv(use_gui=True)

# Reset the environment to get the initial state
state = env.reset("train", 0)

# Get reference to the robot
robot = env._pybullet_robot
physics_client_id = env._physics_client_id

# Define function to move the robot base
def move_robot_base(robot_id, target_position, target_orientation, physics_client_id):
    """Move the robot's base to a new position and orientation"""
    p.resetBasePositionAndOrientation(
        robot_id,
        target_position,
        target_orientation,
        physicsClientId=physics_client_id
    )
    # Allow time for the physics to settle
    for _ in range(10):
        p.stepSimulation(physicsClientId=physics_client_id)

# Get initial state
initial_state = robot.get_state()
print(f"Initial robot state: {initial_state[:3]}")

# Define base movement waypoints
base_waypoints = [
    # x, y, z
    (1.5, 0.75, 0.0),   # Move slightly forward
    (1.5, 1.0, 0.0),    # Move to the right
    (1.7, 1.0, 0.0),    # Move forward
    (1.7, 0.75, 0.0),   # Move left
    (1.3, 0.75, 0.0),   # Back to starting position
]

# Current orientation - keep it the same
current_orientation = p.getBasePositionAndOrientation(
    robot.robot_id, 
    physicsClientId=physics_client_id
)[1]

# Test moving the base to different positions
print("\nTesting base movement...")
for i, base_pos in enumerate(base_waypoints):
    print(f"\nMoving base to position {i+1}: {base_pos}")
    
    # Move the robot base
    move_robot_base(robot.robot_id, base_pos, current_orientation, physics_client_id)
    
    # After moving the base, adjust the arm to maintain its position relative to the world
    # This may require some IK planning in a real scenario, but we'll keep it simple
    
    # Render the current state
    env.render()
    
    # Pause to observe the movement
    time.sleep(1.0)
    
    # Now demonstrate arm movement at this base position
    print("Moving arm at this base position...")
    
    # Get current joint positions
    current_joints = robot.get_joints()
    
    # Move arm slightly up
    target_joints = current_joints.copy()
    arm_joint_indices = [0, 1, 2, 3]  # These may need to be adjusted based on your robot
    for idx in arm_joint_indices:
        # Small random change to the joint position
        target_joints[idx] += np.random.uniform(-0.2, 0.2)
    
    # Create an action from the target joints
    action = Action(np.array(target_joints, dtype=np.float32))
    
    # Step the environment with the action
    state = env.step(action)
    
    # Render and add a small delay
    env.render()
    time.sleep(1.0)

print("\nBase movement testing complete!")

# Keep the window open
while True:
    time.sleep(0.1)