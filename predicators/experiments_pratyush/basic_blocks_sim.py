import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.structs import Action

# Set up the configuration for the simulation
CFG.pybullet_control_mode = "position"  # or "position" for real-time control
CFG.blocks_block_size = 0.05  # Set the block size
#CFG.blocks_num_blocks_train = 3  # Number of blocks to use in the simulation

CFG.seed = 123

# Initialize the environment
env = PyBulletBlocksEnv(use_gui=True)

# Reset the environment to get the initial state
state = env.reset("train", 0)

# Run a simple simulation loop
for _ in range(100):  # Run for 100 steps
    # Sample a random action within the action space
    action = env.action_space.sample()

    #Create an Action instance with the sampled action
    action = Action(action)

    print("Action type:", type(action))

    
    # Step the environment with the action
    state = env.step(action)

    # Print the current robot state for debugging
    print("Current robot state:", env._pybullet_robot.get_state())
    
    # Render the current state (if using GUI)
    env.render()
    
    # Sleep for a short duration to visualize the simulation
    time.sleep(0.15)

# Close the PyBullet connection
p.disconnect()