import numpy as np
import pybullet as p
import time
import logging
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.structs import Action

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s')

CFG.seed = 123
CFG.pybullet_robot = "fetch_mobile"

def test_smooth_position_control():
    """Test smooth position control for mobile robot."""
    # Create environment with GUI
    env = PyBulletBlocksEnv(use_gui=True)

    # Initialize the environment (critical step!)
    # observation = env.reset("train", 0)

    robot = env._pybullet_robot
    physics_client_id = env._physics_client_id
    
    # Get current base pose
    initial_pose = robot.get_base_pose(physics_client_id)
    print(f"Initial base pose: {initial_pose}")
    
    # Define target position
    target_pose = (0.8, 0.5, np.pi/4)
    print(f"Target pose: {target_pose}")
    
    # Create action with smooth position movement
    action = Action(np.array([], dtype=np.float32))
    action.set_base_motion(target_pose, mode="smooth_position")
    
    # Execute the action
    print("Moving robot smoothly...")
    env.step(action)
    
    # Verify final position
    final_pose = robot.get_base_pose(physics_client_id)
    print(f"Final base pose: {final_pose}")
    
    # Keep window open for inspection
    input("Press Enter to exit...")

if __name__ == "__main__":
    test_smooth_position_control()
