import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.pybullet_helpers.motion_planning import run_base_motion_planning

def test_birrt_base_planning():
    """Test BiRRT planning for mobile robot base."""
    # Configure settings
    CFG.seed = 123
    CFG.pybullet_robot = "fetch_mobile"
    
    # Initialize environment with GUI
    print("Initializing environment...")
    env = PyBulletBlocksEnv(use_gui=True)
    physics_client_id = env._physics_client_id
    robot = env._pybullet_robot
    
    # Reset the environment to have a valid state
    observation = env.reset("train", 0)
    
    # Get initial pose
    initial_pose = robot.get_base_pose(physics_client_id)
    print(f"Initial base pose: {initial_pose}")
    
    # Define target pose (make sure it's reachable and collision-free)
    target_pose = (0.8, 0.5, np.pi/4)  # Example target pose (x, y, theta)
    print(f"Target pose: {target_pose}")
    
    # Temporarily move table further away to make planning easier
    original_table_pos, original_table_orn = p.getBasePositionAndOrientation(
        env._table_id, physicsClientId=physics_client_id)
    
    p.resetBasePositionAndOrientation(
        env._table_id, 
        [2.0, 0.75, 0.0],  # Move table further away
        original_table_orn, 
        physicsClientId=physics_client_id
    )
    
    # Add collision bodies (table is the main obstacle)
    collision_bodies = [env._table_id]
    
    # Run BiRRT base planning
    print("Running BiRRT base planning...")
    base_path = run_base_motion_planning(
        robot=robot,
        target_pose=target_pose,
        collision_bodies=collision_bodies,
        seed=0,
        physics_client_id=physics_client_id,
        workspace_bounds=(0.0, 0.0, 2.0, 2.0)  # Use expanded workspace bounds
    )
    
    # Reset table to original position
    p.resetBasePositionAndOrientation(
        env._table_id, 
        original_table_pos, 
        original_table_orn, 
        physicsClientId=physics_client_id
    )
    
    if base_path is None:
        print("Failed to find a base path!")
        return
    
    print(f"Found base path with {len(base_path)} waypoints")
    print(f"First few waypoints: {base_path[:3] if len(base_path) > 3 else base_path}")
    
    # Execute base path
    print("Executing base path...")
    robot.execute_path(
        path=base_path,
        physics_client_id=physics_client_id,
        timestep=0.1,
        render_fn=lambda: time.sleep(0.01)
    )
    
    # Verify final pose
    final_pose = robot.get_base_pose(physics_client_id)
    print(f"Final base pose: {final_pose}")
    
    # Calculate distance to target
    pos_error = np.sqrt((final_pose[0] - target_pose[0])**2 + 
                       (final_pose[1] - target_pose[1])**2)
    angle_error = abs((final_pose[2] - target_pose[2] + np.pi) % (2*np.pi) - np.pi)
    
    print(f"Position error: {pos_error:.4f} m")
    print(f"Angle error: {angle_error:.4f} rad")
    
    # Test with obstacles in normal positions
    print("\nTesting planning with obstacles in normal positions...")
    
    # Reset robot to initial position
    robot.move_base_to(initial_pose, physics_client_id)
    
    # Different target to avoid previous path
    new_target_pose = (0.7, 0.6, np.pi/3)
    print(f"New target pose: {new_target_pose}")
    
    # Run BiRRT with obstacles in original positions
    new_base_path = run_base_motion_planning(
        robot=robot,
        target_pose=new_target_pose,
        collision_bodies=collision_bodies,
        seed=1,  # Different seed
        physics_client_id=physics_client_id,
        workspace_bounds=(0.0, 0.0, 2.0, 2.0)
    )
    
    if new_base_path is not None:
        print(f"Found path with obstacles in original positions ({len(new_base_path)} waypoints)")
        
        # Execute new path
        robot.execute_path(
            path=new_base_path,
            physics_client_id=physics_client_id,
            timestep=0.1,
            render_fn=lambda: time.sleep(0.01)
        )
        
        final_pose = robot.get_base_pose(physics_client_id)
        print(f"Final base pose: {final_pose}")
    else:
        print("Could not find path with obstacles in original positions")
    
    print("BiRRT base planning test completed successfully!")
    
    # Keep window open for inspection
    input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        test_birrt_base_planning()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
