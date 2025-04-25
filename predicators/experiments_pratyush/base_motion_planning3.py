import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.pybullet_helpers.motion_planning import run_base_motion_planning

def test_mobile_fetch():
    """Test the mobile fetch robot with direct robot control."""
    
    # Set up environment configuration
    CFG.pybullet_robot = "fetch_mobile"
    CFG.seed = 123
    
    # Initialize the PyBullet environment
    env = PyBulletBlocksEnv(use_gui=True)
    physics_client_id = env._physics_client_id
    robot = env._pybullet_robot
    
    # Initialize the environment state
    observation = env.reset("train", 0)
    
    print(f"Robot created: {robot.get_name()} with ID: {robot.robot_id}")
    initial_pose = robot.get_base_pose(physics_client_id)
    print(f"Initial base pose: {initial_pose}")
    
    # Define target pose
    target_pose = (0.5, 0.5, np.pi/4)
    print(f"Target pose: {target_pose}")
    
    # Test 1: Direct Position Setting
    print("\n--- Test 1: Direct Position Setting ---")
    
    # Directly move the robot to the target pose
    robot.move_base_to(target_pose, physics_client_id)
    
    # Verify position
    new_pose = robot.get_base_pose(physics_client_id)
    print(f"New pose after position setting: {new_pose}")
    
    pos_error = np.sqrt((new_pose[0] - target_pose[0])**2 + 
                        (new_pose[1] - target_pose[1])**2)
    angle_error = abs((new_pose[2] - target_pose[2] + np.pi) % (2*np.pi) - np.pi)
    print(f"Position error: {pos_error:.4f} m")
    print(f"Angle error: {angle_error:.4f} rad")
    
    # Test 2: Motion Planning
    print("\n--- Test 2: Motion Planning ---")
    
    # Reset to initial position
    robot.move_base_to(initial_pose, physics_client_id)
    print(f"Reset to initial pose: {robot.get_base_pose(physics_client_id)}")
    
    # Temporarily move table out of the way to make planning easier
    table_id = env._table_id
    original_table_pos, original_table_orn = p.getBasePositionAndOrientation(
        table_id, physicsClientId=physics_client_id)
    
    # Move table further away
    p.resetBasePositionAndOrientation(
        table_id, 
        [2.0, 1.0, 0.0], 
        original_table_orn, 
        physicsClientId=physics_client_id
    )
    
    # Plan path with expanded workspace
    path = run_base_motion_planning(
        robot=robot,
        target_pose=target_pose,
        collision_bodies=[env._table_id],
        seed=0,
        physics_client_id=physics_client_id,
        workspace_bounds=(0.0, 0.0, 2.0, 2.0)
    )
    
    # Return table to original position
    p.resetBasePositionAndOrientation(
        table_id, 
        original_table_pos, 
        original_table_orn, 
        physicsClientId=physics_client_id
    )
    
    if path is None:
        print("ERROR: Could not find a path to the target!")
    else:
        print(f"Path found with {len(path)} waypoints")
        # Print first few waypoints
        print(f"First few waypoints: {path[:3]}")
        
        # Execute the path
        robot.execute_path(
            path=path,
            physics_client_id=physics_client_id,
            timestep=0.1,
            render_fn=lambda: time.sleep(0.01)
        )
        
        final_pose = robot.get_base_pose(physics_client_id)
        print(f"Final base pose: {final_pose}")
    
    # Test 3: Velocity Control
    print("\n--- Test 3: Velocity Control ---")
    
    # Reset position
    robot.move_base_to(initial_pose, physics_client_id)
    print(f"Reset to initial pose: {robot.get_base_pose(physics_client_id)}")
    
    # Directly apply velocity control with higher values
    v, omega = 1.0, 0.2
    
    print("Applying velocity control for 5 seconds...")
    
    # Ensure wheels can move freely
    for wheel_id in robot.wheel_ids:
        p.setJointMotorControl2(
            robot.robot_id, 
            wheel_id, 
            p.VELOCITY_CONTROL, 
            targetVelocity=0, 
            force=0, 
            physicsClientId=physics_client_id
        )
    
    start_time = time.time()
    while time.time() - start_time < 5.0:
        robot.drive_base_twist(v, omega, physics_client_id, max_force=50)
        p.stepSimulation(physicsClientId=physics_client_id)
        
        # Print debug info occasionally
        if int((time.time() - start_time) * 5) % 5 == 0:
            current_pose = robot.get_base_pose(physics_client_id)
            current_distance = np.sqrt((current_pose[0] - initial_pose[0])**2 + 
                                      (current_pose[1] - initial_pose[1])**2)
            print(f"Current position: {current_pose}, Distance: {current_distance:.4f} m")
        
        time.sleep(0.01)
    
    # Verify movement
    velocity_final_pose = robot.get_base_pose(physics_client_id)
    print(f"Final pose after velocity control: {velocity_final_pose}")
    
    distance_moved = np.sqrt((velocity_final_pose[0] - initial_pose[0])**2 + 
                            (velocity_final_pose[1] - initial_pose[1])**2)
    print(f"Distance moved: {distance_moved:.4f} m")
    
    # Keep window open
    input("Tests complete. Press Enter to exit...")

if __name__ == "__main__":
    try:
        test_mobile_fetch()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
