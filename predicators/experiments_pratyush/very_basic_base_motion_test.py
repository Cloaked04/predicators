import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG

CFG.seed = 123
CFG.pybullet_robot = "fetch_mobile"

def test_mobile_fetch_base_only():
    """Test only the base movement capabilities"""
    # Create environment with GUI
    env = PyBulletBlocksEnv(use_gui=True)
    robot = env._pybullet_robot
    physics_client_id = env._physics_client_id
    
    # Get current base pose
    initial_pose = robot.get_base_pose(physics_client_id)
    print(f"Initial base pose: {initial_pose}")
    
    # Test direct base movement
    target_pose = (0.8, 0.5, np.pi/4)
    print(f"Moving base to: {target_pose}")
    robot.move_base_to(target_pose, physics_client_id)
    
    # Verify new position
    new_pose = robot.get_base_pose(physics_client_id)
    print(f"New base pose: {new_pose}")
    
    # Reset to initial position for velocity test
    print("Resetting to initial position...")
    robot.move_base_to(initial_pose, physics_client_id)
    print(f"Reset position: {robot.get_base_pose(physics_client_id)}")
    
    # Print wheel joint information
    print("\nWheel joint information:")
    for i, wheel_id in enumerate(robot.wheel_ids):
        info = p.getJointInfo(robot.robot_id, wheel_id, physicsClientId=physics_client_id)
        dynamics = p.getDynamicsInfo(robot.robot_id, wheel_id, physicsClientId=physics_client_id)
        print(f"Wheel {i}: Name={info[1].decode()}, Max Force={info[10]}, Friction={dynamics[1]}")
    
    # # Modify wheel dynamics to reduce friction
    # print("\nReducing wheel friction...")
    # for wheel_id in robot.wheel_ids:
    #     p.changeDynamics(
    #         robot.robot_id,
    #         wheel_id,
    #         lateralFriction=0.1,
    #         spinningFriction=0.01,
    #         rollingFriction=0.01,
    #         linearDamping=0.0,
    #         angularDamping=0.0,
    #         jointDamping=0.0,
    #         physicsClientId=physics_client_id
    #     )
    
    # # Completely unlock wheels with zero force before velocity control
    # print("Unlocking wheel joints...")
    # for wheel_id in robot.wheel_ids:
    #     p.setJointMotorControl2(
    #         robot.robot_id, 
    #         wheel_id, 
    #         p.VELOCITY_CONTROL, 
    #         targetVelocity=0, 
    #         force=0,  # Disable existing motor constraints
    #         physicsClientId=physics_client_id
    #     )
    
    # Run simulation for a moment to let physics settle
    for _ in range(50):
        p.stepSimulation(physicsClientId=physics_client_id)
        time.sleep(0.001)
    
    # Test velocity control with much higher force
    print("\nTesting velocity control with high force...")
    v, omega = 1, 0.2
    max_force = 20  # Significantly increased from default 20

    
    # Start timing
    start_time = time.time()
    duration = 5.0  # Run for 5 seconds

    print("Setting up proper dynamics...")
    # Ensure the base is at a good height
    initial_pos, initial_orn = p.getBasePositionAndOrientation(
        robot.robot_id, physicsClientId=physics_client_id)
    proper_height_pos = [initial_pos[0], initial_pos[1], 0.15]
    p.resetBasePositionAndOrientation(
        robot.robot_id, proper_height_pos, initial_orn, physicsClientId=physics_client_id)
        
    #while time.time() - start_time < duration:
    for _ in range(500):
        # Apply velocity control with high force
        robot.drive_base_twist(v, omega, physics_client_id, max_force=max_force)
        
        # Step simulation multiple times per control update
        for _ in range(10):  # More simulation steps per control command
            p.stepSimulation(physicsClientId=physics_client_id)
        time.sleep(0.01)
        
        # Print current position and wheel velocities occasionally
        if int((time.time() - start_time) * 5) % 5 == 0:
            wheel_states = [p.getJointState(robot.robot_id, wheel_id, physicsClientId=physics_client_id) 
                           for wheel_id in robot.wheel_ids]
            wheel_velocities = [state[1] for state in wheel_states]
            print(f"Wheel velocities: {wheel_velocities}")
            
            current_pose = robot.get_base_pose(physics_client_id)
            distance_moved = np.sqrt((current_pose[0] - initial_pose[0])**2 + 
                                   (current_pose[1] - initial_pose[1])**2)
            print(f"Current pose: {current_pose}, Distance: {distance_moved:.4f} m")
        
        time.sleep(0.01)
    
    final_pose = robot.get_base_pose(physics_client_id)
    print(f"Final base pose: {final_pose}")
    
    distance_moved = np.sqrt((final_pose[0] - initial_pose[0])**2 + 
                           (final_pose[1] - initial_pose[1])**2)
    print(f"Distance moved: {distance_moved:.4f} m")
    
    # Keep window open for inspection
    input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        test_mobile_fetch_base_only()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
