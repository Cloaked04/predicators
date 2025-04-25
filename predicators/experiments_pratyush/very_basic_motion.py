#!/usr/bin/env python3
"""
Test script for mobile Fetch robot base motion in PyBullet with BiRRT path planning.
This script demonstrates path planning with collision avoidance.
"""

import time
import numpy as np
import pybullet as p
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.pybullet_helpers.motion_planning import run_base_motion_planning
from predicators.utils import Rectangle, Circle

# Configure the robot and path planning
CFG.seed = 123
CFG.pybullet_robot = "fetch_mobile"
CFG.pybullet_control_mode = "position"

# BiRRT planning parameters
CFG.pybullet_birrt_num_attempts = 50  # Increase attempts for better success
CFG.pybullet_birrt_num_iters = 8000   # More iterations for complex environments
CFG.pybullet_birrt_smooth_amt = 15    # Higher smoothing for better paths

# For IK solver
CFG.pybullet_max_ik_iters = 200
CFG.pybullet_ik_tol = 0.1

def create_test_obstacles(physics_client_id):
    """Create test obstacles in the environment."""
    obstacles = []
    
    # Create a box obstacle
    box_half_extents = [0.1, 0.1, 0.2]
    visual_id = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=box_half_extents,
        rgbaColor=[1, 0, 0, 0.7],
        physicsClientId=physics_client_id
    )
    collision_id = p.createCollisionShape(
        p.GEOM_BOX, 
        halfExtents=box_half_extents,
        physicsClientId=physics_client_id
    )
    
    # Place box in middle of path
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[0.8, 0.8, 0.2],
        physicsClientId=physics_client_id
    )
    obstacles.append(box_id)
    
    # Create a cylindrical obstacle
    cylinder_radius = 0.1
    cylinder_height = 0.4
    visual_id = p.createVisualShape(
        p.GEOM_CYLINDER, 
        radius=cylinder_radius,
        length=cylinder_height,
        rgbaColor=[0, 1, 0, 0.7],
        physicsClientId=physics_client_id
    )
    collision_id = p.createCollisionShape(
        p.GEOM_CYLINDER, 
        radius=cylinder_radius,
        height=cylinder_height,
        physicsClientId=physics_client_id
    )
    
    # Place cylinder in another path
    cylinder_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[1.2, 0.5, 0.2],
        physicsClientId=physics_client_id
    )
    obstacles.append(cylinder_id)
    
    return obstacles

def test_birrt_planning():
    """Main test function for BiRRT motion planning."""
    # Create environment with GUI
    print("Initializing PyBullet environment...")
    env = PyBulletBlocksEnv(use_gui=True)
    robot = env._pybullet_robot
    physics_id = env._physics_client_id
    
    # Enable debug visualization
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    
    try:
        # Create obstacles
        print("Creating test obstacles...")
        obstacles = create_test_obstacles(physics_id)
        
        # Get current base pose
        initial_pose = robot.get_base_pose(physics_id)
        print(f"Initial base pose: {initial_pose}")
        
        # Set a proper starting position with safe height
        start_pose = (0.5, 0.5, 0.0)
        print(f"Moving to starting position: {start_pose}")
        robot.move_base_to(start_pose, physics_id)
        time.sleep(1)  # Pause to observe
        
        # Define goal position - on other side of obstacles
        goal_pose = (1.5, 1.2, np.pi/4)
        print(f"Goal position: {goal_pose}")
        
        # Create collision body list for BiRRT
        collision_bodies = obstacles.copy()
        
        # Add table to collision bodies if it exists
        for i in range(p.getNumBodies(physicsClientId=physics_id)):
            body_name = p.getBodyInfo(i, physicsClientId=physics_id)[1].decode()
            if "table" in body_name.lower():
                collision_bodies.append(i)
                print(f"Added table (body ID {i}) to collision bodies")
        
        print(f"Planning with {len(collision_bodies)} collision bodies: {collision_bodies}")
        
        # Visualize start and goal poses
        start_marker = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[0, 1, 0, 0.7],
            physicsClientId=physics_id
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_marker,
            basePosition=[start_pose[0], start_pose[1], 0.05],
            physicsClientId=physics_id
        )
        
        goal_marker = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[1, 0, 0, 0.7],
            physicsClientId=physics_id
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_marker,
            basePosition=[goal_pose[0], goal_pose[1], 0.05],
            physicsClientId=physics_id
        )
        
        # Run path planning
        print("Running BiRRT path planning...")
        
        # First, define a footprint for the robot
        footprint_fn = lambda pose: robot.footprint_circle_at(pose)
        
        # Define collision checking function
        def collision_fn(pose):
            # Move robot to the position without stepping simulation
            current_pos, current_orn = p.getBasePositionAndOrientation(
                robot.robot_id, physicsClientId=physics_id)
            target_x, target_y, target_theta = pose
            target_pos = [target_x, target_y, 0.2]  # Keep safe height
            target_orn = p.getQuaternionFromEuler([0, 0, target_theta])
            
            # Temporarily move robot to check collision
            p.resetBasePositionAndOrientation(
                robot.robot_id, target_pos, target_orn, physicsClientId=physics_id)
            
            # Check for collisions
            for body_id in collision_bodies:
                contacts = p.getContactPoints(
                    bodyA=robot.robot_id, bodyB=body_id, physicsClientId=physics_id)
                if contacts:
                    # Reset back to original position and return collision
                    p.resetBasePositionAndOrientation(
                        robot.robot_id, current_pos, current_orn, physicsClientId=physics_id)
                    return True  # In collision
            
            # Reset back to original position and return no collision
            p.resetBasePositionAndOrientation(
                robot.robot_id, current_pos, current_orn, physicsClientId=physics_id)
            return False  # No collision
            
        # Run the BiRRT planner
        path = run_base_motion_planning(
            start_pose, 
            goal_pose,
            footprint_fn,
            collision_fn,
            obstacles=collision_bodies,
            physics_client_id=physics_id
        )
        
        if path is None:
            print("Failed to find a path! Trying fallback planning...")
            # Implement a simple direct path with additional waypoints
            path = []
            # Try to go around obstacles with manually set waypoints
            path.append(start_pose)  # Start
            path.append((0.7, 0.3, np.pi/4))  # Waypoint 1
            path.append((1.0, 0.3, 0))        # Waypoint 2
            path.append((1.3, 0.7, np.pi/4))  # Waypoint 3
            path.append(goal_pose)  # Goal
            
            # Verify the path is collision-free
            for pose in path:
                if collision_fn(pose):
                    print(f"Warning: Waypoint {pose} is in collision!")
        else:
            print(f"Path found with {len(path)} waypoints!")
            
        # Visualize the path
        for i, waypoint in enumerate(path):
            marker = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0, 0, 1, 0.5],
                physicsClientId=physics_id
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=marker,
                basePosition=[waypoint[0], waypoint[1], 0.05],
                physicsClientId=physics_id
            )
            print(f"Waypoint {i}: {waypoint}")
        
        # Execute the path
        print("Executing planned path...")
        robot.execute_path(path, physics_id, timestep=0.05)
        
        print("Path execution complete!")
        time.sleep(1)
        
        # Return to start for another demo
        print("Returning to start...")
        robot.move_base_to(start_pose, physics_id)
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    # Keep window open for inspection
    print("Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    test_birrt_planning()