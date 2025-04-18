import time
import numpy as np
import pybullet as p

from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG
from predicators.pybullet_helpers.geometry import Pose

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def plan_cartesian_path(start: Pose, goal: Pose, steps: int):
    """Linearly interpolate between two end‑effector Poses."""
    path = []
    for i in range(steps + 1):
        α = i / steps
        pos = (1 - α) * np.array(start.position) + α * np.array(goal.position)
        # simple linear interp of quaternions (not perfect but OK for small rotations)
        orn = (1 - α) * np.array(start.orientation) + α * np.array(goal.orientation)
        path.append(Pose(tuple(pos), tuple(orn)))
    return path

def plan_joint_path(start: np.ndarray, goal: np.ndarray, steps: int):
    """Linearly interpolate joint vectors."""
    return [(1 - α) * start + α * goal
            for α in np.linspace(0, 1, steps + 1)]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Config ---
    CFG.blocks_block_size   = 0.05
    CFG.seed                = 123
    CFG.pybullet_robot      = "fetch"    # or "panda"
    STEPS                  = 100
    DT                     = 0.02        # seconds per sim step

    # --- Initialize env & PyBullet ---
    env = PyBulletBlocksEnv(use_gui=True)
    env.reset("train", 0)
    client = env._physics_client_id
    robot = env._pybullet_robot

    # get a clean start‑joint vector and its corresponding EE pose
    q_start = np.array(robot.get_joints(), dtype=np.float32)
    ee_start = robot.forward_kinematics(q_start)

    # define a goal end‑effector pose: 0.1 m above current in Z
    goal_pos = (ee_start.position[0],
                ee_start.position[1],
                ee_start.position[2] + 0.1)
    ee_goal = Pose(goal_pos, ee_start.orientation)

    print("Starting Cartesian path…")
    cart_path = plan_cartesian_path(ee_start, ee_goal, steps=STEPS)
    for pose in cart_path:
        # IK → joint target (don’t re‑commit into sim yet)
        q_target = robot.inverse_kinematics(pose,
                                            validate=False,
                                            set_joints=False)
        # command motors  
        robot.set_motors(q_target)
        # step sim & render
        p.stepSimulation(physicsClientId=client)
        env.render()
        time.sleep(DT)

    print("Cartesian motion done. Pausing…")
    time.sleep(1.0)

    # --- Now do a joint‑space lift
    # lift by adding +0.2 on joint 2 (should move roughly same EE)
    q_lift = q_start.copy()
    q_lift[2] += 0.2

    print("Starting joint‑space path…")
    joint_path = plan_joint_path(q_start, q_lift, steps=STEPS)
    for q in joint_path:
        robot.set_motors(q.tolist())
        p.stepSimulation(physicsClientId=client)
        env.render()
        time.sleep(DT)

    print("Joint‑space motion done. Press Ctrl+C to exit.")
    while True:
        time.sleep(0.1)
