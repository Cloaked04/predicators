import numpy as np
import pybullet as p
import time
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.settings import CFG

# Configuration
CFG.blocks_block_size = 0.05
CFG.seed = 123
CFG.pybullet_robot = "fetch"

# Initialize
env   = PyBulletBlocksEnv(use_gui=True)
_     = env.reset("train", 0)
robot = env._pybullet_robot
client = env._physics_client_id
table_id = env._table_id

robot_id = robot.robot_id
# keep the robot's base at this Z
base_z = p.getBasePositionAndOrientation(robot_id, physicsClientId=client)[0][2]

print("Robot ID:", robot_id, "Table ID:", table_id)

def check_base_collision(robot_id, test_pos, test_orn, client, obstacle_id):
    """
    Teleport the robot base to (test_pos, test_orn),
    step the sim once, check for any closest points ≤ 0,
    then immediately rollback.
    """
    # 1) remember old pose
    old_pos, old_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client)
    # 2) teleport to test
    p.resetBasePositionAndOrientation(robot_id, test_pos, test_orn, physicsClientId=client)
    # 3) step sim so Bullet computes distances
    p.stepSimulation(physicsClientId=client)
    # 4) check if any points are penetrating or touching (distance ≤ 0)
    pts = p.getClosestPoints(bodyA=robot_id,
                             bodyB=obstacle_id,
                             distance=0.0,
                             physicsClientId=client)
    collision = len(pts) > 0
    if collision:
        print(f"collision at {test_pos}")
    # 5) rollback to old pose
    p.resetBasePositionAndOrientation(robot_id, old_pos, old_orn, physicsClientId=client)
    # 6) step again to clear any contact caches
    p.stepSimulation(physicsClientId=client)
    return collision

def move_base_safely(robot_id, waypoints, client, obstacle_id):
    """
    For each (pos,orn) in waypoints, test with check_base_collision,
    then commit only if it's collision-free.
    """
    for idx, (pos, orn) in enumerate(waypoints, 1):
        print(f"waypoint {idx}: {pos}")
        if check_base_collision(robot_id, pos, orn, client, obstacle_id):
            print("    aborting path—hit obstacle.")
            return False
        # commit the move
        p.resetBasePositionAndOrientation(robot_id, pos, orn, physicsClientId=client)
        # a few sim steps so you actually see it
        for _ in range(5):
            p.stepSimulation(physicsClientId=client)
        env.render()
        time.sleep(0.1)
    return True

# build a simple straight‑line path through your five targets
targets = [
    (1.4, 0.85),
    (1.5, 0.85),
    (1.5, 0.65),
    (1.3, 0.65),
    (1.3, 0.75),
]
steps_per_segment = 10
initial_xy = p.getBasePositionAndOrientation(robot_id, physicsClientId=client)[0][:2]
path = []
prev = initial_xy
for tx, ty in targets:
    for i in range(1, steps_per_segment+1):
        alpha = i / steps_per_segment
        x = prev[0]*(1-alpha) + tx*alpha
        y = prev[1]*(1-alpha) + ty*alpha
        path.append(((x, y, base_z), p.getBasePositionAndOrientation(robot_id, physicsClientId=client)[1]))
    prev = (tx, ty)

print("\nStarting safe base moves…")
ok = move_base_safely(robot_id, path, client, table_id)
print("Done. Success?", ok)

# keep open
while True:
    time.sleep(0.1)
