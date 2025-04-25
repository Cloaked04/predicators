#!/usr/bin/env python3
# ---------------------------------------------------------------
# mobile_base_path_test.py
#
# Simple sanity‑check for run_base_motion_planning().
# ---------------------------------------------------------------

import time
import numpy as np
import pybullet as p

from predicators.settings import CFG
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.structs import Action
from predicators.pybullet_helpers.motion_planning import run_base_motion_planning

# ----------------------- CONFIGURATION -------------------------
CFG.pybullet_control_mode = "position"      # teleport chassis / position‑control arm
CFG.pybullet_robot        = "fetch_mobile"  # our MobileSingleArm subclass
CFG.seed                  = 123

###########################

CFG.pybullet_birrt_num_attempts     = 30      # default 5
CFG.pybullet_birrt_num_iters        = 4000    # default 500
CFG.pybullet_birrt_extend_num_interp = 20     # default  5

# ------------------- INITIALISE ENVIRONMENT --------------------
env = PyBulletBlocksEnv(use_gui=True)       # loads plane, table & blocks for us
physics_id = env._physics_client_id
state      = env.reset("train", 0)          # get the initial task
robot      = env._pybullet_robot            # MobileFetchRobot instance

# ------------------ PUT ROBOT IN A CLEAR POSE ------------------
# Move the chassis well clear of the table so the start pose is collision‑free.
#start_pose = (0.4, 0.30, 0.0)              # (x, y, yaw)
#robot.move_base_to(start_pose, physics_id)  # direct teleport

print("Start pose:", robot.get_base_pose(physics_id))

# -------------------- DEFINE GOAL POSE -------------------------
goal_pose  = (1.10, 0.95, np.pi/2)       
print("Goal pose :", goal_pose)

# --------------- BUILD COLLISION‑BODY LIST  --------------------
num_bodies      = p.getNumBodies(physicsClientId=physics_id)
plane_id        = 0                         # first body loaded inside PyBulletEnv
collision_bodies = [i for i in range(num_bodies)
                    if i not in {robot.robot_id, plane_id}]

# collision_bodies = [i for i in range(num_bodies)
#                     if i not in {robot.robot_id, plane_id, table_id}]


# -------------------- PLAN A BASE PATH -------------------------
path = run_base_motion_planning(
    robot             = robot,
    target_pose       = goal_pose,
    collision_bodies  = collision_bodies,
    seed              = CFG.seed,
    physics_client_id = physics_id,
)

assert path is not None, "Planner failed to find a base path!"
print(f"Found path with {len(path)} way‑points")

# ------------------- EXECUTE THE WAY‑POINTS --------------------
# We teleport the chassis to each waypoint via Action.set_base_motion(..., mode='position').
action_dim = robot.action_space.shape[0]    # arm joints (+2 for v,ω)

for (x, y, theta) in path:
    act = Action(np.zeros(action_dim, dtype=np.float32))
    act.set_base_motion(x, y, theta, mode="position")
    env.step(act)
    env.render()
    time.sleep(0.05)

print("Done – base reached the goal pose!")

# ---------------- KEEP GUI ALIVE (optional) --------------------
while True:
    time.sleep(0.1)
