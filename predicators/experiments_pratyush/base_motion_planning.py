"""
Demo for the mobile Fetch robot inside the PyBulletBlocks environment.
–moves the base with Bi‑RRT,
–then picks the left‑most block on the table.

"""
import time
import numpy as np

from predicators.settings import CFG
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.pybullet_helpers.robots.mobile_single_arm import \
    MobileSingleArmPyBulletRobot
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Action
from predicators.pybullet_helpers.motion_planning import run_base_motion_planning
from predicators import utils   # for BiRRT smoothing seed

# ---------------------------------------------------------------------------
# 1)  Environment & robot initialisation
# ---------------------------------------------------------------------------

# Tell Predicators which robot we want the env to build.
CFG.env = "pybullet_blocks"          # needed internally by settings
CFG.pybullet_robot = "fetch_mobile"  # maps to MobileFetchRobot
CFG.seed = 123

env = PyBulletBlocksEnv(use_gui=True)      # table/plane/blocks get auto‑loaded 
obs  = env.reset("train", 0)               # get initial fully‑observed State
robot : MobileSingleArmPyBulletRobot = env._pybullet_robot    # type cast OK

physics_id = env._physics_client_id
table_id   = env._table_id
all_blocks = env._block_ids               # list of pybullet IDs for collision

#Move the robot out of the table before starting
robot.move_base_to((0.65, 0.30, 0.0), physics_id)
# ---------------------------------------------------------------------------
# 2)  Base‑level motion planning
# ---------------------------------------------------------------------------

print("Current base pose:", robot.get_base_pose(physics_id))

target_pose = (1.10, 0.30, np.deg2rad(90))   # x,y,θ in map frame
path = run_base_motion_planning(             # :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
        robot,
        target_pose,
        collision_bodies=[table_id, *all_blocks],
        seed=0,
        physics_client_id=physics_id)

assert path is not None, "No collision‑free base path found!"
print(f"Base path has {len(path)} way‑points.")

# convert path → wheel velocities and execute
for v, omega, n_steps in robot.path_to_wheel_vels(path):       # :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
    for _ in range(n_steps):
        robot.drive_base_twist(v, omega, physics_client_id=physics_id)
        p.stepSimulation(physicsClientId=physics_id)
        time.sleep(1./240.)   # real‑time, drop to remove delay

print("Arrived at target base pose:", robot.get_base_pose(physics_id))

# ---------------------------------------------------------------------------
# 3)  Pick the nearest block
# ---------------------------------------------------------------------------

# a) locate block centres in world coordinates
block_positions = [p.getBasePositionAndOrientation(b, physicsClientId=physics_id)[0]
                   for b in all_blocks]
closest_id = all_blocks[np.argmin([pos[0] for pos in block_positions])]  # left‑most
block_x, block_y, _ = p.getBasePositionAndOrientation(
                        closest_id, physicsClientId=physics_id)[0]

# b) two simple cartesian way‑points – above then at grasp‑height
above   = Pose((block_x, block_y, robot._ee_home_pose.position[2]))  # keep home z
grasp   = Pose((block_x, block_y, env.table_height + 0.02))          # just above top

# helper to move EE pose via IK and send the joint action through env.step()
def go(pose: Pose, fingers: float):
    q = robot.inverse_kinematics(pose, validate=False)               # closed‑form IK
    q[robot.left_finger_joint_idx]  = fingers
    q[robot.right_finger_joint_idx] = fingers
    env.step(Action(np.asarray(q, dtype=np.float32)))

# open gripper, move, descend, close, lift
go(above,  robot.open_fingers)
go(grasp,  robot.open_fingers)
go(grasp,  robot.closed_fingers)
go(above,  robot.closed_fingers)

print("Pick executed – watch the GUI!")

# ---------------------------------------------------------------------------
# 4)  Shut everything down cleanly
# ---------------------------------------------------------------------------
time.sleep(2)
env.close()
