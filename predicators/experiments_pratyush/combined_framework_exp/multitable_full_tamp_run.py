import ipdb, traceback
import numpy as np
import itertools
import logging
import random

from predicators.settings import CFG, GlobalSettings
from predicators.envs.pybullet_multitable_blocks import PyBulletMultiTableBlocksEnv
from predicators.approaches.oracle_approach import OracleApproach  
from predicators.ground_truth_models import get_gt_options, get_gt_nsrts  
from predicators.structs import GroundAtom, EnvironmentTask, Task, Object  
from predicators.utils import option_plan_to_policy  
from predicators.option_model import create_option_model
from predicators import utils

logging.basicConfig(
    filename="tamp_run.log",
    filemode="w",
    level=logging.DEBUG,                    
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class Seeds:
    def __init__(self, start=random.randint(0, 10000)):
        self._counter = itertools.count(start)

    @property
    def seed(self) -> int:
        # Each access returns the next unique int
        return next(self._counter)

s = Seeds()

CFG.blocks_block_size = 0.05
CFG.pybullet_birrt_num_iters = 50
CFG.pybullet_birrt_num_attempts = 10
CFG.pybullet_birrt_smooth_amt = 100
# CFG.pybullet_sim_steps_per_action = 100
CFG.seed = s.seed
CFG.pybullet_robot = "fetch_mobile"
CFG.option_model_terminate_on_repeat = False
# CFG.option_model_use_gui = True


# 1. Initialize your multi-table environment  
env = PyBulletMultiTableBlocksEnv(use_gui=False, num_tables=3)
CFG.env = env.get_name()

symbolic_robot = env._robot
pybullet_robot = env._pybullet_robot
physics_client_id = env._physics_client_id
assert physics_client_id is not None

args = {
    "env": env.get_name(),       
    "approach": "oracle", 
}
updates = GlobalSettings.get_arg_specific_settings(args)
for k, v in updates.items():
    setattr(CFG, k, v)  
  
# 2. Create initial state using your set_state function  
table_configs = {  
    0: {  # Table 0: Random piles  
        'exact_state': {},  
        'setup': 'pile',  
        'params': [2, 3]  # 2 piles, 3 blocks per pile  
    },  
    1: {  # Table 1: Exact pile configuration  
        'exact_state': {  
            'pile1': ['red', 'blue', 'green'],  
            'pile2': ['yellow', 'purple']  
        },  
        'setup': 'exact_pile',  
        'params': None  
    },  
    2: {  # Table 2: Empty initially  
        'exact_state': [],  
        'setup': 'exact_scattered',  
        'params': None  
    }  
}  
  
# Create the initial state  
initial_state = env.set_state(table_configs)

# table_pose = env._table_poses[0]  # (1.0, -0.5, 0.0)  

# def points_on_circle(x, y, r=0.83, num_points=1):
#     """Return points on a circle centered at (x, y) with radius r."""
#     angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
#     xs = x + r * np.cos(angles)
#     ys = y + r * np.sin(angles)
#     return [xs, ys]

# # Position robot near the table  
# robot_target_x, robot_target_y  = points_on_circle(table_pose[0], table_pose[1])

# # Move the robot base  
# pybullet_robot.move_base_to((robot_target_x, robot_target_y, 0.0),   
#                                 physics_client_id=physics_client_id)
  
# 3. Define goal using your multi-table predicates  
# Example goal: Move specific blocks to table 2, robot should end at table 1:
# Block from table 0
block_to_move1 = Object("block0_0_0", env._block_type)
# Block from table 1      
block_to_move2 = Object("block1_1_0", env._block_type)
target_table = env._tables[2]  # Table 2  
robot_target_table = env._tables[1]  # Table 1 

  
# goal = {  
#     GroundAtom(env._OnTable, [block_to_move1, target_table]),  
#     GroundAtom(env._OnTable, [block_to_move2, target_table]),  
#     GroundAtom(env._At, [symbolic_robot, robot_target_table])  
# }

goal = {  
    GroundAtom(env._OnTable, [block_to_move1, target_table]),
}

  
# 4. Create Task object  
task = Task(initial_state, goal)
option_model, option_model_env = create_option_model('oracle')
# ipdb.set_trace() 
initial_options=get_gt_options(env.get_name(), robot=option_model_env._pybullet_robot, env_obj=option_model_env)
initial_nsrts=get_gt_nsrts(env.get_name(), predicates_to_keep=env.predicates, options_to_keep=initial_options, 
                            robot=option_model_env._pybullet_robot, env_obj=option_model_env)
# for item in initial_nsrts:
#     print(f"\n{item}")
# input()

# Debug robot position and table workspaces******************************************** 
# robot_x, robot_y, _ = env._pybullet_robot.get_base_pose(physics_client_id=env._physics_client_id)  
# print(f"Robot position: ({robot_x}, {robot_y})")  
  
# for i, table in enumerate(env._tables):  
#     tx = task.init.get(table, "pose_x")   
#     ty = task.init.get(table, "pose_y")  
#     print(f"Table {i} position: ({tx}, {ty})")  
      
#     # Check workspace bounds  
#     x_workspace = (tx-0.125, tx+0.125)  
#     y_workspace = (ty-0.2, ty+0.2)  
#     print(f"  Workspace: x({x_workspace[0]:.3f}, {x_workspace[1]:.3f}), y({y_workspace[0]:.3f}, {y_workspace[1]:.3f})")  
      
#     # Check if robot is within bounds  
#     robot_at_atom = GroundAtom(env._At, [env._robot, table])  
#     holds = robot_at_atom.holds(task.init)  
#     print(f"  RobotAt(robby, table{i}): {holds}")

# input()
#****************************************************************************************
# 5. Create Oracle approach with your environment's components  
approach = OracleApproach(  
    initial_predicates=env.predicates, 
    initial_options=initial_options,
    types=env.types,
    action_space=env.action_space,  
    train_tasks=[],
    task_planning_heuristic="lmcut",
    nsrts=initial_nsrts,
    option_model=option_model
)  
  
# 6. Solve the TAMP problem using bilevel planning  
print("Starting TAMP planning with bilevel approach...")  
try:
    # ipdb.set_trace()  
    policy = approach.solve(task, timeout=20000, continuous_env=env)
    # policy = approach.solve(task, timeout=20000)  
    print("Planning completed successfully!")  
      
    # 7. Execute the policy  
    print("Executing policy...")  
    state = task.init  
    step_count = 0  
    max_steps = 1000    
              
# except Exception as e:  
#     print(f"Planning failed: {e}") 
except Exception:
    traceback.print_exc() 
    ipdb.post_mortem()
    raise
    
print("\nPlanning metrics:")  
for metric_name, value in approach.metrics.items():  
    print(f"  {metric_name}: {value}")

try:
    # action = policy(state)  
    # state = env.step(action)  
    # step_count += 1  
      
    # # Optional: Print progress every 10 steps  
    # if step_count % 10 == 0:  
    #     print(f"Step {step_count}: Executing action...")
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=max_steps)

    print(f"\nSuccess: {task.goal_holds(traj.states[-1])}")
              
except Exception as e:  
    traceback.print_exc() 
    ipdb.post_mortem()
    raise
  
# 8. Check if goal was achieved  
if task.goal_holds(state):  
    print(f"SUCCESS! Goal achieved in {step_count} steps.")  
    print("Final goal state:")  
    for atom in goal:  
        print(f"  {atom}: {atom.holds(state)}")  
else:  
    print(f"FAILED: Goal not achieved after {step_count} steps.")  
    print("Goal status:")  
    for atom in goal:  
        print(f"  {atom}: {atom.holds(state)}")