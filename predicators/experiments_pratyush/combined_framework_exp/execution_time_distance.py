import random
import sys
from typing import Dict, List
from predicators.structs import Object, GroundAtom

import ipdb, traceback
import numpy as np
import itertools
import logging
import random
import json
from pathlib import Path

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
    level=logging.CRITICAL,                    
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
CFG.option_model_use_gui = False


use_num_goal_items = {1:[0], 2:[0]}

# Initialize your multi-table environment  
env = PyBulletMultiTableBlocksEnv(use_gui=False, num_tables=3, use_num_goal_items=use_num_goal_items)
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

plan_count = 0

while plan_count<100:

    # env = PyBulletMultiTableBlocksEnv(use_gui=False, num_tables=3, use_num_goal_items=use_num_goal_items)
    # CFG.env = env.get_name()

    # symbolic_robot = env._robot
    # pybullet_robot = env._pybullet_robot
    # physics_client_id = env._physics_client_id
    # assert physics_client_id is not None


    table_config1 = {  
        2: {  # Table 2: Single block
            'exact_state': ['black'],  
            'setup': 'exact_scattered',  
            'params': None
        },  
        1: {  # Table 1: Random pile
            'exact_state': {},  
            'setup': 'pile',  
            'params': [3, 3]  
        },  
        0: {  # Table 0: Empty
            'exact_state': [],  
            'setup': 'exact_scattered',  
            'params': None  
        }  
    }

    table_config2 = {  
        1: {  # Table 1: Single block
            'exact_state': ['black'],  
            'setup': 'exact_scattered',  
            'params': None
        },  
        2: {  # Table 2: Random pile
            'exact_state': {},  
            'setup': 'pile',  
            'params': [3, 3]  
        },  
        0: {  # Table 0: Empt
            'exact_state': [],  
            'setup': 'exact_scattered',  
            'params': None  
        }  
    }


    table_config = random.choice([table_config1, table_config2])


    try:
        # Create the initial state  
        initial_state = env.set_state(table_config)

        goal = {
        			GroundAtom(env._GOALOBJHELD, []),
        }

        # preferred = f"block2_0_0"
        # goal_block = Object(preferred, env._block_type)
        # target_table = env._tables[2]
        # assert goal_block is not None

        # goal = {GroundAtom(env._Holding, [goal_block])}

        # print(initial_state)
        # input()
        # Create Task object  
        task = Task(initial_state, goal)
        option_model, option_model_env = create_option_model('oracle', use_num_goal_items=use_num_goal_items)
        initial_options=get_gt_options(env.get_name(), robot=option_model_env._pybullet_robot, env_obj=option_model_env)
        initial_nsrts=get_gt_nsrts(env.get_name(), predicates_to_keep=env.predicates, options_to_keep=initial_options, 
                                    robot=option_model_env._pybullet_robot, env_obj=option_model_env)

        # init_atoms = utils.abstract(initial_state, env.predicates)
        # robot_at_atoms = [atom for atom in init_atoms if atom.predicate.name == 'RobotAt']
        # robot_not_at_atoms = [atom for atom in init_atoms if atom.predicate.name == 'RobotNotAt']
        # at_home_atoms = [atom for atom in init_atoms if atom.predicate.name == 'AtHome']

        # print(f"\nAtHome atoms: {at_home_atoms}")
        # print(f"\nRobotAt atoms: {robot_at_atoms}")
        # print(f"\nRobotNotAt atoms: {robot_not_at_atoms}")

        # table_objects = initial_state.get_objects(env._table_type)

        # ipdb.set_trace()

        # for table in table_objects:
        #     print(f"Robot at {table.name}: {env._AtHolds(initial_state, [symbolic_robot, table])}")
        #     print(f"Robot pos: ({initial_state.get(symbolic_robot, 'pose_x')}, {initial_state.get(symbolic_robot, 'pose_y')})")
        #     print(f"Table pos: ({initial_state.get(table, 'pose_x')}, {initial_state.get(table, 'pose_y')})")
        # ipdb.set_trace()

        # Create Oracle approach with environment's components  
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

        # Solve the TAMP problem using bilevel planning  
        print("Starting TAMP planning with bilevel approach...")  
        # ipdb.set_trace()  
        policy = approach.solve(task, timeout=20000, continuous_env=env)
        print("Planning completed successfully!")
        plan_count += 1

    except AssertionError as e:
        continue
    except Exception as e:
        traceback.print_exc() 
        # ipdb.post_mortem()
        continue
     