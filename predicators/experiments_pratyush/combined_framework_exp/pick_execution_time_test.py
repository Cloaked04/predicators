import random
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
CFG.option_model_use_gui = False

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



# ---------- config ----------
TARGET_TABLE_IDX = 0          # put piles on table 0; change if you want another
TOTAL_BLOCKS     = 17
NUM_SINGLETONS   = 1         # exactly 10 height-1 piles
MIN_MULTI_H      = 4          # min height for non-singleton piles
MAX_MULTI_H      = 4          # max height for non-singleton piles

PALETTE = [
    "red","blue","green","yellow","purple","orange","cyan","magenta","pink","brown",
    "navy","teal","maroon","olive","gold","silver","coral","indigo","lime","turquoise",
]

def generate_exact_pile_state_fixed(
    total_blocks: int,
    num_singletons: int,
    min_h: int,
    max_h: int
) -> Dict[str, List[str]]:
    """Return an insertion-ordered dict for exact_pile:
       - First `num_singletons` piles are singletons (height 1)
       - Remaining blocks distributed into piles with height in [min_h, max_h]
    """
    # assert 0 < num_singletons < total_blocks
    assert min_h >= 1 and min_h <= max_h

    remaining = total_blocks - num_singletons  # to place in multi-height piles

    # Compose `remaining` into heights in [min_h, max_h]
    multi_heights: List[int] = []
    while remaining > 0:
        h = random.randint(min_h, max_h)
        if h > remaining:
            # Ensure final piece is still >= min_h; if not, fix by borrowing or splitting
            if remaining >= min_h:
                h = remaining
            else:
                # borrow from previous pile if possible
                if multi_heights and multi_heights[-1] - (min_h - remaining) >= min_h:
                    multi_heights[-1] -= (min_h - remaining)
                    h = min_h
                else:
                    # fallback: just finish with min_h and accept going over by a tiny amount
                    h = min_h
        multi_heights.append(h)
        remaining -= h

    needed = num_singletons + sum(multi_heights)
    colors = (PALETTE * ((needed // len(PALETTE)) + 2))[:needed]
    random.shuffle(colors)
    c = 0

    piles: Dict[str, List[str]] = {}
    # First: singletons
    for i in range(1, num_singletons + 1):
        piles[f"pile{i}"] = [colors[c]]; c += 1
    # Then: multi-height piles
    for j, h in enumerate(multi_heights, start=num_singletons + 1):
        piles[f"pile{j}"] = colors[c:c+h]; c += h
    return piles


FILE = Path("cluttered_table2.jsonl")

def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)

start_count = count_jsonl(FILE)

option_model, option_model_env = create_option_model('oracle')
# ipdb.set_trace() 
initial_options=get_gt_options(env.get_name(), robot=option_model_env._pybullet_robot, env_obj=option_model_env)
initial_nsrts=get_gt_nsrts(env.get_name(), predicates_to_keep=env.predicates, options_to_keep=initial_options, 
                            robot=option_model_env._pybullet_robot, env_obj=option_model_env)



while True:
    current_count = count_jsonl(FILE)
    if current_count-start_count >= 100:
        break

    print(f"\n*******************************************")
    print(f"\n Looking for {current_count}-th pick action time.")
    print(f"\n*******************************************")
    # ---------- build exact state for table TARGET_TABLE_IDX ----------
    exact_state = generate_exact_pile_state_fixed(
        TOTAL_BLOCKS, NUM_SINGLETONS, MIN_MULTI_H, MAX_MULTI_H
    )

    table_configs = {
        0: {
            "exact_state": exact_state,
            "setup": "exact_pile",
            "params": None,
        },
        # keep other tables empty
        1: {"exact_state": [], "setup": "exact_scattered", "params": None},
        2: {"exact_state": [], "setup": "exact_scattered", "params": None},
    }

    # ---------- create state ----------
    initial_state = env.set_state(table_configs)

    # goal: pick block0_0_0
    preferred = f"block{TARGET_TABLE_IDX}_0_0"
    names = {o.name for o in initial_state.get_objects(env._block_type)}
    # if preferred in names:
    goal_block = Object(preferred, env._block_type)
    target_table = env._tables[2]
    assert goal_block is not None

    # else:
    #     # find the first singleton on TARGET_TABLE_IDX and use it
    #     table_obj = env._tables[TARGET_TABLE_IDX]
    #     # group by (x,y) to detect singletons
    #     eps = env._block_size * 0.25
    #     def xy_key(x, y): return (round(x/eps), round(y/eps))
    #     groups = {}
    #     for blk in initial_state.get_objects(env._block_type):
    #         if env._BlockAt(initial_state, [blk, table_obj]):
    #             x = float(initial_state.get(blk, "pose_x"))
    #             y = float(initial_state.get(blk, "pose_y"))
    #             groups.setdefault(xy_key(x, y), []).append(blk)
    #     singletons = [grp[0] for grp in groups.values() if len(grp) == 1]
    #     assert singletons, "No singleton piles found (unexpected)."
    #     goal_block = sorted(singletons, key=lambda b: b.name)[0]
    #     print(f"[warn] {preferred!r} not present; using {goal_block.name!r} instead.")

    goal = {GroundAtom(env._Holding, [goal_block])}


    # 4. Create Task object  
    task = Task(initial_state, goal)
    
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
                  
    # except Exception as e:  
    #     print(f"Planning failed: {e}") 
    except Exception:
        traceback.print_exc() 
        ipdb.post_mortem()
        continue
