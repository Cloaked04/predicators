"""
pybullet_blocks_demo.py

Demonstrates how to initialize and step through the PyBullet Blocks
environment from predicators.envs.pybullet_blocks.
"""

import numpy as np
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.structs import Action

def main():
    # --- 1) Initialize PyBullet and load the Fetch+Table+Blocks scene ---
    # use_gui=False runs in DIRECT/headless mode
    env = PyBulletBlocksEnv(use_gui=False)

    # --- 2) Reset to get the initial symbolic observation/state ---
    # 'train' or 'test', and integer task index
    obs = env.reset('train', 0)  
    print("\n=== Initial State ===")
    print(obs.pretty_str())

    # --- 3) Pick up the first block in the scene ---
    #   * Identify block objects from the State
    blocks = obs.get_objects(env._block_type)
    block0 = blocks[0]
    #   * Read its x,y,z from the symbolic State
    x = float(obs.get(block0, 'pose_x'))
    y = float(obs.get(block0, 'pose_y'))
    z = float(obs.get(block0, 'pose_z'))
    print(f"\nPicking up block '{block0.name}' at ({x:.3f}, {y:.3f}, {z:.3f})")

    #   * Create an Action: [x, y, z, fingers]; fingers<0.5 means "pick"
    pick_arr = np.array([x, y, z, 0.0], dtype=np.float32)
    pick_action = Action(pick_arr)

    #   * Simulate: returns a new State with that pick applied
    post_pick = env.simulate(obs, pick_action)
    print("\n=== State After Pick ===")
    print(post_pick.pretty_str())

    # --- 4) Place the block back down on the table ---
    #   * Choose a z just above the table so that z < table_height + block_size
    put_z = env.table_height + env._block_size * 0.5
    print(f"\nPutting block '{block0.name}' back at z = {put_z:.3f}")

    #   * fingers≥0.5 triggers put-on-table
    put_arr = np.array([x, y, put_z, 1.0], dtype=np.float32)
    put_action = Action(put_arr)

    post_put = env.simulate(post_pick, put_action)
    print("\n=== State After PutOnTable ===")
    print(post_put.pretty_str())

    # --- 5) Inspect the high-level options available in this env ---
    print("\n=== Parameterized Options Available ===")
    for opt in env.options:
        print(f" • {opt.name}   (parameter space shape: {opt.params_space.shape})")

if __name__ == '__main__':
    main()
