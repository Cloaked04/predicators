import os
import sys
import time
import ipdb
import copy
import logging
import numpy as np
from operator import attrgetter
from typing import List, Dict, Set, Any, Sequence, Optional

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose, Pose3D
from predicators.structs import GroundAtom, _GroundNSRT, State
from predicators.envs.blocks import BlocksEnv


OPTIONS = ["pick", "place", "move", "stack", "put"]
MOVE_LOCATIONS = ["home", "table0", "table1", "table2", "table3"]
PICK_PLACE_LOCATIONS = ["table0", "table1", "table2"]
MOVE_CONSTANT = 5
PICK_CONSTANT = 0.5
PLACE_CONSTANT = 0.5
# DIST_LIST = []

def compute_area(bounds: dict) -> float:
    """
    Compute the area of a rectangle given its bounding box.

    Args:
        bounds (dict): Must contain keys 'x_lb', 'x_ub', 'y_lb', 'y_ub'.

    Returns:
        float: The computed area.
    """
    # Extract bounds
    x_lb = bounds['x_lb']
    x_ub = bounds['x_ub']
    y_lb = bounds['y_lb']
    y_ub = bounds['y_ub']

    # Compute width and height
    width = abs(x_ub - x_lb)
    height = abs(y_ub - y_lb)

    # Compute area
    area = width * height
    return area

def ordered_matches(text: str, patterns: list[str]) -> list[str]:
    """
    Return a list of substrings from 'patterns' that appear in 'text',
    ordered by the position they first appear in 'text'.
    """
    positions = []
    for pat in patterns:
        idx = text.find(pat)
         # if found
        if idx != -1:
            positions.append((idx, pat))

    # sort by index (first occurrence)
    positions.sort(key=lambda x: x[0])

    # extract the patterns in sorted order
    return [pat for _, pat in positions]

def get_unique_pile_height(tower_chain_dict: dict, block_below: str) -> int:
    """
    Takes in a dist that has key-value pairs of the form:
        block_below: '(On block_above block_below)',
    and name of a block to check if a block exists above it.

    Output: height of pile.
    """
    tower_height = 0
    
    try:
        block_above_pred = tower_chain_dict[block_below]
        block_above = block_above_pred.split()[1]
        return tower_height + get_unique_tower_height(tower_chain_dict, block_below) + 1
    except KeyError:
        return tower_height+1



def get_geometric_cost_while_search(ground_nsrts: List[_GroundNSRT], current_op: str, 
                                    init_atoms: Set[GroundAtom], state_atoms: Set[GroundAtom], 
                                    continuous_env: BlocksEnv) -> float:
    """Geometric cost fn. to be used during A* search in the loop. 
    This is disjoint of the heuristic fn. defined below.
    Receives the name of current NSRT being applied during search, current state atoms,
    and initial countinuous state.
    Returns: the value of executing current option in
    current state.
    """

    static_atoms = utils.get_static_atoms(ground_nsrts, init_atoms)
    # Filter out the static atoms from the input state:
    non_static_state_atoms = set(state_atoms) - static_atoms

    # Convert the state into a hashable representation for caching.
    state = frozenset({utils._atom_to_pyperplan_fact(a) for a in non_static_state_atoms})

    for option_name in OPTIONS:
        if option_name in current_op.lower():
            option_type = option_name

    if option_type == "move":
        #Get the two locations: Depends on how the option name is presented.
        #Assuming name is of the form: moveloc0-loc1
        # stripped_option_name = current_op.lower().replace(option_type,"")
        # loc_start, loc_dest = stripped_option_name.split("-", 1)
        #Assuming locations are only of a single type like table0.
        # print(f"\nLoc_start:{loc_start}; Loc_dest:{loc_dest}.")
        # input()
        locations = ordered_matches(current_op.lower(), MOVE_LOCATIONS)
        # if len(locations) == 1:
        #     locations = [locations[0], locations[0]]
        assert len(locations) == 2
        loc_start, loc_dest = locations[0], locations[1]
        assert type(loc_start) == type(loc_dest) == str, f"Both loc_start, loc_dest much be of type str."

        if "home" in loc_start:
            loc_start_pos = continuous_env._home_xy
            loc_start_pos = np.array([loc_start_pos[0], loc_start_pos[1], 0.0])
        else:
            loc_start_idx = int(loc_start[-1])
            loc_start_pos = np.array(continuous_env._default_table_poses[loc_start_idx])
        
        if "home" in loc_dest:
            loc_dest_pos = continuous_env._home_xy
            loc_dest_pos = np.array([loc_dest_pos[0], loc_dest_pos[1], 0.0])
        else:
            loc_dest_idx = int(loc_dest[-1])
            loc_dest_pos = np.array(continuous_env._default_table_poses[loc_dest_idx])
        
        #Assuming they are vectors, get distance between them
        dist = np.linalg.norm(loc_start_pos-loc_dest_pos)
        #Any function based on the distance goes below this:
        time = dist/0.3
        # if time < MOVE_CONSTANT:
        #     return MOVE_CONSTANT
        return time

    elif option_type == "pick":
        #Get the item that is to be picked:
        #Based on option name format: pickbook-shelf-loc2-left_gripper
        #Here book is to be picked from shelf
        # stripped_option_name = current_op.lower().replace(option_type,"")
        # option_obj, option_loc, *_ = stripped_option_name.split("-", 2) + ["", ""]

        return 3.11

        """

        option_loc = ordered_matches(current_op.lower(), PICK_PLACE_LOCATIONS)[0]

        # Count number of piles; this is the number of blocks placed directly on
        # table.
        obj_on_table = [fact_name for fact_name in state if ("OnTable" in fact_name)
                         and (option_loc in fact_name)]
        num_piles = len(facts_for_loc)

        # Create a dict based on last string in all On predicates:
        pile_chain_dict : dict = {}
        for predicate in state:
            if "On " in predicate:
                key = predicate.split()[2].strip(")")
                pile_chain_dict[key] = predicate

        unique_pile_heights: List = []

        for fact in obj_on_table:
            on_table_block = fact.split()[2].strip(')')
            unique_pile_heights.append(get_unique_pile_height(pile_chain_dict, on_table_block))

        # idx is computed based on assumption that locations are: table0, table1 etc.
        loc_idx = int(option_loc[-1])
        workspace_bounds = continuous_env._table_workspaces[loc_idx]
        workspace_area = compute_area(workspace_bounds)

        #Any other logic or function to be computed on objects on a loc where pick is to be performed goes below:
        option_cost = (PICK_CONSTANT * obj_count_on_loc)/workspace_area

        return option_cost

        """
                

    elif option_type == "place" or option_type == "stack" or option_type == "put":
        #CURRENTLY, THIS IS IMPLEMENTED BASED ON THE ASSUMPTION THAT 
        #SIMILAR FACTORS AFFECT BOTH PICK AND PLACE
        #Get the item that is to be picked:
        #Based on option name format: placecup-table-loc0-left_gripper
        #Here book is to be picked from shelf
        # stripped_option_name = current_op.lower().replace(option_type,"")
        # option_obj, option_loc, *_ = stripped_option_name.split("-", 2) + ["", ""]

        return 3.11

        """
        option_loc = ordered_matches(current_op.lower(), PICK_PLACE_LOCATIONS)[0]

        # Count number of piles; this is the number of blocks placed directly on
        # table.
        obj_on_table = [fact_name for fact_name in state if ("OnTable" in fact_name)
                         and (option_loc in fact_name)]
        num_piles = len(facts_for_loc)

        # Create a dict based on last string in all On predicates:
        pile_chain_dict : dict = {}
        for predicate in state:
            if "On " in predicate:
                key = predicate.split()[2].strip(")")
                pile_chain_dict[key] = predicate

        unique_pile_heights: List = []

        for fact in obj_on_table:
            on_table_block = fact.split()[2].strip(')')
            unique_pile_heights.append(get_unique_pile_height(pile_chain_dict, on_table_block))


        # idx is computed based on assumption that locations are: table0, table1 etc.
        loc_idx = int(option_loc[-1])
        workspace_bounds = continuous_env._table_workspaces[loc_idx]
        workspace_area = compute_area(workspace_bounds)

        #Any other logic or function to be computed on objects on a loc where pick
        # is to be performed goes below:
        # TODO: Write a proper expression that incorporates both pile num and its 
        # height for cost.
        option_cost = (PLACE_CONSTANT * num_piles)/workspace_area

        return option_cost

        """

    else:
        raise NotImplementedError(f"\n Geometric cost computation not "
                                    "implemented for option_type:{option_type}.")




def get_geometric_cost(current_op, ops, facts, state, continuous_env):
    """Receives the current_operator object to compute cost for,
    list of operator objects, facts dict, symbolic state, continuous state.
    Returns: the value of executing current option in
    current state.
    """
    # ipdb.set_trace()
    option_type = None

    ops_dict = {operator.name: operator for operator in ops}
    current_symbolic_state_approximation = approximate_current_symbolic_state(state, ops_dict, facts, current_op)

    for option_name in OPTIONS:
        if option_name in current_op.name.lower():
            option_type = option_name

    assert option_type is not None, f"option_type can't be None for current option:{current_op.name}."

    if option_type == "move":
        #Get the two locations: Depends on how the option name is presented.
        #Assuming name is of the form: moveloc0-loc1
        # stripped_option_name = current_op.name.lower().replace(option_type,"")
        # loc_start, loc_dest = stripped_option_name.split("-", 1)
        #Assuming locations are only of a single type like table0.
        # print(f"\nLoc_start:{loc_start}; Loc_dest:{loc_dest}.")
        # input()
        locations = ordered_matches(current_op.name.lower(), MOVE_LOCATIONS)
        # if len(locations) == 1:
        #     locations = [locations[0], locations[0]]
        assert len(locations) == 2
        loc_start, loc_dest = locations[0], locations[1]
        assert type(loc_start) == type(loc_dest) == str, f"Both loc_start, loc_dest must be of type str."

        if "home" in loc_start:
            loc_start_pos = continuous_env._home_xy
            loc_start_pos = np.array([loc_start_pos[0], loc_start_pos[1], 0.0])
        else:
            loc_start_idx = int(loc_start[-1])
            loc_start_pos = np.array(continuous_env._default_table_poses[loc_start_idx])
        
        if "home" in loc_dest:
            loc_dest_pos = continuous_env._home_xy
            loc_dest_pos = np.array([loc_dest_pos[0], loc_dest_pos[1], 0.0])
        else:
            loc_dest_idx = int(loc_dest[-1])
            loc_dest_pos = np.array(continuous_env._default_table_poses[loc_dest_idx])
        #Assuming they are vectors, get distance between them
        dist = np.linalg.norm(loc_start_pos-loc_dest_pos)
        #Any function based on the distance goes below this:
        # DIST_LIST.append(MOVE_CONSTANT)
        # print(DIST_LIST)
        time = dist/0.3
        # if time < MOVE_CONSTANT:
        #     return MOVE_CONSTANT
        return time

    elif option_type == "pick":
        #Get the item that is to be picked:
        #Based on option name format: pickbook-shelf-loc2-left_gripper
        #Here book is to be picked from shelf
        # stripped_option_name = current_op.name.lower().replace(option_type,"")
        # option_obj, option_loc, *_ = stripped_option_name.split("-", 2) + ["", ""]

        return 3.11

        """ 
        option_loc = ordered_matches(current_op.name.lower(), PICK_PLACE_LOCATIONS)[0]

        # Count number of piles; this is the number of blocks placed directly on
        # table.
        obj_on_table = [fact_name for fact_name in state if ("OnTable" in fact_name)
                         and (option_loc in fact_name)]
        num_piles = len(facts_for_loc)

        # Create a dict based on last string in all On predicates:
        pile_chain_dict : dict = {}
        for predicate in state:
            if "On " in predicate:
                key = predicate.split()[2].strip(")")
                pile_chain_dict[key] = predicate

        unique_pile_heights: List = []

        for fact in obj_on_table:
            on_table_block = fact.split()[2].strip(')')
            unique_pile_heights.append(get_unique_pile_height(pile_chain_dict, on_table_block))

        # idx is computed based on assumption that locations are: table0, table1 etc.
        loc_idx = int(option_loc[-1])
        workspace_bounds = continuous_env._table_workspaces[loc_idx]
        workspace_area = compute_area(workspace_bounds)

        #Any other logic or function to be computed on objects on a loc where pick is to be performed goes below:
        option_cost = (PICK_CONSTANT * obj_count_on_loc)/workspace_area

        # DIST_LIST.append(option_cost)
        # print(DIST_LIST)
        return option_cost
        """
                

    elif option_type == "place" or option_type == "stack" or option_type == "put":
        #CURRENTLY, THIS IS IMPLEMENTED BASED ON THE ASSUMPTION THAT 
        #SIMILAR FACTORS AFFECT BOTH PICK AND PLACE
        #Get the item that is to be picked:
        #Based on option name format: placecup-table-loc0-left_gripper
        #Here book is to be picked from shelf
        # stripped_option_name = current_op.name.lower().replace(option_type,"")
        # option_obj, option_loc, *_ = stripped_option_name.split("-", 2) + ["", ""]

        return 3.11

        """
        option_loc = ordered_matches(current_op.name.lower(), PICK_PLACE_LOCATIONS)[0]

        # Count number of piles; this is the number of blocks placed directly on
        # table.
        obj_on_table = [fact_name for fact_name in state if ("OnTable" in fact_name)
                         and (option_loc in fact_name)]
        num_piles = len(facts_for_loc)

        # Create a dict based on last string in all On predicates:
        pile_chain_dict : dict = {}
        for predicate in state:
            if "On " in predicate:
                key = predicate.split()[2].strip(")")
                pile_chain_dict[key] = predicate

        unique_pile_heights: List = []

        for fact in obj_on_table:
            on_table_block = fact.split()[2].strip(')')
            unique_pile_heights.append(get_unique_pile_height(pile_chain_dict, on_table_block))

        # idx is computed based on assumption that locations are: table0, table1 etc.
        loc_idx = int(option_loc[-1])
        workspace_bounds = continuous_env._table_workspaces[loc_idx]
        workspace_area = compute_area(workspace_bounds)

        #Any other logic or function to be computed on objects on a loc where pick is to be performed goes below:
        option_cost = (PLACE_CONSTANT * obj_count_on_loc)/workspace_area

        # DIST_LIST.append(option_cost)
        # print(DIST_LIST)
        return option_cost
        """

    else:
        raise NotImplementedError(f"\n Geometric cost computation not "
                                    "implemented for option_type:{option_type}.")





def _get_operator_chain(state, fact, facts, ops_dict, found_ops, visiting=None, 
                        depth=0, max_depth=1000):
    """Follow the chain of operators that led to the current op becoming 
    finite/available.  On cycle: return the partial chain collected so far 
    (stop at the cycle).
    """
    # print(f"\n_____________________________________________")
    # print(f"\nProcessing fact: {fact.name}")
    # print(f"\nFact distance: {fact.distance}")
    # if fact.cheapest_achiever is not None:
    #     print(f"\nFact cheapest_achiever: {fact.cheapest_achiever.name}")
    # else:
    #     print(f"\nFact cheapest_achiever: None")
    # print(f"\nCurrent state: {state}")
    # print(f"\n_____________________________________________")

    if fact.name in state:
        return []

    if visiting is None:
        visiting = set()

    # if fact.cheapest_achiever is None:
    #     return []

    # cheapest_op_for_fact = ops_dict[fact.cheapest_achiever.name]

    try:
        cheapest_op_for_fact = ops_dict[fact.cheapest_achiever.name]
    except KeyError as e:
        raise KeyError(
            f"Unknown cheapest_achiever '{fact.cheapest_achiever}' for fact '{fact.name}'"
        ) from e

    op_id = getattr(cheapest_op_for_fact, "name", cheapest_op_for_fact)

    if cheapest_op_for_fact in found_ops:
        return []

    if op_id in visiting:
        return []

    visiting.add(op_id)
    cheapest_op_preconds = []

    try:
        cheapest_op_preconds = [facts[precond] for precond in cheapest_op_for_fact.preconditions]

        # stop deep/degenerate chains but still include current op
        if depth >= max_depth:
            return [cheapest_op_for_fact]
        # assert len(cheapest_op_preconds) > 0, f"List of preconditions can't be empty."
        most_expensive_precondition = max(cheapest_op_preconds, key=attrgetter("distance"))
        if most_expensive_precondition.distance == 0:
            return [cheapest_op_for_fact]
        chained_ops = _get_operator_chain(state, most_expensive_precondition, facts, ops_dict,
                                                found_ops, visiting, depth+1, max_depth
                                                )
        return [cheapest_op_for_fact] + chained_ops
    finally:
        visiting.discard(op_id)


def approximate_current_symbolic_state(state, ops_dict, facts, current_op):
    """Takes in the state for which the heuristic was called, the list of
    operators, and facts and the current operator for which the cost needs
    to be computed.
    """

    approximate_current_state = list(state)

    #Arrange current_op's preconditions in increasing order of 
    #distance:
    current_op_preconditions = [facts[precond] for precond in current_op.preconditions]
    ordered_pre_conditions = sorted(current_op_preconditions, key=lambda f: f.distance)
    # print(f"\n______________________________________________________________________")
    # print(f"\nProcessing option for cost: {current_op.name}")
    # print(f"\nPreconditions to process in order: {[(precond.name, precond.distance) for precond in ordered_pre_conditions]}")
    # print(f"\n______________________________________________________________________")

    #Start with current_op's predicates
    # - go over all its peconds in increasing order of distance
    # - for each of them trace back to cheapest achievers till 
    #   most expensive precond is not a part of state.
    # Store the ops in each precond's chain
    # Don't follow a chain deeper if it already exists in the list
    # available ops.
    topological_sorted_ops = {}
    found_ops = set()
    # ipdb.set_trace()
    for fact in ordered_pre_conditions:
        topological_sorted_ops[fact] = _get_operator_chain(state, fact, facts, ops_dict, found_ops)
        # found_ops = (op for val in topological_sorted_ops.values() for op in val)
        found_ops.update(topological_sorted_ops[fact])


    #Apply operators corresponding to pre-conditions for current_op:
    for fact in topological_sorted_ops:
        for op in topological_sorted_ops[fact]:
            #If some preconds are unsatisfied, add them to the state
            unsatisfied_preconds = list(set(op.preconditions)-set(approximate_current_state))
            if unsatisfied_preconds:
                approximate_current_state.extend(unsatisfied_preconds)

            approximate_current_state.extend(list(op.add_effects))
            approximate_current_state = list(set(approximate_current_state) - set(op.delete_effects))

    return approximate_current_state


def extrapolate_continuous_state(state, approximate_current_state, **kwargs):
    """Get an approximate continuous representation of current approximate symbolic
    state based on continuous state of objects at initialization.
    """
    raise NotImplementedError(f"Function to approximate continuous state not implemented.")


