import numpy as np
from typing import Set, List, Dict, Any, FrozenSet
from predicators.structs import GroundAtom, _GroundNSRT
import heapq
import time
import logging

logging.basicConfig(level=logging.INFO)



class CombinedHeuristic:
    def __init__(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], 
                 ground_nsrts: List[_GroundNSRT], heuristic_type: str = "hadd"):
        self.init_atoms = init_atoms
        self.goal_atoms = goal_atoms
        self.ground_nsrts = ground_nsrts
        self.heuristic_type = heuristic_type
        self.hadd = HAddHeuristic(init_atoms, goal_atoms, ground_nsrts)
        self.lmcut = LMCutHeuristic(init_atoms, goal_atoms, ground_nsrts)

    def __call__(self, state_atoms: Set[GroundAtom]) -> float:
        if self.heuristic_type == "hadd":
            return self.hadd(state_atoms)
        elif self.heuristic_type == "lmcut":
            return self.lmcut(state_atoms)
        else:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")



class HAddHeuristic:
    """Implements the h_add heuristic for A* search in task planning.
    
    The h_add heuristic sums the estimated cost of achieving each goal atom 
    independently in a relaxed problem where delete effects are ignored.
    """
    
    def __init__(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], 
                 ground_nsrts: List[_GroundNSRT]):
        """Initialize the heuristic.
        
        Args:
            init_atoms: The initial state atoms (not used directly but kept for interface consistency)
            goal_atoms: The goal state atoms that must be achieved
            ground_nsrts: List of ground NSRTs (actions)
        """
        self.goal_atoms = goal_atoms
        self.ground_nsrts = ground_nsrts
    
    def __call__(self, state_atoms: Set[GroundAtom]) -> float:
        """Calculate the h_add heuristic for the given state.
        
        Args:
            state_atoms: The current state (set of ground atoms that are true)
            
        Returns:
            The h_add heuristic value (sum of costs for each goal atom)
        """
        # Collect all atoms that appear in the problem
        all_atoms = set()
        all_atoms.update(state_atoms)
        all_atoms.update(self.goal_atoms)
        for nsrt in self.ground_nsrts:
            all_atoms.update(nsrt.preconditions)
            all_atoms.update(nsrt.add_effects)
        
        # Initialize costs: 0 for atoms in the current state, infinity otherwise
        costs: Dict[GroundAtom, float] = {}
        for atom in all_atoms:
            costs[atom] = 0 if atom in state_atoms else float('inf')
        
        # Fixed-point iteration until costs stabilize
        changed = True
        while changed:
            changed = False
            for nsrt in self.ground_nsrts:
                # Check if all preconditions are achievable
                precond_cost = 0
                all_achievable = True
                
                for precond in nsrt.preconditions:
                    if costs.get(precond, float('inf')) == float('inf'):
                        all_achievable = False
                        break
                    precond_cost += costs[precond]
                
                if all_achievable:
                    # Update costs for add effects
                    action_cost = 1.0 + precond_cost  # Action cost + precondition cost
                    for atom in nsrt.add_effects:
                        if action_cost < costs.get(atom, float('inf')):
                            costs[atom] = action_cost
                            changed = True
        
        # Sum up the costs for all goal atoms
        total_cost = 0
        for atom in self.goal_atoms:
            if costs.get(atom, float('inf')) < float('inf'):
                total_cost += costs[atom]
            else:
                # If any goal atom can't be achieved, return infinity
                return float('inf')
        
        return total_cost




class RelaxedFact:
    """
    Represents a relaxed fact (atom) in the LM-Cut computation.
    """

    def __init__(self, atom):
        self.atom = atom  # Can be a GroundAtom or other type
        self.name = str(atom)
        self.hmax_value = float("inf")
        self.precondition_of = []  # List of RelaxedOp objects
        self.effect_of = []        # List of RelaxedOp objects

    def __lt__(self, other):
        return self.hmax_value < other.hmax_value

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, RelaxedFact):
            return self.name == other.name
        return False

    def clear(self):
        self.hmax_value = float("inf")

    def __str__(self):
        return self.name


class RelaxedOp:
    """
    Represents a relaxed operator in the LM-Cut computation.
    """

    def __init__(self, nsrt=None, name="", cost_zero=False):
        self.nsrt = nsrt
        self.name = name if name else str(nsrt)
        self.precondition = []  # List of RelaxedFact objects
        self.effects = []       # List of RelaxedFact objects
        self.hmax_supporter = None  # Most expensive precondition (RelaxedFact)
        self.hmax_value = float("inf")
        self.cost_zero = cost_zero
        self.preconditions_unsat = 0
        self.cost = 0.0 if cost_zero else 1.0

    def __lt__(self, other):
        return self.hmax_value < other.hmax_value

    def clear(self, clear_op_cost):
        self.preconditions_unsat = len(self.precondition)
        if clear_op_cost and not self.cost_zero:
            self.cost = 1.0
        self.hmax_supporter = None
        self.hmax_value = float("inf")

    def __str__(self):
        return self.name


class LMCutHeuristic:
    """
    LM-Cut heuristic (without fallback).
    """

    def __init__(self, init_atoms: Set[Any], goal_atoms: Set[Any], ground_nsrts: List[Any]):
        # Special names.
        self.ALWAYS_TRUE = "ALWAYSTRUE"
        self.EXPLICIT_GOAL = "GOAL"
        self.GOAL_OP_NAME = "GOALOP"

        self.init_atoms = init_atoms
        self.goal_atoms = goal_atoms
        self.ground_nsrts = ground_nsrts

        # Data structures for the relaxed planning graph.
        self.relaxed_facts = {}  # Mapping: atom -> RelaxedFact.
        self.relaxed_ops = {}    # Mapping: nsrt or special name -> RelaxedOp.
        self.reachable = set()
        self.goal_plateau = set()
        self.dead_end = True

        # Cache for computed heuristic values.
        self.heuristic_cache = {}

        # Overall time budget (in seconds).
        self.max_time = 10.0

        self._build_relaxed_planning_graph()
        logging.info(f"LM-Cut: {len(self.relaxed_facts)} facts, {len(self.relaxed_ops)} operators")

    def _build_relaxed_planning_graph(self):
        """Constructs the relaxed planning graph from NSRTs."""
        def link_op_to_pre(op, fact_obj):
            op.precondition.append(fact_obj)
            fact_obj.precondition_of.append(op)

        def link_op_to_eff(op, fact_obj):
            op.effects.append(fact_obj)
            fact_obj.effect_of.append(op)

        # Collect all atoms from NSRTs, initial state, and goals.
        all_atoms = set()
        for nsrt in self.ground_nsrts:
            all_atoms.update(nsrt.preconditions)
            all_atoms.update(nsrt.add_effects)
        all_atoms.update(self.init_atoms)
        all_atoms.update(self.goal_atoms)

        for atom in all_atoms:
            self.relaxed_facts[atom] = RelaxedFact(atom)

        # Add the special always-true fact.
        always_true_fact = RelaxedFact(self.ALWAYS_TRUE)
        self.relaxed_facts[self.ALWAYS_TRUE] = always_true_fact

        # Create relaxed operators for each NSRT.
        for nsrt in self.ground_nsrts:
            op = RelaxedOp(nsrt=nsrt)
            self.relaxed_ops[nsrt] = op
            if not nsrt.preconditions:
                link_op_to_pre(op, always_true_fact)
            else:
                for pre in nsrt.preconditions:
                    link_op_to_pre(op, self.relaxed_facts[pre])
            for eff in nsrt.add_effects:
                link_op_to_eff(op, self.relaxed_facts[eff])

        # Create the explicit goal fact and corresponding goal operator.
        goal_fact = RelaxedFact(self.EXPLICIT_GOAL)
        self.relaxed_facts[self.EXPLICIT_GOAL] = goal_fact

        goal_op = RelaxedOp(name=self.GOAL_OP_NAME, cost_zero=True)
        self.relaxed_ops[self.GOAL_OP_NAME] = goal_op
        link_op_to_eff(goal_op, goal_fact)
        for g in self.goal_atoms:
            link_op_to_pre(goal_op, self.relaxed_facts[g])

    def compute_hmax(self, state_atoms, clear_op_cost=True):
        """Compute hmax values using a Dijkstra-like propagation (without local expansion limits)."""
        self.reachable.clear()
        facts_seen = set()
        unexpanded = []
        op_cleared_in_this_hmax_computation = set()
        fact_cleared_in_this_hmax_computation = set() # Keep track of facts cleared in this call

        # Reset all facts' hmax_values
        for fact_obj in self.relaxed_facts.values():
            fact_obj.clear()
            # No need to add to fact_cleared_in_this_hmax_computation here, 
            # as they are reset. Clearing happens when an effect is considered.

        # Initialize with the current state facts.
        for atom in state_atoms:
            if atom in self.relaxed_facts:
                fact_obj = self.relaxed_facts[atom]
                fact_obj.hmax_value = 0.0
                facts_seen.add(fact_obj)
                heapq.heappush(unexpanded, fact_obj)
                fact_cleared_in_this_hmax_computation.add(fact_obj) # Mark initial facts as 'cleared' (i.e., processed for init)
        # Also add the always-true fact.
        if self.ALWAYS_TRUE in self.relaxed_facts:
            fact_obj = self.relaxed_facts[self.ALWAYS_TRUE]
            fact_obj.hmax_value = 0.0
            facts_seen.add(fact_obj)
            heapq.heappush(unexpanded, fact_obj)
            fact_cleared_in_this_hmax_computation.add(fact_obj) # Mark initial facts as 'cleared'

        while unexpanded:
            fact_obj = heapq.heappop(unexpanded)
            self.reachable.add(fact_obj) # Add to reachable when popped, as in pyperplan
            # Mark goal as reachable.
            if self.EXPLICIT_GOAL in self.relaxed_facts and fact_obj == self.relaxed_facts[self.EXPLICIT_GOAL]:
                self.dead_end = False
            current_hmax_val = fact_obj.hmax_value # Renamed for clarity
            for op in fact_obj.precondition_of:
                # Clear op only once per hmax computation
                if op not in op_cleared_in_this_hmax_computation:
                    op.clear(clear_op_cost)
                    op_cleared_in_this_hmax_computation.add(op)

                op.preconditions_unsat -= 1
                if op.preconditions_unsat == 0:
                    # Update hmax_supporter and hmax_value for the operator
                    if op.hmax_supporter is None or current_hmax_val > op.hmax_supporter.hmax_value:
                        op.hmax_supporter = fact_obj
                    # op.hmax_value should be based on its current hmax_supporter
                    op.hmax_value = op.hmax_supporter.hmax_value + op.cost
                    
                    h_next = op.hmax_value # Use the operator's hmax_value for propagation

                    for eff in op.effects:
                        if eff not in fact_cleared_in_this_hmax_computation:
                            eff.clear() # Clear fact if not yet processed in this hmax computation
                            fact_cleared_in_this_hmax_computation.add(eff)
                        if h_next < eff.hmax_value:
                            eff.hmax_value = h_next
                        if eff not in facts_seen:
                            facts_seen.add(eff)
                            heapq.heappush(unexpanded, eff)

    def compute_hmax_from_last_cut(self, last_cut):
        """Recompute hmax values incrementally from the last computed cut."""
        unexpanded = []
        for op in last_cut:
            if op.hmax_supporter is not None:
                op.hmax_value = op.hmax_supporter.hmax_value + op.cost
                heapq.heappush(unexpanded, op)
            else:
                logging.warning(f"Operator {op.name} has no hmax_supporter")
        while unexpanded:
            op = heapq.heappop(unexpanded)
            next_val = op.hmax_value
            for fact_obj in op.effects:
                if fact_obj.hmax_value > next_val:
                    fact_obj.hmax_value = next_val
                    for next_op in fact_obj.precondition_of:
                        if next_op.hmax_supporter == fact_obj:
                            next_op.hmax_value = next_val + next_op.cost
                            for supp in next_op.precondition:
                                if supp.hmax_value + next_op.cost > next_op.hmax_value:
                                    next_op.hmax_supporter = supp
                                    next_op.hmax_value = supp.hmax_value + next_op.cost
                            heapq.heappush(unexpanded, next_op)

    def compute_goal_plateau(self, fact_name):
        """Recursively mark the goal plateau, following pyperplan's design."""
        if fact_name not in self.relaxed_facts:
            return
        fact = self.relaxed_facts[fact_name]
        if fact not in self.reachable or fact in self.goal_plateau:
            return
        self.goal_plateau.add(fact)
        for op in fact.effect_of:
            if op.cost == 0 and op.hmax_supporter is not None:
                self.compute_goal_plateau(op.hmax_supporter.name)

    def find_cut(self, state_atoms):
        """Extract a cut from the justification graph."""
        unexpanded = []
        facts_seen = set()
        op_cleared = set()
        cut = set()
        start_state = set(state_atoms)
        if self.ALWAYS_TRUE in self.relaxed_facts:
            start_state.add(self.ALWAYS_TRUE)
        for atom in start_state:
            # Ensure atom exists in relaxed_facts before trying to access it.
            # This can happen if state_atoms contains atoms not in the initial problem construction.
            if atom not in self.relaxed_facts:
                # Optionally log a warning or handle as a special case
                # logging.warning(f"Atom {atom} from state_atoms not in relaxed_facts during find_cut.")
                continue
            fact_obj = self.relaxed_facts[atom]
            facts_seen.add(fact_obj)
            heapq.heappush(unexpanded, fact_obj)
        while unexpanded:
            fact_obj = heapq.heappop(unexpanded)
            for op in fact_obj.precondition_of:
                if op not in op_cleared:
                    op.preconditions_unsat = len(op.precondition) # Corrected attribute name
                    op_cleared.add(op)
                op.preconditions_unsat -= 1 # Corrected attribute name
                if op.preconditions_unsat == 0: # Corrected attribute name
                    for eff in op.effects:
                        if eff in facts_seen:
                            continue
                        if eff in self.goal_plateau:
                            cut.add(op)
                        else:
                            facts_seen.add(eff)
                            heapq.heappush(unexpanded, eff)
        return cut

    def __call__(self, state_atoms: Set[Any]) -> float:
        """
        Compute the LM-Cut heuristic value.
        If the computation exceeds the allowed time budget, a TimeoutError is raised.
        """
        state_key = frozenset(state_atoms)
        if state_key in self.heuristic_cache:
            return self.heuristic_cache[state_key]

        start_time = time.time()
        heuristic_value = 0.0
        goal_fact = self.relaxed_facts[self.EXPLICIT_GOAL]
        self.dead_end = True

        try:
            # Initial propagation.
            self.compute_hmax(state_atoms, True)
            if time.time() - start_time > self.max_time:
                raise TimeoutError("LM-Cut computation timed out during initial hmax.")

            if goal_fact.hmax_value == float("inf"):
                self.heuristic_cache[state_key] = float("inf")
                return float("inf")

            # Main LM-Cut loop.
            while goal_fact.hmax_value != 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError("LM-Cut computation timed out during cut iterations.")

                self.goal_plateau.clear()
                self.compute_goal_plateau(self.EXPLICIT_GOAL)
                cut = self.find_cut(state_atoms)
                if not cut:
                    # If no cut is found, but goal hmax is not 0 and not inf, it's an issue.
                    # This might indicate that the goal is unreachable through applicable actions from current plateau state.
                    logging.warning("LM-Cut: No cut found but goal hmax is not 0. Setting heuristic to infinity.")
                    self.heuristic_cache[state_key] = float("inf")
                    return float("inf") # Or break, leading to dead_end check
                    
                min_cost = min([op.cost for op in cut])
                
                # Pyperplan also does not have the min_cost <= 0 check.
                # It relies on hmax_from_last_cut to make progress.
                # If min_cost is 0, heuristic_value doesn't increase, costs don't change for ops already at 0.
                # The hmax_from_last_cut must then ensure goal_fact.hmax_value is driven to 0.
                # Removing this break to align more with pyperplan's implicit assumptions.
                # if min_cost <= 0:
                #     break
                heuristic_value += min_cost
                for op in cut:
                    op.cost -= min_cost
                self.compute_hmax_from_last_cut(cut)

            if self.dead_end and goal_fact.hmax_value != float("inf"):
                 # If dead_end is true, it means compute_hmax decided the goal was initially unreachable.
                 # However, if the loop finished because goal_fact.hmax_value became 0, it's not a dead end.
                 # This check is to ensure we don't return inf if the loop successfully reduced hmax to 0.
                 if goal_fact.hmax_value == 0:
                     self.dead_end = False # Correct the dead_end status

            if self.dead_end:
                self.heuristic_cache[state_key] = float("inf")
                return float("inf")
            self.heuristic_cache[state_key] = heuristic_value
            return heuristic_value

        except (TimeoutError, MemoryError) as e:
            # With no fallback, we raise the error (or alternatively, return float("inf"))
            raise e