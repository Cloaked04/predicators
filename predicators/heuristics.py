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


class RelaxedFact:
    """
    A relaxed fact (atom) for LM-Cut.
    """

    def __init__(self, atom):
        self.atom = atom  # The original atom or identifier.
        self.name = str(atom)
        self.hmax_value = float("inf")
        self.precondition_of = []  # List of RelaxedOp instances.
        self.effect_of = []        # List of RelaxedOp instances.

    def __lt__(self, other):
        return self.hmax_value < other.hmax_value

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, RelaxedFact) and self.name == other.name

    def clear(self):
        self.hmax_value = float("inf")

    def __str__(self):
        return self.name


class RelaxedOp:
    """
    A relaxed operator (action) for LM-Cut.
    """

    def __init__(self, nsrt=None, name="", cost_zero=False):
        self.nsrt = nsrt
        self.name = name if name else str(nsrt)
        self.precondition = []  # List of RelaxedFact instances.
        self.effects = []       # List of RelaxedFact instances.
        self.hmax_supporter = None
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
    Improved LM-Cut heuristic (without fallback).
    Adjustments:
      - Increased overall time budget.
      - Removed artificial expansion/cut limits.
      - The goal plateau and cut extraction follow the pyperplan design.
      - If the time budget is exceeded, a TimeoutError is raised.
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

        # Initialize with the current state facts.
        for atom in state_atoms:
            if atom in self.relaxed_facts:
                fact_obj = self.relaxed_facts[atom]
                fact_obj.hmax_value = 0.0
                facts_seen.add(fact_obj)
                heapq.heappush(unexpanded, fact_obj)
                self.reachable.add(fact_obj)
        # Also add the always-true fact.
        if self.ALWAYS_TRUE in self.relaxed_facts:
            fact_obj = self.relaxed_facts[self.ALWAYS_TRUE]
            fact_obj.hmax_value = 0.0
            facts_seen.add(fact_obj)
            heapq.heappush(unexpanded, fact_obj)
            self.reachable.add(fact_obj)

        while unexpanded:
            fact_obj = heapq.heappop(unexpanded)
            # Mark goal as reachable.
            if self.EXPLICIT_GOAL in self.relaxed_facts and fact_obj == self.relaxed_facts[self.EXPLICIT_GOAL]:
                self.dead_end = False
            current = fact_obj.hmax_value
            for op in fact_obj.precondition_of:
                op.clear(clear_op_cost)
                op.preconditions_unsat -= 1
                if op.preconditions_unsat == 0:
                    if op.hmax_supporter is None or current > op.hmax_supporter.hmax_value:
                        op.hmax_supporter = fact_obj
                        op.hmax_value = current + op.cost
                    h_next = op.hmax_supporter.hmax_value + op.cost
                    for eff in op.effects:
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
            fact_obj = self.relaxed_facts[atom]
            facts_seen.add(fact_obj)
            heapq.heappush(unexpanded, fact_obj)
        while unexpanded:
            fact_obj = heapq.heappop(unexpanded)
            for op in fact_obj.precondition_of:
                if op not in op_cleared:
                    op.precond_unsat = len(op.precondition)
                    op_cleared.add(op)
                op.precond_unsat -= 1
                if op.precond_unsat == 0:
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
                    break
                min_cost = min([op.cost for op in cut])
                if min_cost <= 0:
                    break  # Prevent potential infinite loops.
                heuristic_value += min_cost
                for op in cut:
                    op.cost -= min_cost
                self.compute_hmax_from_last_cut(cut)

            if self.dead_end:
                self.heuristic_cache[state_key] = float("inf")
                return float("inf")
            self.heuristic_cache[state_key] = heuristic_value
            return heuristic_value

        except (TimeoutError, MemoryError) as e:
            # With no fallback, we raise the error (or alternatively, return float("inf"))
            raise e