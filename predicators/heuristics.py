import numpy as np
from heapq import *
import time
import logging
import functools
import ipdb

from .heuristic_base import Heuristic
from predicators import utils
from predicators import geometric_eval
from predicators.structs import Object, Predicate, State, GroundAtom, _GroundNSRT
from predicators.utils import _create_pyperplan_task, _atom_to_pyperplan_fact
from predicators.envs.blocks import BlocksEnv
from typing import Set, List, Dict, Any, FrozenSet, Sequence, Optional

logging.basicConfig(level=logging.INFO)

# Forward-declare this type for the type hint.
_PyperplanTask = Any 


# class CombinedHeuristic:
#     def __init__(self, pyperplan_task: _PyperplanTask, heuristic_type: str = "lmcut"):
#         # self.init_atoms = init_atoms
#         # self.goal_atoms = goal_atoms
#         # self.ground_nsrts = ground_nsrts
#         # self.heuristic_type = heuristic_type
#         # self.hadd = HAddHeuristic(init_atoms, goal_atoms, ground_nsrts)
#         # self.lmcut = LMCutHeuristic(init_atoms, goal_atoms, ground_nsrts)

#         self.heuristic_type = heuristic_type

#         if self.heuristic_type == "lmcut":
#             self.lmcut = LMCutHeuristic(pyperplan_task)

#         elif self.heuristic_type != "lmcut":
#             raise ValueError(f"Unknown or unsupported heuristic type: {self.heuristic_type}.")

#     def __call__(self, state_atoms: Set[GroundAtom]) -> float:
#         # if self.heuristic_type == "hadd":
#         #     return self.hadd(state_atoms)
#         if self.heuristic_type == "lmcut":
#             # LMCutHeuristic expects a node with a .state attribute
#             # containing string facts.
#             class DummyNode:
#                 def __init__(self, state_facts):
#                     self.state = state_facts

#             pyperplan_facts = {_atom_to_pyperplan_fact(a) for a in state_atoms}
#             node = DummyNode(pyperplan_facts)
#             return self.lmcut(node)
#         else:
#             raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")

class CombinedHeuristic:
    def __init__(self, init_atoms: Set[GroundAtom], goal_atoms:Set[GroundAtom],
                 ground_nsrts: List[_GroundNSRT], predicates: Set[Predicate], 
                 objects: Sequence[Object], continuous_env: Optional[BlocksEnv] = None,
                 heuristic_type: str = "lmcut"):
        self.heuristic_type = heuristic_type

        # Compute and store static atoms. This is the key change.
        self.static_atoms = utils.get_static_atoms(ground_nsrts, init_atoms)

        pyperplan_task = _create_pyperplan_task(init_atoms, goal_atoms, ground_nsrts,
                                                predicates, objects, self.static_atoms)

        # This will be used when you re-implement HAdd
        # if self.heuristic_type == "hadd":
        #     self.hadd = HAddHeuristic(pyperplan_task)

        if self.heuristic_type == "lmcut":
            self.heuristic = LMCutHeuristic(pyperplan_task)
        elif self.heuristic_type == "hmax":
            self.heuristic = HMaxHeuristic(pyperplan_task)
        elif self.heuristic_type == "hadd":
            self.heuristic = HAddHeuristic(pyperplan_task)
        elif self.heuristic_type == "haddgeometric":
            self.heuristic = HAddGeometricHeuristic(pyperplan_task, continuous_env)
        else:
            raise ValueError(f"Unknown or unsupported heuristic type: {self.heuristic_type}")

    def __call__(self, state_atoms: Set[GroundAtom]) -> float:

        # Filter out the static atoms from the input state:
        non_static_state_atoms = set(state_atoms)-self.static_atoms

        # Convert the state into a hashable representation for caching.
        pyperplan_facts = frozenset({_atom_to_pyperplan_fact(a) for a in non_static_state_atoms})

        # if self.heuristic_type == "lmcut":
        #     return self._evaluate_lmcut(pyperplan_facts, self.lmcut)

        # Call the cached evaluation method.
        return self._evaluate_heuristic(pyperplan_facts, self.heuristic) 

        raise ValueError(f"Heuristic logic for {self.heuristic_type} not implemented in __call__.")

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _evaluate_heuristic(pyperplan_facts: frozenset, heuristic_obj: Heuristic) -> float:
        """
        Caches the result of the heuristic evaluation.
        """
        # A dummy node is needed because the heuristic expects a .state attribute
        class DummyNode:
            def __init__(self, state_facts):
                self.state = state_facts
        
        node = DummyNode(pyperplan_facts)
        return heuristic_obj(node)



def _compare(op):
    """
    General compare function for objects containing hmax values.
    This is picked up straight from Pyperplan.
    """

    def comp(self, x):
        m = getattr(self.hmax_value, op)
        return m(x.hmax_value)

    return comp


class RelaxedFact:
    def __init__(self, name):
        self.name = name
        self.hmax_value = float("inf")
        self.precondition_of = list()  # list of RelaxedOp
        self.effect_of = list()  # list of RelaxedOp

    # We want to be able to insert RelaxedFact into a heap.
    # We thus use a general compare function here
    # and instantiate the __lt__, __gt__ etc. class methods with this function.
    (__lt__, __leq__, __gt__, __geq__) = map(
        _compare, ["__lt__", "__leq__", "__gt__", "__geq__"]
    )

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, RelaxedFact) and self.name == other.name

    def clear(self):
        self.hmax_value = float("inf")

    def dump(self):
        return "< FACT name: {}, hmax: {:f}, precond_of: {}, effect_of: {} >".format(
            self.name,
            self.hmax_value,
            [str(p) for p in self.precondition_of],
            [str(e) for e in self.effect_of],
        )

    def __str__(self):
        return self.name

    __repr__ = dump


class RelaxedOp:
    def __init__(self, name, cost_zero=False):
        self.name = name
        # list of RelaxedFact
        self.precondition = list()
        # list of RelaxedFact
        self.effects = list()
        # the most expensive predecessor (a RelaxedFact)
        self.hmax_supporter = None
        self.hmax_value = float("inf")
        self.cost_zero = cost_zero
        # used to check whether an operator can be applied
        self.preconditions_unsat = 0
        if self.cost_zero:
            self.cost = 0.0
        else:
            self.cost = 1.0

    # We want to be able to insert RelaxedOp into a heap.
    # We thus use a general compare function for Operators here
    # and instantiate the __lt__, __gt__ etc. class methods with this function.
    (__lt__, __leq__, _gt__, __geq__) = map(
        _compare, ["__lt__", "__leq__", "__gt__", "__geq__"]
    )

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, RelaxedOp) and self.name == other.name

    def clear(self, clear_op_cost):
        """This method resets the operator values to its defaults.

        It is called during the hmax computation on each operator.
        Effect:
        -------
        clears preconditions_unsat
        sets cost to 1
        """
        self.preconditions_unsat = len(self.precondition)
        if clear_op_cost and not self.cost_zero:
            self.cost = 1.0
        self.hmax_supporter = None
        self.hmax_value = float("inf")

    def dump(self):
        return (
            "< OPERATOR name: %s, "
            "hmax_supp: %s, precond: %s, effects: %s, cost: %d >"
            % (
                self.name,
                str(self.hmax_supporter),
                [str(p) for p in self.precondition],
                [str(e) for e in self.effects],
                self.cost,
            )
        )

    def __str__(self):
        return self.name

    __repr__ = dump


class LMCutHeuristic(Heuristic):
    """Class and methods for computing the LM-cut heuristic value.

    We define some constant names for special facts and operators.
    NOTE: we use upper case names here as the PDDL tasks generally do not
    contain any upper case names. This way it is ensured that the denominators
    'ALWAYSTRUE', 'GOAL' and 'GOALOP' are always unique.
    """

    # operators without precondition get ALWAYSTRUE as precondition
    always_true = "ALWAYSTRUE"
    # we use this to have a single goal instead of multiple goals
    explicit_goal = "GOAL"
    goal_operator_name = "GOALOP"

    def __init__(self, task):
        self.relaxed_facts = dict()  # fact name -> RelaxedFact
        self.relaxed_ops = dict()
        self.reachable = set()
        self.goal_plateau = set()
        self.dead_end = True

        self._compute_relaxed_facts_and_operators(task)

    def _compute_relaxed_facts_and_operators(self, task):
        """Store all facts from the task as relaxed facts into our dict."""

        # little helper functions that build the relaxed operator graph
        def link_op_to_precondition(relaxed_op, factname):
            relaxed_op.precondition.append(self.relaxed_facts[factname])
            self.relaxed_facts[factname].precondition_of.append(relaxed_op)

        def link_op_to_effect(relaxed_op, factname):
            relaxed_op.effects.append(self.relaxed_facts[factname])
            self.relaxed_facts[factname].effect_of.append(relaxed_op)

        for fact in task.facts:
            self.relaxed_facts[fact] = RelaxedFact(fact)

        for op in task.operators:
            assert not op.name in self.relaxed_ops
            # build new relaxed operator from the task operator
            relaxed_op = RelaxedOp(op.name)
            # insert all preconditions into relaxed_op and
            # mark all preconditions in the relaxed_facts

            if not op.preconditions:
                # insert one fact that is always true if not already defined
                # --> this fact will be used for all operators with empty
                # preconditions
                if not self.always_true in self.relaxed_facts:
                    self.relaxed_facts[self.always_true] = RelaxedFact(self.always_true)
                link_op_to_precondition(relaxed_op, self.always_true)
            else:
                for fact in op.preconditions:
                    assert fact in self.relaxed_facts
                    link_op_to_precondition(relaxed_op, fact)
            # insert all effects into relaxed_op and
            # mark all effects in relaxed_facts
            for fact in op.add_effects:
                assert fact in self.relaxed_facts
                link_op_to_effect(relaxed_op, fact)
            # insert relaxed_op into hash
            self.relaxed_ops[op.name] = relaxed_op

        # insert explicit goal and goal operator
        goalfact = RelaxedFact(self.explicit_goal)
        goalop = RelaxedOp(self.goal_operator_name, True)
        self.relaxed_facts[self.explicit_goal] = goalfact
        self.relaxed_ops[self.goal_operator_name] = goalop

        link_op_to_effect(goalop, self.explicit_goal)

        # link all goals to the explicit goal
        for fact in task.goals:
            assert fact in self.relaxed_facts
            link_op_to_precondition(goalop, fact)

    def compute_hmax(self, state, clear_op_cost=True):
        """Compute hmax values with a Dijkstra like procedure."""
        self.reachable.clear()
        facts_seen = set()
        unexpanded = []
        op_cleared = set()
        fact_cleared = set()
        start_state = {x for x in state}
        if self.always_true in self.relaxed_facts:
            start_state.add(self.always_true)
        for fact in start_state:
            self.reachable.add(fact)
            fact_obj = self.relaxed_facts[fact]
            fact_obj.hmax_value = 0.0
            # mark all initial facts such that they are not cleared again!
            fact_cleared.add(fact_obj)
            facts_seen.add(fact_obj)
            heappush(unexpanded, fact_obj)
        while unexpanded:
            fact_obj = heappop(unexpanded)
            if fact_obj == self.relaxed_facts[self.explicit_goal]:
                self.dead_end = False
            # store fact as reachable
            self.reachable.add(fact_obj)
            hmax_value = fact_obj.hmax_value
            # update all operators that have this fact
            # as their precondition
            for op in fact_obj.precondition_of:
                # check if we have explored this operator in this iteration
                # --> if this is not the case then precond_fulfilled might
                # still contain facts from a previous heuristic computation
                # hence we need to clear it first!
                if not op in op_cleared:
                    op.clear(clear_op_cost)
                    op_cleared.add(op)
                op.preconditions_unsat -= 1
                # first check if all preconditions are fullfilled
                if op.preconditions_unsat == 0:
                    # update hmax_supporter if necessary
                    if (
                        op.hmax_supporter is None
                        or hmax_value > op.hmax_supporter.hmax_value
                    ):
                        op.hmax_supporter = fact_obj
                        # store for next hmax iteration
                        op.hmax_value = hmax_value + op.cost
                    hmax_next = op.hmax_supporter.hmax_value + op.cost
                    for eff in op.effects:
                        if not eff in fact_cleared:
                            # clear fact if necessary
                            eff.clear()
                            fact_cleared.add(eff)
                        if hmax_next < eff.hmax_value:
                            eff.hmax_value = hmax_next
                        if not eff in facts_seen:
                            # enqueue effect if not already explored
                            facts_seen.add(eff)
                            heappush(unexpanded, eff)

    def compute_hmax_from_last_cut(self, state, last_cut):
        """This computes hmax values starting from the last cut.

        This saves us from recomputing the hmax values of all facts/operators
        that have not changed anyway.
        NOTE: a complete cut procedure needs to be finished (i.e. one cut must
        be computed) for this to work!
        """
        unexpanded = []
        # add all operators from the last cut
        # to the queue of operators for which the hmax value needs to be
        # recomouted
        for op in last_cut:
            op.hmax_value = op.hmax_supporter.hmax_value + op.cost
            heappush(unexpanded, op)
        while unexpanded:
            # iterate over all operators whose effects might need updating
            op = heappop(unexpanded)
            next_hmax = op.hmax_value
            # op_seen.add(op)
            for fact_obj in op.effects:
                # if hmax value of this fact is outdated
                fact_hmax = fact_obj.hmax_value
                if fact_hmax > next_hmax:
                    # update hmax value
                    # logging.debug('updating %s' % fact_obj)
                    fact_obj.hmax_value = next_hmax
                    # enqueue all ops of which fact_obj is a hmax supporter
                    for next_op in fact_obj.precondition_of:
                        if next_op.hmax_supporter == fact_obj:
                            next_op.hmax_value = next_hmax + next_op.cost
                            for supp in next_op.precondition:
                                if supp.hmax_value + next_op.cost > next_op.hmax_value:
                                    next_op.hmax_supporter = supp
                                    next_op.hmax_value = supp.hmax_value + next_op.cost
                            heappush(unexpanded, next_op)

    def compute_goal_plateau(self, fact_name):
        """Recursively mark a goal plateau."""
        # assure the fact itself is not in an unreachable region
        fact_in_plateau = self.relaxed_facts[fact_name]
        if (
            fact_in_plateau in self.reachable
            and not fact_in_plateau in self.goal_plateau
        ):
            # add this fact to the goal plateau
            self.goal_plateau.add(fact_in_plateau)
            for op in fact_in_plateau.effect_of:
                # recursive call to mark hmax_supporters of all operators
                if op.cost == 0:
                    self.compute_goal_plateau(op.hmax_supporter.name)

    def find_cut(self, state):
        """This returns the set of relaxed operators which are in the cut."""
        unexpanded = []
        facts_seen = set()
        op_cleared = set()
        cut = set()

        start_state = {x for x in state}
        if self.always_true in self.relaxed_facts:
            start_state.add(self.always_true)
        for fact in start_state:
            assert fact in self.relaxed_facts
            fact_obj = self.relaxed_facts[fact]
            facts_seen.add(fact_obj)
            heappush(unexpanded, fact_obj)
        while unexpanded:
            fact_obj = heappop(unexpanded)
            for relaxed_op in fact_obj.precondition_of:
                if not relaxed_op in op_cleared:
                    relaxed_op.precond_unsat = len(relaxed_op.precondition)
                    op_cleared.add(relaxed_op)
                relaxed_op.precond_unsat -= 1
                # check if the operator preconditions are all satisfied
                if relaxed_op.precond_unsat == 0:
                    # if so we can expand this operator
                    for eff in relaxed_op.effects:
                        if eff in facts_seen:
                            continue
                        if eff in self.goal_plateau:
                            cut.add(relaxed_op)
                        else:
                            facts_seen.add(eff)
                            heappush(unexpanded, eff)
        return cut

    def __call__(self, node):
        state = node.state
        heuristic_value = 0.0
        goal_state = self.relaxed_facts[self.explicit_goal]
        # reset dead end flag
        # --> asume node to be a dead end unless proven otherwise by the hmax
        # computation
        self.dead_end = True
        # next find all cuts
        # first compute hmax starting from the current state
        self.compute_hmax(state, True)
        if goal_state.hmax_value == float("inf"):
                return float("inf")
        while goal_state.hmax_value != 0:
            # next find an appropriate cut
            # first calculate the goal plateau
            self.goal_plateau.clear()
            self.compute_goal_plateau(self.explicit_goal)
            # then find the cut itself
            cut = self.find_cut(state)
            # finally update heuristic value
            min_cost = min([o.cost for o in cut])
            # logging.debug("compute cut done")
            heuristic_value += min_cost
            for o in cut:
                o.cost -= min_cost
                logging.debug(repr(o))
            # compute next hmax
            self.compute_hmax_from_last_cut(state, cut)
        if self.dead_end:
            return float("inf")
        else:
            return heuristic_value


##################################################################################################
############## Implementation of h_max and h_add.#################################################
############## For these, we slight change in the way data structures are defined.################
############## This structure and logic is completely inspired from Pyperplan's implementation.###
############## Some would say I copied it from there and they's be around 80% correct. :-)########
##################################################################################################



class _RelaxedFactForH:
    """A relaxed fact specifically for h_add and h_max."""
    def __init__(self, name):
        self.name = name
        self.precondition_of = []
        self.expanded = False
        self.distance = float("inf")
        self.cheapest_achiever = None

    def __eq__(self, other):
        return isinstance(other, _RelaxedFactForH) and self.name == other.name

    def __hash__(self):
        return hash((_RelaxedFactForH, self.name))


class _RelaxedOperatorForH:
    """A relaxed operator specifically for h_add and h_max."""
    def __init__(self, name, preconditions, add_effects, delete_effects=None):
        self.name = name
        self.preconditions = preconditions
        self.remaining = set(preconditions)
        self.add_effects = add_effects
        if delete_effects is not None:
            self.delete_effects = delete_effects
        self.cost = 1
        self.counter = len(self.preconditions)

    def __eq__(self, other):
        return isinstance(other, _RelaxedOperatorForH) and self.name == other.name

    def __hash__(self):
        return hash((_RelaxedOperatorForH, self.name))


class _PyperplanRelaxationHeuristicBase(Heuristic):
    """
    This is a port of Pyperplan's _RelaxationHeuristic base class.
    It implements the core Dijkstra-style search for h_add and h_max.
    """
    def __init__(self, task: _PyperplanTask):
        self.facts = {fact: _RelaxedFactForH(fact) for fact in task.facts}
        self.operators = []
        self.goals = {pred for pred in task.goals}
        self.init = task.initial_state
        self.tie_breaker = 0
        self.start_state = _RelaxedFactForH("start")
        self.eval = sum  # Default to h_add
        self.NAME = "HADD"
        self.state = None

        for op in task.operators:
            ro = _RelaxedOperatorForH(op.name, op.preconditions, op.add_effects)
            self.operators.append(ro)
            for var in op.preconditions:
                self.facts[var].precondition_of.append(ro)
            if not op.preconditions:
                self.start_state.precondition_of.append(ro)

    def __call__(self, node):
        self.state = set(node.state)
        # ipdb.set_trace()
        self._init_distance(self.state)
        heap = []
        heappush(heap, (0, self.tie_breaker, self.start_state))
        self.tie_breaker += 1

        for fact in self.state:
            heappush(
                heap, (self.facts[fact].distance, self.tie_breaker, self.facts[fact])
            )
            self.tie_breaker += 1
        
        self._dijkstra(heap)
        h_value = self._calc_goal_h()
        return h_value

    def _init_distance(self, state):

        def reset_fact(fact):
            fact.expanded = False
            fact.cheapest_achiever = None
            if fact.name in state:
                fact.distance = 0
            else:
                fact.distance = float("inf")
        reset_fact(self.start_state)
        for fact in self.facts.values():
            reset_fact(fact)
        for operator in self.operators:
            operator.counter = len(operator.preconditions)

    def _get_cost(self, operator):
        if operator.preconditions:
            cost = self.eval(
                [self.facts[pre].distance for pre in operator.preconditions]
            )
        else:
            cost = 0
        if self.NAME == "HGEOMETRIC":
            # ipdb.set_trace()
            return cost + geometric_eval.get_geometric_cost(operator, self.operators, self.facts,
                                                             self.state, self.continuous_env)
        return cost + operator.cost

    def _calc_goal_h(self):
        if not self.goals:
            return 0
        # If any goal is unreachable, the value is infinity.
        goal_distances = [self.facts[fact].distance for fact in self.goals]
        if float("inf") in goal_distances:
            return float("inf")
        return self.eval(goal_distances)

    def finished(self, achieved_goals, queue):
        """
        This function is used as a stopping criterion for the Dijkstra search,
        which differs for different heuristics.
        """
        return achieved_goals == self.goals or not queue

    def _dijkstra(self, queue):
        achieved_goals = set()
        while not self.finished(achieved_goals, queue):
            #Pop last fact from heap, check if it's in goal,
            #Add to achieved_goals if it is.
            _dist, _tie, fact = heappop(queue)
            if fact.name != 'start':
                assert fact.distance != float("inf"), f"\n Fact with infinite distance can't make it\
                                                        into the queue."
            if fact.name in self.goals:
                achieved_goals.add(fact.name)
                # print(f"\nCurrent set of achieved goals:{achieved_goals}.")
                # print(f"\nGoals are:{self.goals}")
                # print(f"\nCurrent queue length: {len(queue)}.")
                
            
            # For h_max, we can stop if all goals are reached.
            # if self.eval == max and achieved_goals == self.goals:
            #     break
            
            if not fact.expanded:
                for operator in fact.precondition_of:
                    # ipdb.set_trace()
                    if fact.name in operator.remaining:
                        if fact.distance != float("inf"):
                            operator.remaining.remove(fact.name)
                        else:
                            continue
                        operator.counter -= 1
                        #Process operator if all its pre-conditions are met:
                        # if operator.counter <= 0:
                        if not operator.remaining:
                            for item in operator.preconditions:
                                if self.facts[item].distance == float("inf"):
                                    ipdb.set_trace()
                            for n in operator.add_effects:
                                neighbor = self.facts[n]
                                tmp_dist = self._get_cost(operator)
                                assert tmp_dist != float("inf"), f"\nNeighbour distance can't be \
                                                                    inf when being pushed into the queue."
                                #Update distance/h_max value for facts in
                                #add effects if new value is less than 
                                #current value.
                                if tmp_dist < neighbor.distance:
                                    neighbor.distance = tmp_dist
                                    neighbor.cheapest_achiever = operator
                                    heappush(
                                        queue, (tmp_dist, self.tie_breaker, neighbor)
                                    )

                                    self.tie_breaker += 1

                fact.expanded = True

            if self.finished(achieved_goals, queue):
                # ipdb.set_trace()
                print(f"\nAt the end of current loop, Finished returns:\
                                             {self.finished(achieved_goals, queue)}.")


# Now, the H_max and H_add heuristics are mere wrappers around the _PyperplanRelaxationHeuristicBase.

class HAddHeuristic(_PyperplanRelaxationHeuristicBase):
    """
    An implementation of the h_add heuristic that conforms to the
    Pyperplan task interface. It is a subclass of the main relaxation
    heuristic implementation.
    """
    def __init__(self, task: _PyperplanTask):
        super().__init__(task)
        self.eval = sum
        self.NAME = "HADD"


class HMaxHeuristic(_PyperplanRelaxationHeuristicBase):
    """
    An implementation of the h_max heuristic that conforms to the
    Pyperplan task interface. It is a subclass of the main relaxation
    heuristic implementation.
    """
    def __init__(self, task: _PyperplanTask):
        super().__init__(task)
        self.eval = max
        self.NAME = "HMAX"


class HAddGeometricHeuristic(_PyperplanRelaxationHeuristicBase):
    """
    An implementation of the h_add heuristic that conforms to the
    Pyperplan task interface. It is a subclass of the main relaxation
    heuristic implementation.
    """
    def __init__(self, task: _PyperplanTask, continuous_env: BlocksEnv):
        super().__init__(task)

        self.operators = []

        for op in task.operators:
            ro = _RelaxedOperatorForH(op.name, op.preconditions, op.add_effects, op.del_effects)
            self.operators.append(ro)
            for var in op.preconditions:
                self.facts[var].precondition_of.append(ro)
            if not op.preconditions:
                self.start_state.precondition_of.append(ro)

        self.eval = sum
        self.NAME = "HGEOMETRIC"
        self.continuous_env = continuous_env