import os
import sys
import time
import copy
import logging
import numpy as np
from operator import attrgetter
from typing import List, Dict, Set, Any, Sequence, Optional

from predicators.pybullet_helpers.geometry import Pose, Pose3D


class StateGeometricCost:

	def __init__(self, data: Dict):
		for key, value in data.items():
			setattr(self, key, value)
		self.present_attributes()

	def present_attributes(self):
		print(f"\nContinuous state attributes present:")
		for attr in self.__dict__:
			print(f"\n{attr}")


	def get_geometric_cost(self, current_op, ops, facts, option_types, state):
		"""Receives the current_operator object to compute cost for,
		list of operator objects, facts dict, and the state.
		Returns: the value of executing current option in
		current state.
		"""
		ops_dict = {operator.name: operator for operator in ops}
		current_symbolic_state_approximation = self.approximate_current_symbolic_state(state, ops_dict, facts, current_op)

		for option_name in self.options:
			if option_name in current_op.name:
				option_type = option_name

		if option_type == "move":
			#Get the two locations: Depends on how the option name is presented.
			#Assuming name is of the form: moveloc0-loc1
			stripped_option_name = current_op.name.replace(option_type)
			loc_start, loc_dest = stripped_option_name.split("-", 1)
			#Assuming the class has an attribute: self.locations
			loc_start_pos = self.locations[loc_start].position
			loc_dest_pos = self.locations[loc_dest].position
			#Assuming they are vectors, get distance between them
			dist = np.linalg.norm(loc_start_pos-loc_dest_pos)
			#Any function based on the distance goes below this:
			return dist

		elif option_type == "pick":
			#Get the item that is to be picked:
			#Based on option name format: pickbook-shelf-loc2-left_gripper
			#Here book is to be picked from shelf
			stripped_option_name = current_op.name.replace(option_type)
			option_obj, option_loc, *_ = text.split("-", 2) + ["", ""]

			facts_for_loc = [fact_name for fact_name in state if "item_at" in fact_name and option_loc in fact_name]
			obj_count_on_loc = len(facts_for_loc)

			#Any other logic or function to be computed on objects on a loc where pick is to be performed goes here

			return obj_count_on_loc
					

		elif option_type == "place":
			#CURRENTLY, THIS IS IMPLEMENTED BASED ON THE ASSUMPTION THAT 
			#SIMILAR FACTORS AFFECT BOTH PICK AND PLACE
			#Get the item that is to be picked:
			#Based on option name format: placecup-table-loc0-left_gripper
			#Here book is to be picked from shelf
			stripped_option_name = current_op.name.replace(option_type)
			option_obj, option_loc, *_ = text.split("-", 2) + ["", ""]

			facts_for_loc = [fact_name for fact_name in state if "item_at" in fact_name and option_loc in fact_name]
			obj_count_on_loc = len(facts_for_loc)

			#Any other logic or function to be computed on objects on a loc where pick is to be performed goes here

			return obj_count_on_loc

		else:
			raise NotImplementedError(f"\n Geometric cost computation not "
										"implemented for option_type:{option_type}.")





	def _get_operator_chain(self, state, fact, facts, ops_dict, found_ops, visiting=None, 
							depth=0, max_depth=1000):
		"""Follow the chain of operators that led to the current op becoming 
		finite/available.  On cycle: return the partial chain collected so far 
		(stop at the cycle).
		"""
		if fact.name in state:
			return []

		if visiting is None:
			visiting = set()

		try:
			cheapest_op_for_fact = ops_dict[fact.cheapest_achiever]
		except KeyError as e:
	        raise KeyError(
	            f"Unknown cheapest_achiever '{fact.cheapest_achiever}' for fact '{fact.name}'"
	        ) from e

	    op_id = getattr(op, "name", op)

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
			chained_ops = self._get_operator_chain(state, most_expensive_precondition, facts, ops_dict,
												 	found_ops, visiting, depth+1, max_depth
												 	)
			return [cheapest_op_for_fact] + chained_ops
		finally:
			visiting.discard(op_id)


	def approximate_current_symbolic_state(self, state, ops_dict, facts, current_op):
		"""Takes in the state for which the heuristic was called, the list of
		operators, and facts and the current operator for which the cost needs
		to be computed.
		"""

		approximate_current_state = list(state)

		#Arrange current_op's preconditions in increasing order of 
		#distance:
		current_op_preconditions = [facts[precond] for precond in current_op.preconditions]
		ordered_pre_conditions = sorted(current_op_preconditions, key=lambda f: f.distance)

		#Start with current_op's predicates
		# - go over all its peconds in increasing order of distance
		# - for each of them trace back to cheapest achievers till 
		#	most expensive precond is not a part of state.
		# Store the ops in each precond's chain
		# Don't follow a chain deeper if it already exists in the list
		# available ops.
		topological_sorted_ops = {}
		found_ops = ()
		for fact in ordered_pre_conditions:
			topological_sorted_ops[fact] = self._get_operator_chain(state, fact, facts, ops_dict, found_ops)
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


	def extrapolate_continuous_state(self, state, approximate_current_state, **kwargs):
		"""Get an approximate continuous representation of current approximate symbolic
		state based on continuous state of objects at initialization.
		"""
		raise NotImplementedError(f"Function to approximate continuous state not implemented.")


