"""An approach that implements a delivery-specific policy.

Example command line:
    python predicators/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Callable, cast

import numpy as np

from predicators.approaches import BaseApproach
from predicators.envs.pddl_env import _PDDLEnvState
from predicators.structs import Action, GroundAtom, State, Task


class DeliverySpecificApproach(BaseApproach):
    """Implements a delivery-specific policy."""

    @classmethod
    def get_name(cls) -> str:
        return "delivery_policy"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # This function takes a task and timeout as parameters
        # It returns a callable function (policy) that takes a State and returns an Action

        def _policy(state: State) -> Action:
            # Inner function that implements the actual policy
            # Takes a state and returns an action to execute

            # Extract the predicators and options from the state.
            options = {o.name: o for o in self._initial_options}
            predicates = {p.name: p for p in self._initial_predicates}
            types = {t.name: t for t in self._types}
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()
            locations = state.get_objects(types["loc"])
            papers = state.get_objects(types["paper"])
            at = predicates["at"]
            wants_paper = predicates["wantspaper"]
            is_home_base = predicates["ishomebase"]
            unpacked = predicates["unpacked"]
            carrying = predicates["carrying"]

            # Main loop over locations that combines location finding and action logic
            for loc in locations:
                if GroundAtom(at, [loc]) in ground_atoms:
                    current_loc = loc
                    
                    # Case 1: At home base with unpacked paper - pick it up
                    if GroundAtom(is_home_base, [current_loc]) in ground_atoms:
                        for paper in papers:
                            if GroundAtom(unpacked, [paper]) in ground_atoms:
                                pack = options["pick-up"]
                                object_args = [paper, current_loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = pack.ground(object_args, params)
                                assert ground_option.initiable(state)
                                return ground_option.policy(state)

                    # Case 2: Carrying paper - find someone who wants it and deliver
                    for paper in papers:
                        if GroundAtom(carrying, [paper]) in ground_atoms:
                            # Look for a location where someone wants a paper
                            for delivery_loc in locations:
                                if GroundAtom(wants_paper, [delivery_loc]) in ground_atoms:
                                    if delivery_loc == current_loc:
                                        # Deliver the paper
                                        deliver = options["deliver"]
                                        object_args = [paper, current_loc]
                                        params = np.zeros(0, dtype=np.float32)
                                        ground_option = deliver.ground(object_args, params)
                                        assert ground_option.initiable(state)
                                        return ground_option.policy(state)
                                    else:
                                        # Move to delivery location
                                        move = options["move"]
                                        object_args = [current_loc, delivery_loc]
                                        params = np.zeros(0, dtype=np.float32)
                                        ground_option = move.ground(object_args, params)
                                        assert ground_option.initiable(state)
                                        return ground_option.policy(state)

                    # Case 3: Not carrying paper and not at home - return to home base
                    for home_loc in locations:
                        if GroundAtom(is_home_base, [home_loc]) in ground_atoms:
                            move = options["move"]
                            object_args = [current_loc, home_loc]
                            params = np.zeros(0, dtype=np.float32)
                            ground_option = move.ground(object_args, params)
                            assert ground_option.initiable(state)
                            return ground_option.policy(state)

            raise ValueError("No valid action found for current state")

        # Return the policy function
        return _policy
