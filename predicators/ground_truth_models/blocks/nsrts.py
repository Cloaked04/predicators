"""Ground-truth NSRTs for the blocks environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class BlocksGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks", "pybullet_blocks", "blocks_clear"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        robot_type = types["robot"]

        # Predicates
        On = predicates["On"]
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
        PutOnTable = options["PutOnTable"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, robot]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [block])}
        delete_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # Unstack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, otherblock, robot]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(unstack_nsrt)

        # Stack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, otherblock, robot]
        option_vars = [robot, otherblock]
        option = Stack
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        add_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, robot]
        option_vars = [robot]
        option = PutOnTable
        preconditions = {LiftedAtom(Holding, [block])}
        add_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(Holding, [block])}

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        return nsrts




class PyBulletMultiTableBlocksGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for multi-table blocks env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_multitable_blocks"}

    @staticmethod
    def get_nsrts(env_name: str,
                  types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        robot_type = types["robot"]
        table_type = types["table"]

        # Predicates (reuse the same ones the blocks env exposes;
        # if your multitable env adds more, pull them here too)
        On = predicates["On"]
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        RobotAt = predicates["RobotAt"]

        # Options — these *must* match the names your multitable option factory exports
        Pick        = options["Pick"]
        Stack       = options["Stack"]
        PutOnTable  = options["PutOnTable"]
        Move        = options["Move"]

        nsrts: Set[NSRT] = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [block, robot, table]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(OnTable, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [block])}
        delete_effects = {
            LiftedAtom(OnTable, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)

        nsrts.add(pickfromtable_nsrt)


        # Unstack
        block  = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters=[block, otherblock, robot, table]
        option_vars = [robot, block, table]
        option=Pick
        preconditions={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(OnTable, [otherblock, table],)
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
        }

        unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(unstack_nsrt)


        # Stack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [block, otherblock, robot, table]
        option_vars = [robot, otherblock]
        option = Stack
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(OnTable, [otherblock, table]),
            LiftedAtom(RobotAt, [robot, table])
        }
        add_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [block, robot, table]
        option_vars = [robot]
        option = PutOnTable
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(RobotAt, [robot, table])
            }
        add_effects = {
            LiftedAtom(OnTable, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(Holding, [block])}


        def put_on_table_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)

        nsrts.add(putontable_nsrt)

        #Move:
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        othertable = Variable("?table", table_type)

        parameters = [robot, table, othertable]
        option_vars = [robot, othertable]
        option = Move

        preconditions = {LiftedAtom(RobotAt, [robot, table])}
        add_effects = {LiftedAtom(RobotAt, [robot, othertable])}
        delete_effects = {LiftedAtom(RobotAt, [robot, table])}

        #Do we need a sampler?

        move_nsrt = NSRT("Move", parameters, preconditions, add_effects,
                        delete_effects, set(), option, option_vars, null_sampler)

        nsrts.add(move_nsrt)



        return nsrts

