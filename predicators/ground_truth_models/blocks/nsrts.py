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
        goal_obj_type = types["goal"]

        # Predicates
        On = predicates["On"]
        OnTable = predicates["OnTable"]
        OnTableGoalObj = predicates["OnTableGoalObj"]
        GoalObjOnBlock = predicates["GoalObjOnBlock"]
        BlockOnGoalObj = predicates["BlockOnGoalObj"]
        GoalObjOnGoalObj = predicates["GoalObjOnGoalObj"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        HoldingGoalObj = predicates["HoldingGoalObj"]
        Clear = predicates["Clear"]
        ClearGoalObj = predicates["ClearGoalObj"]
        AtHome = predicates["AtHome"]
        RobotAt = predicates["RobotAt"]
        RobotNotAt = predicates["RobotNotAt"]
        DifferentTable = predicates["DifferentTable"]
        BlockAt = predicates["BlockAt"]
        GoalObjAt = predicates["GoalObjAt"]
        GOALOBJHELD = predicates["GOALOBJHELD"]


        # Options — these *must* match the names your multitable option factory exports
        Pick = options["Pick"]
        PickGoalObj = options["PickGoalObj"]
        Stack = options["Stack"]
        StackOnGoalObj = options["StackOnGoalObj"]
        PutOnTable = options["PutOnTable"]
        PutGoalObjOnTable = options["PutGoalObjOnTable"]
        MoveTo = options["MoveTo"]
        MoveToPick = options["MoveToPick"]

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

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, putontable_sampler)

        nsrts.add(pickfromtable_nsrt)


        # PickGoalObjFromTable
        goal_obj = Variable("?goal_obj", goal_obj_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [goal_obj, robot, table]
        option_vars = [robot, goal_obj]
        option = PickGoalObj
        preconditions = {
            LiftedAtom(OnTableGoalObj, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        # Adding the GOALOBJHELD fact to enable multiple distinct objects can lead to 
        # achieving the goal; emulating the disjunctive goal
        add_effects = {LiftedAtom(HoldingGoalObj, [goal_obj]), LiftedAtom(GOALOBJHELD, [])}
        delete_effects = {
            LiftedAtom(OnTableGoalObj, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot])
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        pickgoalobjfromtable_nsrt = NSRT("PickGoalObjFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, putontable_sampler)

        nsrts.add(pickgoalobjfromtable_nsrt)


        # UnstackFromTable
        block  = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters=[block, otherblock, robot, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, block]
        option=Pick
        preconditions={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(OnTable, [otherblock, table]),
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

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackfromtable_nsrt = NSRT("UnstackFromTable", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            putontable_sampler)
        nsrts.add(unstackfromtable_nsrt)


        # UnstackGoalObjOnBlockFromTable
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters=[goal_obj, otherblock, robot, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, goal_obj]
        option=PickGoalObj
        preconditions={
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(OnTable, [otherblock, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(GOALOBJHELD, [])
        }
        delete_effects={
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackgoalobjonblockfromtable_nsrt = NSRT("UnstackGoalObjOnBlockFromTable", parameters, 
                                                    preconditions, add_effects, delete_effects, 
                                                    set(), option, option_vars,
                                                    putontable_sampler)
        nsrts.add(unstackgoalobjonblockfromtable_nsrt)

        # UnstackBlockOnGoalObjFromTable
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        block = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters=[goal_obj, block, robot, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, block]
        option=Pick
        preconditions={
            LiftedAtom(BlockOnGoalObj, [block, goal_obj]),
            LiftedAtom(OnTableGoalObj, [goal_obj, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
        }
        delete_effects={
            LiftedAtom(BlockOnGoalObj, [block, goal_obj]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackblockongoalobjfromtable_nsrt = NSRT("UnstackBlockOnGoalObjFromTable", parameters, 
                                                    preconditions, add_effects, delete_effects, 
                                                    set(), option, option_vars,
                                                    putontable_sampler)
        nsrts.add(unstackblockongoalobjfromtable_nsrt)


        # UnstackGoalObjOnGoalObjFromTable
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        othergoal_obj = Variable("?othergoal_obj", goal_obj_type)
        robot  = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters=[goal_obj, othergoal_obj, robot, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, goal_obj]
        option=PickGoalObj
        preconditions={
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(OnTableGoalObj, [othergoal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(ClearGoalObj, [othergoal_obj]),
            LiftedAtom(GOALOBJHELD, [])
        }
        delete_effects={
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackgoalobjongoalobjfromtable_nsrt = NSRT("UnstackGoalObjOnGoalObjFromTable", parameters, 
                                                    preconditions, add_effects, delete_effects, 
                                                    set(), option, option_vars,
                                                    putontable_sampler)
        nsrts.add(unstackgoalobjongoalobjfromtable_nsrt)


        # UnstackFromBlock
        block  = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        otherotherblock = Variable("?otherotherblock", block_type)

        parameters=[block, otherblock, robot, otherotherblock, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, block]
        option=Pick
        preconditions={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(On, [otherblock, otherotherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackfromblock_nsrt = NSRT("UnstackFromBlock", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            putontable_sampler)
        nsrts.add(unstackfromblock_nsrt)


        # UnstackBlockFromBlockOnGoalObj
        block  = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        otherothergoal_obj = Variable("?otherothergoal_obj", goal_obj_type)

        parameters=[block, otherblock, robot, otherothergoal_obj, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, block]
        option=Pick
        preconditions={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(BlockOnGoalObj, [otherblock, otherothergoal_obj]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects={
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackblockfromblockongoalobj_nsrt = NSRT("UnstackBlockFromBlockOnGoalObj", parameters, 
                                                    preconditions, add_effects, delete_effects, 
                                                    set(), option, option_vars, putontable_sampler)
        nsrts.add(unstackblockfromblockongoalobj_nsrt)


        # UnstackBlockFromGoalObjOnBlock
        block  = Variable("?block", block_type)
        othergoal_obj = Variable("?othergoal_obj", goal_obj_type)
        robot  = Variable("?robot", robot_type)
        otherotherblock = Variable("?otherotherblock", block_type)

        parameters=[block, othergoal_obj, robot, otherotherblock, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, block]
        option=Pick
        preconditions={
            LiftedAtom(BlockOnGoalObj, [block, othergoal_obj]),
            LiftedAtom(GoalObjOnBlock, [othergoal_obj, otherotherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearGoalObj, [othergoal_obj])
        }
        delete_effects={
            LiftedAtom(BlockOnGoalObj, [block, othergoal_obj]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackblockfromgoalobjonblock_nsrt = NSRT("UnstackBlockFromGoalObjOnBlock", parameters, 
                                                    preconditions, add_effects,
                                                    delete_effects, set(), option, option_vars,
                                                    putontable_sampler)
        nsrts.add(unstackblockfromgoalobjonblock_nsrt)


        # UnstackBlockFromGoalObjOnGoalObj
        block  = Variable("?block", block_type)
        othergoal_obj = Variable("?otherblock", goal_obj_type)
        robot  = Variable("?robot", robot_type)
        otherothergoal_obj = Variable("?otherothergoal_obj", goal_obj_type)

        parameters=[block, othergoal_obj, robot, otherothergoal_obj, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, block]
        option=Pick
        preconditions={
            LiftedAtom(BlockOnGoalObj, [block, othergoal_obj]),
            LiftedAtom(GoalObjOnGoalObj, [othergoal_obj, otherothergoal_obj]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearGoalObj, [othergoal_obj])
        }
        delete_effects={
            LiftedAtom(BlockOnGoalObj, [block, othergoal_obj]),
            LiftedAtom(BlockAt, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackblockfromgoalobjongoalobj_nsrt = NSRT("UnstackBlockFromGoalObjOnGoalObj", parameters, 
                                                    preconditions, add_effects, delete_effects, 
                                                    set(), option, option_vars, putontable_sampler)
        nsrts.add(unstackblockfromgoalobjongoalobj_nsrt)


        # UnstackGoalObjFromBlockOnBlock
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        otherotherblock = Variable("?otherotherblock", block_type)

        parameters=[goal_obj, otherblock, robot, otherotherblock, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, goal_obj]
        option=PickGoalObj
        preconditions={
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(On, [otherblock, otherotherblock]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(GOALOBJHELD, [])
        }
        delete_effects={
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackgoalobjfromblockonblock_nsrt = NSRT("UnstackGoalObjFromBlockOnBlock", parameters, preconditions, 
                                                    add_effects, delete_effects, set(), option, option_vars,
                                                    putontable_sampler)
        nsrts.add(unstackgoalobjfromblockonblock_nsrt)


        # UnstackGoalObjFromBlockOnGoalObj
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        otherblock = Variable("?otherblock", block_type)
        robot  = Variable("?robot", robot_type)
        otherothergoal_obj = Variable("?otherothergoal_obj", goal_obj_type)

        parameters=[goal_obj, otherblock, robot, otherothergoal_obj, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, goal_obj]
        option=PickGoalObj
        preconditions={
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(BlockOnGoalObj, [otherblock, otherothergoal_obj]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(GOALOBJHELD, [])
        }
        delete_effects={
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackgoalobjfromblockongoalobj_nsrt = NSRT("UnstackGoalObjFromBlockOnGoalObj", parameters, preconditions, 
                                            add_effects, delete_effects, set(), option, option_vars,
                                            putontable_sampler)
        nsrts.add(unstackgoalobjfromblockongoalobj_nsrt)


        # UnstackGoalObjFromGoalObjOnBlock
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        othergoal_obj = Variable("?othergoal_obj", goal_obj_type)
        robot  = Variable("?robot", robot_type)
        otherotherblock = Variable("?otherotherblock", block_type)

        parameters=[goal_obj, othergoal_obj, robot, otherotherblock, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, goal_obj]
        option=PickGoalObj
        preconditions={
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(GoalObjOnBlock, [othergoal_obj, otherotherblock]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(ClearGoalObj, [othergoal_obj]),
            LiftedAtom(GOALOBJHELD, [])
        }
        delete_effects={
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackgoalobjfromgoalobjonblock_nsrt = NSRT("UnstackGoalObjFromGoalObjOnBlock", parameters, preconditions, 
                                            add_effects, delete_effects, set(), option, option_vars,
                                            putontable_sampler)
        nsrts.add(unstackgoalobjfromgoalobjonblock_nsrt)


        # UnstackGoalObjFromGoalObjOnGoalObj
        goal_obj  = Variable("?goal_obj", goal_obj_type)
        othergoal_obj = Variable("?othergoal_obj", goal_obj_type)
        robot  = Variable("?robot", robot_type)
        otherothergoal_obj = Variable("?otherothergoal_obj", goal_obj_type)

        parameters=[goal_obj, othergoal_obj, robot, otherothergoal_obj, table]
        # option_vars = [robot, block, table]
        option_vars = [robot, goal_obj]
        option=PickGoalObj
        preconditions={
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(GoalObjOnGoalObj, [othergoal_obj, otherothergoal_obj]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects={
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(ClearGoalObj, [othergoal_obj]),
            LiftedAtom(GOALOBJHELD, [])
        }
        delete_effects={
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(GoalObjAt, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot]),
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        unstackgoalobjfromgoalobjongoalobj_nsrt = NSRT("UnstackGoalObjFromGoalObjOnGoalObj", parameters, preconditions, 
                                            add_effects, delete_effects, set(), option, option_vars,
                                            putontable_sampler)
        nsrts.add(unstackgoalobjfromgoalobjongoalobj_nsrt)



        # Stack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [block, otherblock, robot, table]
        option_vars = [robot, otherblock, table]
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

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)


        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          putontable_sampler)
        nsrts.add(stack_nsrt)

        # StackBlockOnGoalObj
        block = Variable("?block", block_type)
        othergoal_obj = Variable("?othergoal_obj", goal_obj_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [block, othergoal_obj, robot, table]
        option_vars = [robot, othergoal_obj, table]
        option = StackOnGoalObj
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearGoalObj, [othergoal_obj]),
            LiftedAtom(OnTableGoalObj, [othergoal_obj, table]),
            LiftedAtom(RobotAt, [robot, table])
        }
        add_effects = {
            LiftedAtom(BlockOnGoalObj, [block, othergoal_obj]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearGoalObj, [othergoal_obj])
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)


        stackblockongoalobj_nsrt = NSRT("StackBlockOnGoalObj", parameters, preconditions, 
                                        add_effects, delete_effects, set(), option, option_vars,
                                        putontable_sampler)
        nsrts.add(stackblockongoalobj_nsrt)

        # StackGoalObjOnBlock
        goal_obj = Variable("?goal_obj", goal_obj_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [goal_obj, otherblock, robot, table]
        option_vars = [robot, otherblock, table]
        option = Stack
        preconditions = {
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(OnTable, [otherblock, table]),
            LiftedAtom(RobotAt, [robot, table])
        }
        add_effects = {
            LiftedAtom(GoalObjOnBlock, [goal_obj, otherblock]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(Clear, [otherblock])
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)


        stackgoalobjonblock_nsrt = NSRT("StackGoalObjOnBlock", parameters, preconditions, 
                                        add_effects, delete_effects, set(), option, option_vars,
                                        putontable_sampler)
        nsrts.add(stackgoalobjonblock_nsrt)

        # StackGoalObjOnGoalObj
        goal_obj = Variable("?goal_obj", goal_obj_type)
        othergoal_obj = Variable("?othergoal_obj", goal_obj_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [goal_obj, othergoal_obj, robot, table]
        option_vars = [robot, othergoal_obj, table]
        option = StackOnGoalObj
        preconditions = {
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(ClearGoalObj, [othergoal_obj]),
            LiftedAtom(OnTableGoalObj, [othergoal_obj, table]),
            LiftedAtom(RobotAt, [robot, table])
        }
        add_effects = {
            LiftedAtom(GoalObjOnGoalObj, [goal_obj, othergoal_obj]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(ClearGoalObj, [othergoal_obj])
        }

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)


        stackgoalobjongoalobj_nsrt = NSRT("StackGoalObjOnGoalObj", parameters, preconditions, add_effects,
                                        delete_effects, set(), option, option_vars,
                                        putontable_sampler)
        nsrts.add(stackgoalobjongoalobj_nsrt)



        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [block, robot, table]
        option_vars = [robot, table, block]
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


        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)

        nsrts.add(putontable_nsrt)

        # PutGoalObjOnTable
        goal_obj = Variable("?goal_obj", goal_obj_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [goal_obj, robot, table]
        option_vars = [robot, table, goal_obj]
        option = PutGoalObjOnTable
        preconditions = {
            LiftedAtom(HoldingGoalObj, [goal_obj]),
            LiftedAtom(RobotAt, [robot, table])
            }
        add_effects = {
            LiftedAtom(OnTableGoalObj, [goal_obj, table]),
            LiftedAtom(ClearGoalObj, [goal_obj]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(HoldingGoalObj, [goal_obj]),
                          LiftedAtom(GOALOBJHELD, [])}


        def putontable_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putgoalobjontable_nsrt = NSRT("PutGoalObjOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)

        nsrts.add(putgoalobjontable_nsrt)

        # MoveFromHomeToTable:
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        parameters = [robot, table]
        option_vars = [robot, table]
        option = MoveTo
        preconditions = {LiftedAtom(AtHome, [robot])}
        add_effects = {LiftedAtom(RobotAt, [robot, table])}
        delete_effects = {LiftedAtom(AtHome, [robot])}

        def move_sampler(state: State, goal: Set[GroundAtom],   
                rng: np.random.Generator, objs: Sequence[Object]) -> Array:  
            return np.array([rng.uniform(0, 1)], dtype=np.float32)

        move_from_home_nsrt = NSRT(
            "MoveFromHome", parameters, preconditions,
            add_effects, delete_effects, set(),
            option, option_vars, move_sampler
        )
        nsrts.add(move_from_home_nsrt)

        #MoveFromTableToTable:
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        othertable = Variable("?othertable", table_type)

        parameters = [robot, table, othertable]
        option_vars = [robot, othertable]
        option = MoveTo

        preconditions = {LiftedAtom(RobotAt, [robot, table]),
                         LiftedAtom(DifferentTable, [table, othertable])}
        add_effects = {LiftedAtom(RobotAt, [robot, othertable])}
        delete_effects = {LiftedAtom(RobotAt, [robot, table])}

        #Do we need a sampler? Yes, in order to connect to global seed
        #based determinism but still have local randomness.
        def move_sampler(state: State, goal: Set[GroundAtom],   
                rng: np.random.Generator, objs: Sequence[Object]) -> Array:  
            return np.array([rng.uniform(0, 1)], dtype=np.float32)

        move_to_nsrt = NSRT("MoveFromTable", parameters, preconditions, add_effects,
                        delete_effects, set(), option, option_vars, move_sampler)

        nsrts.add(move_to_nsrt)

        #MoveFromHomeToPick:
        robot = Variable("?robot", robot_type)
        block = Variable("?block", block_type)
        table = Variable("?table", table_type)

        parameters = [robot, table, block]
        # parameters = [robot, table]
        option_vars = [robot, block]
        option = MoveToPick

        preconditions = {LiftedAtom(AtHome, [robot]),
                         LiftedAtom(BlockAt, [block, table])}
        add_effects = {LiftedAtom(RobotAt, [robot, table])}
        delete_effects = {LiftedAtom(AtHome, [robot])}

        move_from_home_to_pick = NSRT("MoveFromHomeToPick", parameters, preconditions, add_effects,
                                    delete_effects, set(), option, option_vars, null_sampler)
        
        #MoveFromTableToPick:
        robot = Variable("?robot", robot_type)
        block = Variable("?block", block_type)
        table = Variable("?table", table_type)
        othertable = Variable("?othertable", table_type)


        parameters = [robot, table, othertable, block]
        option_vars = [robot, block]
        option = MoveToPick

        preconditions = {LiftedAtom(RobotAt, [robot, table]),
                         LiftedAtom(BlockAt, [block, othertable])}
        add_effects = {LiftedAtom(RobotAt, [robot, othertable])}
        delete_effects = {LiftedAtom(RobotAt, [robot, table])}

        move_from_table_to_pick = NSRT("MoveFromTableToPick", parameters, preconditions, add_effects,
                                    delete_effects, set(), option, option_vars, null_sampler)



        return nsrts

