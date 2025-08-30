from typing import List
from predicators.envs.pddl_env import _FixedTasksPDDLEnv
from predicators import utils

class FixedTasksMyDomainEnv(_FixedTasksPDDLEnv):
    @classmethod
    def get_name(cls) -> str:
        # This name is used by the framework for dynamic discovery.
        return "multitasking_room"

    @classmethod
    def get_domain_str(cls) -> str:
        # Load your domain file.
        path = utils.get_env_asset_path("pddl/multitasking/multitable/multitable_domain.pddl")
        with open(path, encoding="utf-8") as f:
            return f.read()

    @property
    def _pddl_problem_asset_dir(self) -> str:
        # The folder name (inside assets/pddl) where your task files reside.
        return "multitasking/multitable"

    @property
    def _train_problem_indices(self) -> List[int]:
        # List the indices for training tasks.
        # For example, if you have one fixed problem file:
        return [0]

    @property
    def _test_problem_indices(self) -> List[int]:
        # Similarly, list indices for testing.
        return [0]