from predicators.envs import get_or_create_env
from predicators.approaches import create_approach
from predicators.planning import sesame_plan
from predicators.ground_truth_models import get_gt_options
from predicators.settings import GlobalSettings,CFG

import traceback
import random

#print("CFG settings:", CFG.__dict__)

#hack for the CFG.sesame_max_skeletons_optimized:
#CFG.sesame_max_skeletons_optimized = 8
#CFG.option_model_name = "oracle"

#  ensures that sesame_max_skeletons_optimized (and any other arg-specific settings) are actually added to CFG.
extra_settings = GlobalSettings.get_arg_specific_settings({"env": "multitasking_room", "approach": "oracle"})
for k, v in extra_settings.items():
   setattr(CFG, k, v)

CFG.env = "multitasking_room"



def setup_and_plan_with_astar():
    env_name = "multitasking_room"  # Replace with your environment name

    # set here becuase of the error; this is a hack until I figure out what exactly is it that I am doing wrong
    # since this error is being thrown by an internal function which is inherently being called by the code.
    # It might be that I am not initializing somethings/ function/ classes in the correct manner.
    CFG.seed = random.randint(0,10000)
    env = get_or_create_env(env_name)
    
    predicates = env.predicates
    options = get_gt_options(env_name)

    #######--DEBUG--#########

    # Check if predicates are empty
    if not predicates:
        print("No predicates found for the environment.")
    else:
        print("Predicates found:")
        for predicate in predicates:
            print(f"  - {predicate}")

    print("******************************************")

    # Check if options are empty
    if not options:
        print("No options found for the environment.")
    else:
        print("Options found:")
        for option in options:
            print(f"  - {option}")

    print("******************************************")

    #########################

    
    # Create the approach (you can choose any approach that fits your needs)
    approach = create_approach("oracle",  # or "pg3", "pg4", etc.
                                predicates,
                                options,
                                env.types,
                                env.action_space,
                                env.get_train_tasks)
    
    task = env.get_train_tasks()[0]  # Choose the first training task
    
    print(task)
    print("******************************************")

    print("\nInitial State Atoms:")
    for atom in task.init_obs:
        print(f"  {atom}")
    
    print("\nGoal Atoms:")
    for atom in task.goal:
        print(f"  {atom}")
    
    # Use the sesame_plan function to generate a plan using Fast Downward
    try:
        plan, skeleton, metrics = sesame_plan(
            task,
            approach._option_model,  # Use the option model from the approach
            approach._nsrts,        # Use the NSRTs from the approach
            predicates,
            env.types,
            timeout=100000,
            seed=CFG.seed,
            task_planning_heuristic=CFG.sesame_task_planning_heuristic,
            max_skeletons_optimized=CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
            heuristic_method = "hmax"
        )
        
        print("\nGenerated Plan:")
        # for i, option in enumerate(plan):
        #     print(f"{i+1}. {option}")

        print(plan)
        
        print("\nMetrics:")
        print(metrics)
        
    except Exception as e:
        print(f"Planning failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    setup_and_plan_with_astar()