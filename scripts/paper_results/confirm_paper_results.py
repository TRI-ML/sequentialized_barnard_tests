"""Script to print out all (most) paper results from camera-ready version.
"""

import copy
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from tqdm import tqdm

from sequentialized_barnard_tests import (
    Hypothesis,
    MirroredLaiTest,
    MirroredSaviTest,
    MirroredStepTest,
)

if __name__ == "__main__":
    """
    Script to confirm paper results. Prints all results to terminal.

    Runtime should be on the order of ~5 seconds
    """

    # Set the data path
    paper_data_path = str(
        Path(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../tests/eval_data/",
            )
        ).resolve()
    )

    # Load the paper data
    eval_clean_up_spill = np.load(
        f"{paper_data_path}/TRI_CLEAN_SPILL_v4.npy"
    )  # Must be flipped for standard form
    eval_fold_red_towel = np.load(
        f"{paper_data_path}/TRI_FOLD_RED_TOWEL.npy"
    )  # ALREADY in standard form
    eval_sim_spoon_on_towel = np.load(
        f"{paper_data_path}/TRI_SIM_SPOON_ON_TOWEL.npy"
    )  # Must be flipped for standard form
    eval_sim_eggplant_in_basket = np.load(
        f"{paper_data_path}/TRI_SIM_EGGPLANT_IN_BASKET.npy"
    )  # Must be flipped for standard form
    eval_sim_stack_cube = np.load(
        f"{paper_data_path}/TRI_SIM_STACK_CUBE.npy"
    )  # Must be flipped for standard form

    # Load SAVI tests
    savi_hardware = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05)
    savi_simulation = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=0.01)

    # Load Lai tests
    lai_hardware_50 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=50, alpha=0.05
    )
    lai_hardware_200 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=200, alpha=0.05
    )
    lai_hardware_200.set_c(0.00014741399676752065)

    lai_hardware_500 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=500, alpha=0.05
    )
    lai_hardware_500.set_c(5.349419043278717e-05)

    lai_simulation_500 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=500, alpha=0.01
    )
    lai_simulation_500.set_c(1.184327928758278e-05)

    # Load STEP tests
    step_random_seed = 42

    step_hardware_50 = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=50,
        alpha=0.05,
        random_seed=step_random_seed,
    )
    step_hardware_200 = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=200,
        alpha=0.05,
        random_seed=step_random_seed,
    )
    step_hardware_500 = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=500,
        alpha=0.05,
        random_seed=step_random_seed,
    )
    step_simulation_500 = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=500,
        alpha=0.01,
        random_seed=step_random_seed,
    )

    # Run without index permutation
    permutation_idx_hardware = np.arange(50)
    permutation_idx_simulation = np.arange(500)

    # Run appropriate tests on each data stream

    ##############################
    ### Result 1: FoldRedTowel ###
    ##############################
    foldredtowel_result_lai_50 = lai_hardware_50.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )
    foldredtowel_result_lai_200 = lai_hardware_200.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )
    foldredtowel_result_lai_500 = lai_hardware_500.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )

    foldredtowel_result_savi = savi_hardware.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )

    foldredtowel_result_step_50 = step_hardware_50.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )
    foldredtowel_result_step_200 = step_hardware_200.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )
    foldredtowel_result_step_500 = step_hardware_500.run_on_sequence(
        eval_fold_red_towel[permutation_idx_hardware, 0],
        eval_fold_red_towel[permutation_idx_hardware, 1],
    )

    print()
    print("FOLD RED TOWEL: ")
    print()
    print("Lai-50  time-to-decision: ", foldredtowel_result_lai_50.info["Time"])
    print("Lai-200  time-to-decision: ", foldredtowel_result_lai_200.info["Time"])
    print("Lai-500  time-to-decision: ", foldredtowel_result_lai_500.info["Time"])
    print("STEP-50 time-to-decision: ", foldredtowel_result_step_50.info["Time"])
    print("STEP-200 time-to-decision: ", foldredtowel_result_step_200.info["Time"])
    print("STEP-500 time-to-decision: ", foldredtowel_result_step_500.info["Time"])
    print(
        "SAVI     time-to-decision: ",
        foldredtowel_result_savi.info["result_for_alternative"].info["Time"],
    )

    ##############################
    ### Result 2: CleanUpSpill ###
    ##############################
    cleanupspill_result_lai_50 = lai_hardware_50.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )
    cleanupspill_result_lai_200 = lai_hardware_200.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )
    cleanupspill_result_lai_500 = lai_hardware_500.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )

    cleanupspill_result_savi = savi_hardware.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )

    cleanupspill_result_step_50 = step_hardware_50.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )
    cleanupspill_result_step_200 = step_hardware_200.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )
    cleanupspill_result_step_500 = step_hardware_500.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 0],
        eval_clean_up_spill[permutation_idx_hardware, 1],
    )

    print()
    print("CLEAN UP SPILL: ")
    print()
    print("Lai-50  time-to-decision: ", cleanupspill_result_lai_50.info["Time"])
    print("Lai-200  time-to-decision: ", cleanupspill_result_lai_200.info["Time"])
    print("Lai-500  time-to-decision: ", cleanupspill_result_lai_500.info["Time"])
    print("STEP-50 time-to-decision: ", cleanupspill_result_step_50.info["Time"])
    print("STEP-200 time-to-decision: ", cleanupspill_result_step_200.info["Time"])
    print("STEP-500 time-to-decision: ", cleanupspill_result_step_500.info["Time"])
    print(
        "SAVI     time-to-decision: ",
        cleanupspill_result_savi.info["result_for_alternative"].info["Time"],
    )

    ##############################
    ### Result 3: SpoonOnTowel ###
    ##############################
    spoonontowel_result_lai_500 = lai_simulation_500.run_on_sequence(
        eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
        eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
    )
    spoonontowel_result_savi = savi_simulation.run_on_sequence(
        eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
        eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
    )
    spoonontowel_result_step_500 = step_simulation_500.run_on_sequence(
        eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
        eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
    )

    print()
    print("SPOON ON TOWEL: ")
    print()
    print("Lai-500  time-to-decision: ", spoonontowel_result_lai_500.info["Time"])
    print("STEP-500 time-to-decision: ", spoonontowel_result_step_500.info["Time"])
    print(
        "SAVI     time-to-decision: ",
        spoonontowel_result_savi.info["result_for_alternative"].info["Time"],
    )

    ##################################
    ### Result 4: EggplantInBasket ###
    ##################################
    eggplantinbasket_result_lai_500 = lai_simulation_500.run_on_sequence(
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
    )
    eggplantinbasket_result_savi = savi_simulation.run_on_sequence(
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
    )
    eggplantinbasket_result_step_500 = step_simulation_500.run_on_sequence(
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
    )

    print()
    print("EGGPLANT IN BASKET: ")
    print()
    print("Lai-500  time-to-decision: ", eggplantinbasket_result_lai_500.info["Time"])
    print("STEP-500 time-to-decision: ", eggplantinbasket_result_step_500.info["Time"])
    print(
        "SAVI     time-to-decision: ",
        eggplantinbasket_result_savi.info["result_for_alternative"].info["Time"],
    )

    ###########################
    ### Result 5: StackCube ###
    ###########################
    stackcube_result_lai_500 = lai_simulation_500.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 0],
        eval_sim_stack_cube[permutation_idx_simulation, 1],
    )
    stackcube_result_savi = savi_simulation.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 0],
        eval_sim_stack_cube[permutation_idx_simulation, 1],
    )
    stackcube_result_step_500 = step_simulation_500.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 0],
        eval_sim_stack_cube[permutation_idx_simulation, 1],
    )

    print()
    print("STACK CUBE: ")
    print()
    print("Lai-500  time-to-decision: ", stackcube_result_lai_500.info["Time"])
    print("STEP-500 time-to-decision: ", stackcube_result_step_500.info["Time"])
    print(
        "SAVI     time-to-decision: ",
        stackcube_result_savi.info["result_for_alternative"].info["Time"],
    )
