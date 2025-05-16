"""Script to print out all (most) paper results from camera-ready version.
"""

import os
from pathlib import Path

import numpy as np
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

    Runtime on the order of 18-20 minutes on semi-powerful desktop.
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
    eval_distribution_carrot_on_plate_supplement = np.load(
        f"{paper_data_path}/PU_HARDWARE_DISTRIBUTION_SUPPLEMENT.npy"
    )  # Must be flipped for standard form
    eval_policy_carrot_on_plate_supplement = np.load(
        f"{paper_data_path}/PU_HARDWARE_POLICY_SUPPLEMENT.npy"
    )  # ALREADY in standard form

    # Load SAVI tests
    savi_hardware = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05)
    savi_simulation = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=0.01)

    # Load Lai tests
    lai_hardware_50 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=50, alpha=0.05
    )
    lai_hardware_50.calibrate_c()
    print("Lai-50 value of c: ", lai_hardware_50.c)

    lai_hardware_200 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=200, alpha=0.05
    )
    lai_hardware_200.calibrate_c()
    print("Lai-200 value of c: ", lai_hardware_200.c)
    # lai_hardware_200.set_c(0.00014741399676752065)

    lai_hardware_500 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=500, alpha=0.05
    )
    lai_hardware_500.calibrate_c()
    print("Lai-500 value of c: ", lai_hardware_500.c)
    # lai_hardware_500.set_c(5.349419043278717e-05)

    lai_simulation_500 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=500, alpha=0.01
    )
    lai_simulation_500.calibrate_c()
    print("Lai-500 SIM value of c: ", lai_simulation_500.c)
    # lai_simulation_500.set_c(1.184327928758278e-05)

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

    seed_int = 42
    # rng = np.random.default_rng(seed=seed_int)
    n_runs = int(400)
    results_FRT = np.zeros((n_runs, 7))
    results_CUS = np.zeros((n_runs, 7))
    results_SOT = np.zeros((n_runs, 3))
    results_EIB = np.zeros((n_runs, 3))
    results_SC = np.zeros((n_runs, 3))
    rng = np.random.default_rng(seed=seed_int)
    sequences_fold_red_towel = np.concatenate(
        (
            rng.binomial(1, 0.56, size=(n_runs, 50, 1)),
            rng.binomial(1, 0.92, size=(n_runs, 50, 1)),
        ),
        axis=2,
    )
    assert np.mean(sequences_fold_red_towel[:, :, 0]) < np.mean(
        sequences_fold_red_towel[:, :, 1]
    )
    sequences_clean_up_spill = np.concatenate(
        (
            rng.binomial(1, 0.28, size=(n_runs, 50, 1)),
            rng.binomial(1, 0.80, size=(n_runs, 50, 1)),
        ),
        axis=2,
    )
    assert np.mean(sequences_clean_up_spill[:, :, 0]) < np.mean(
        sequences_clean_up_spill[:, :, 1]
    )
    sequences_spoon_on_towel = np.concatenate(
        (
            rng.binomial(1, 0.084, size=(n_runs, 500, 1)),
            rng.binomial(1, 0.386, size=(n_runs, 500, 1)),
        ),
        axis=2,
    )
    assert np.mean(sequences_spoon_on_towel[:, :, 0]) < np.mean(
        sequences_spoon_on_towel[:, :, 1]
    )
    sequences_eggplant_in_basket = np.concatenate(
        (
            rng.binomial(1, 0.40, size=(n_runs, 500, 1)),
            rng.binomial(1, 0.564, size=(n_runs, 500, 1)),
        ),
        axis=2,
    )
    assert np.mean(sequences_eggplant_in_basket[:, :, 0]) < np.mean(
        sequences_eggplant_in_basket[:, :, 1]
    )
    sequences_stack_cube = np.concatenate(
        (
            rng.binomial(1, 0.000, size=(n_runs, 500, 1)),
            rng.binomial(1, 0.030, size=(n_runs, 500, 1)),
        ),
        axis=2,
    )
    assert np.mean(sequences_stack_cube[:, :, 0]) < np.mean(
        sequences_stack_cube[:, :, 1]
    )
    for i in tqdm(range(n_runs)):
        ##############################
        ### Result 1: FoldRedTowel ###
        ##############################
        foldredtowel_result_lai_50 = lai_hardware_50.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )
        foldredtowel_result_lai_200 = lai_hardware_200.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )
        foldredtowel_result_lai_500 = lai_hardware_500.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )

        foldredtowel_result_savi = savi_hardware.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )
        foldredtowel_result_step_50 = step_hardware_50.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )
        foldredtowel_result_step_200 = step_hardware_200.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )
        foldredtowel_result_step_500 = step_hardware_500.run_on_sequence(
            sequences_fold_red_towel[i, :, 0],
            sequences_fold_red_towel[i, :, 1],
        )

        results_FRT[i, 0] = foldredtowel_result_lai_50.info["Time"]
        results_FRT[i, 1] = foldredtowel_result_lai_200.info["Time"]
        results_FRT[i, 2] = foldredtowel_result_lai_500.info["Time"]
        results_FRT[i, 3] = foldredtowel_result_step_50.info["Time"]
        results_FRT[i, 4] = foldredtowel_result_step_200.info["Time"]
        results_FRT[i, 5] = foldredtowel_result_step_500.info["Time"]
        results_FRT[i, 6] = foldredtowel_result_savi.info[
            "result_for_alternative"
        ].info["Time"]

        ##############################
        ### Result 2: CleanUpSpill ###
        ##############################
        cleanupspill_result_lai_50 = lai_hardware_50.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )
        cleanupspill_result_lai_200 = lai_hardware_200.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )
        cleanupspill_result_lai_500 = lai_hardware_500.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )

        cleanupspill_result_savi = savi_hardware.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )
        cleanupspill_result_step_50 = step_hardware_50.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )
        cleanupspill_result_step_200 = step_hardware_200.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )
        cleanupspill_result_step_500 = step_hardware_500.run_on_sequence(
            sequences_clean_up_spill[i, :, 0],
            sequences_clean_up_spill[i, :, 1],
        )

        results_CUS[i, 0] = cleanupspill_result_lai_50.info["Time"]
        results_CUS[i, 1] = cleanupspill_result_lai_200.info["Time"]
        results_CUS[i, 2] = cleanupspill_result_lai_500.info["Time"]
        results_CUS[i, 3] = cleanupspill_result_step_50.info["Time"]
        results_CUS[i, 4] = cleanupspill_result_step_200.info["Time"]
        results_CUS[i, 5] = cleanupspill_result_step_500.info["Time"]
        results_CUS[i, 6] = cleanupspill_result_savi.info[
            "result_for_alternative"
        ].info["Time"]

        ##############################
        ### Result 3: SpoonOnTowel ###
        ##############################
        spoonontowel_result_lai_500 = lai_simulation_500.run_on_sequence(
            sequences_spoon_on_towel[i, :, 0],
            sequences_spoon_on_towel[i, :, 1],
        )
        spoonontowel_result_savi = savi_simulation.run_on_sequence(
            sequences_spoon_on_towel[i, :, 0],
            sequences_spoon_on_towel[i, :, 1],
        )
        spoonontowel_result_step_500 = step_simulation_500.run_on_sequence(
            sequences_spoon_on_towel[i, :, 0],
            sequences_spoon_on_towel[i, :, 1],
        )

        results_SOT[i, 0] = spoonontowel_result_lai_500.info["Time"]
        results_SOT[i, 1] = spoonontowel_result_step_500.info["Time"]
        results_SOT[i, 2] = spoonontowel_result_savi.info[
            "result_for_alternative"
        ].info["Time"]

        ##################################
        ### Result 4: EggplantInBasket ###
        ##################################
        eggplantinbasket_result_lai_500 = lai_simulation_500.run_on_sequence(
            sequences_eggplant_in_basket[i, :, 0],
            sequences_eggplant_in_basket[i, :, 1],
        )
        eggplantinbasket_result_savi = savi_simulation.run_on_sequence(
            sequences_eggplant_in_basket[i, :, 0],
            sequences_eggplant_in_basket[i, :, 1],
        )
        eggplantinbasket_result_step_500 = step_simulation_500.run_on_sequence(
            sequences_eggplant_in_basket[i, :, 0],
            sequences_eggplant_in_basket[i, :, 1],
        )

        results_EIB[i, 0] = eggplantinbasket_result_lai_500.info["Time"]
        results_EIB[i, 1] = eggplantinbasket_result_step_500.info["Time"]
        results_EIB[i, 2] = eggplantinbasket_result_savi.info[
            "result_for_alternative"
        ].info["Time"]

        ###########################
        ### Result 5: StackCube ###
        ###########################
        stackcube_result_lai_500 = lai_simulation_500.run_on_sequence(
            sequences_stack_cube[i, :, 0],
            sequences_stack_cube[i, :, 1],
        )
        stackcube_result_savi = savi_simulation.run_on_sequence(
            sequences_stack_cube[i, :, 0],
            sequences_stack_cube[i, :, 1],
        )
        stackcube_result_step_500 = step_simulation_500.run_on_sequence(
            sequences_stack_cube[i, :, 0],
            sequences_stack_cube[i, :, 1],
        )

        results_SC[i, 0] = stackcube_result_lai_500.info["Time"]
        results_SC[i, 1] = stackcube_result_step_500.info["Time"]
        results_SC[i, 2] = stackcube_result_savi.info["result_for_alternative"].info[
            "Time"
        ]

    n_runs_correction = np.sqrt(float(n_runs))
    # Synthesize mean and standard deviation
    print()
    print("FOLD RED TOWEL (AVERAGE): ")
    print()
    print(
        "Lai-50  time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 0]),
        ", ",
        np.std(results_FRT[:, 0]) / n_runs_correction,
        ")",
    )
    print(
        "Lai-200  time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 1]),
        ", ",
        np.std(results_FRT[:, 1]) / n_runs_correction,
        ")",
    )
    print(
        "Lai-500  time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 2]),
        ", ",
        np.std(results_FRT[:, 2]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-50 time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 3]),
        ", ",
        np.std(results_FRT[:, 3]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-200 time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 4]),
        ", ",
        np.std(results_FRT[:, 4]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-500 time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 5]),
        ", ",
        np.std(results_FRT[:, 5]) / n_runs_correction,
        ")",
    )
    print(
        "SAVI time-to-decision (mu, std): (",
        np.mean(results_FRT[:, 6]),
        ", ",
        np.std(results_FRT[:, 6]) / n_runs_correction,
        ")",
    )

    print()
    print("CLEAN UP SPILL (AVERAGE): ")
    print()
    print(
        "Lai-50  time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 0]),
        ", ",
        np.std(results_CUS[:, 0]) / n_runs_correction,
        ")",
    )
    print(
        "Lai-200  time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 1]),
        ", ",
        np.std(results_CUS[:, 1]) / n_runs_correction,
        ")",
    )
    print(
        "Lai-500  time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 2]),
        ", ",
        np.std(results_CUS[:, 2]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-50 time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 3]),
        ", ",
        np.std(results_CUS[:, 3]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-200 time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 4]),
        ", ",
        np.std(results_CUS[:, 4]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-500 time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 5]),
        ", ",
        np.std(results_CUS[:, 5]) / n_runs_correction,
        ")",
    )
    print(
        "SAVI time-to-decision (mu, std): (",
        np.mean(results_CUS[:, 6]),
        ", ",
        np.std(results_CUS[:, 6]) / n_runs_correction,
        ")",
    )

    print()
    print("SPOON ON TOWEL (AVERAGE): ")
    print()
    print(
        "Lai-500  time-to-decision (mu, std): (",
        np.mean(results_SOT[:, 0]),
        ", ",
        np.std(results_SOT[:, 0]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-500 time-to-decision (mu, std): (",
        np.mean(results_SOT[:, 1]),
        ", ",
        np.std(results_SOT[:, 1]) / n_runs_correction,
        ")",
    )
    print(
        "SAVI time-to-decision (mu, std): (",
        np.mean(results_SOT[:, 2]),
        ", ",
        np.std(results_SOT[:, 2]) / n_runs_correction,
        ")",
    )

    print()
    print("EGGPLANT IN BASKET (AVERAGE): ")
    print()
    print(
        "Lai-500  time-to-decision (mu, std): (",
        np.mean(results_EIB[:, 0]),
        ", ",
        np.std(results_EIB[:, 0]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-500 time-to-decision (mu, std): (",
        np.mean(results_EIB[:, 1]),
        ", ",
        np.std(results_EIB[:, 1]) / n_runs_correction,
        ")",
    )
    print(
        "SAVI time-to-decision (mu, std): (",
        np.mean(results_EIB[:, 2]),
        ", ",
        np.std(results_EIB[:, 2]) / n_runs_correction,
        ")",
    )

    print()
    print("STACK CUBE (AVERAGE): ")
    print()
    print(
        "Lai-500  time-to-decision (mu, std): (",
        np.mean(results_SC[:, 0]),
        ", ",
        np.std(results_SC[:, 0]) / n_runs_correction,
        ")",
    )
    print(
        "STEP-500 time-to-decision (mu, std): (",
        np.mean(results_SC[:, 1]),
        ", ",
        np.std(results_SC[:, 1]) / n_runs_correction,
        ")",
    )
    print(
        "SAVI time-to-decision (mu, std): (",
        np.mean(results_SC[:, 2]),
        ", ",
        np.std(results_SC[:, 2]) / n_runs_correction,
        ")",
    )
