"""Script to print out all (most) paper results from camera-ready version
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


def visualize_step_policy_and_trajectory(
    n_max: int,
    alpha: float,
    time_of_decision: int,
    input_trajectory: ArrayLike,
    task_string: str,
    risk_budget_shape_parameter: float = 0.0,
    use_p_norm: bool = False,
    mirrored: bool = True,
    alternative: Hypothesis = Hypothesis.P0LessThanP1,
):
    """Modified version of visualize_step_policy.py which also allows for visualization of the data
       trajectory (green star). This allows us to verify that the policy's actions are consistent with
       the policy visualization (e.g., that there is not an offset in the visualization vis-a-vis the data).

    Args:
        n_max (int): _description_
        alpha (float): _description_
        time_of_decision (int): _description_
        input_trajectory (ArrayLike): _description_
        task_string (str): _description_
        risk_budget_shape_parameter (float, optional): _description_. Defaults to 0.0.
        use_p_norm (bool, optional): _description_. Defaults to False.
        mirrored (bool, optional): _description_. Defaults to True.
        alternative (Hypothesis, optional): _description_. Defaults to Hypothesis.P0LessThanP1.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    STEP_test = MirroredStepTest(
        alternative, n_max, alpha, risk_budget_shape_parameter, use_p_norm
    )
    STEP_test.load_existing_policy()

    if STEP_test.policy is None:
        raise ValueError(
            "Unable to find a policy with these parameters. Please double check or run appropriate policy synthesis. "
        )

    # Set up and create the directory in which to save the appropriate images.
    policy_id_str = f"n_max_{n_max}_alpha_{alpha}_shape_parameter_{risk_budget_shape_parameter}_pnorm_{use_p_norm}/"
    media_save_path = "scripts/im/policies/" + policy_id_str + "/" + task_string + "/"

    if not os.path.isdir(media_save_path):
        os.makedirs(media_save_path)

    # Extract STEP test policy and query in order to construct images
    policy_to_visualize = copy.deepcopy(STEP_test.policy)

    try:
        assert len(policy_to_visualize) == n_max + 1
    except:
        print(
            f"Issue with policy consistency; should be length {n_max + 1}, but is length {len(policy_to_visualize)}"
        )
        raise ValueError(
            "policy appears to be of incorrect length. Please verify the synthesis procedure."
        )

    fig2, ax2 = plt.subplots(figsize=(10, 10))

    # Construct running state trajectory
    results = (
        np.cumsum(input_trajectory, axis=0)
        / np.arange(1, input_trajectory.shape[0] + 1).reshape(-1, 1)
    )[: time_of_decision + 1, :]

    # Iterate through loop to generate policy_array and associated images
    for t in tqdm(range(time_of_decision + 1)):
        try:
            del policy_array
            del decision_array_t
        except:
            pass

        policy_array = np.zeros((t + 1, t + 1))
        if t >= 1:
            decision_array_t = policy_to_visualize[t]
        else:
            decision_array_t = [0]

        for i in range(t + 1):
            for j in range(i, t + 1):
                x_absolute = min(i, j)
                y_absolute = max(i, j)

                if y_absolute - x_absolute > 0:
                    decision_array = decision_array_t[x_absolute]
                    # Number of non-zero / non-unity policy bins at this x and t
                    L = decision_array.shape[0] - 1

                    # Highest value of y for which we CONTINUE [i.e., policy = 0]
                    critical_zero_y = int(decision_array[0])

                    if mirrored:
                        # Find the decision and assign it to [x_abs, y_abs], and assign negation to [y_abs, x_abs]
                        if y_absolute <= critical_zero_y:
                            pass
                        elif y_absolute > (critical_zero_y + L):
                            policy_array[x_absolute, y_absolute] = 1.0
                            policy_array[y_absolute, x_absolute] = -1.0
                        else:
                            prob_stop = decision_array[y_absolute - critical_zero_y]
                            policy_array[x_absolute, y_absolute] = prob_stop
                            policy_array[y_absolute, x_absolute] = -prob_stop

                    elif alternative == Hypothesis.P0LessThanP1:
                        # Find the decision and assign it to [x_abs, y_abs], and assign negation to [y_abs, x_abs]
                        if y_absolute <= critical_zero_y:
                            pass
                        elif y_absolute > (critical_zero_y + L):
                            policy_array[x_absolute, y_absolute] = 1.0
                        else:
                            prob_stop = decision_array[y_absolute - critical_zero_y]
                            policy_array[x_absolute, y_absolute] = prob_stop

                    else:
                        # Find the decision and assign it to [x_abs, y_abs], and assign negation to [y_abs, x_abs]
                        if y_absolute <= critical_zero_y:
                            pass
                        elif y_absolute > (critical_zero_y + L):
                            policy_array[y_absolute, x_absolute] = -1.0
                        else:
                            prob_stop = decision_array[y_absolute - critical_zero_y]
                            policy_array[y_absolute, x_absolute] = -prob_stop

        # Save off policy array as an image
        ax2.cla()
        # plt.cla()

        # ax.imshow(np.transpose(SIGN_ARRAY), cmap='RdYlGn', origin='lower')
        ax2.pcolormesh(
            np.arange(t + 2) / (t + 1),
            np.arange(t + 2) / (t + 1),
            np.transpose(policy_array),
            cmap="RdYlBu",  # "RdYlGn",
            vmin=-1.2,
            vmax=1.2,
        )
        ax2.plot([0, 1], [0, 1], "k--", linewidth=5)
        ax2.plot(results[t, 0], results[t, 1], "g*", markersize=5)
        ax2.set_xlabel("Baseline Performance", fontsize=24)
        ax2.set_ylabel("Test Policy Performance", fontsize=24)
        ax2.tick_params(labelsize=20)
        ax2.text(0.05, 0.95, f"n = {t}", color="#FFFFFF", fontsize=24, weight="heavy")
        ax2.set_aspect("equal")
        ax2.grid(True)
        fig2.savefig(media_save_path + f"{t:03d}.png", dpi=450)

    return 1


if __name__ == "__main__":
    paper_data_path = str(
        Path(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../tests/eval_data/",
            )
        ).resolve()
    )
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

    # Load tests of varying length
    savi_hardware = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05)
    savi_simulation = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=0.01)

    step_random_seed = 42

    lai_hardware_50 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=50, alpha=0.05
    )
    lai_hardware_200 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=200, alpha=0.05
    )
    lai_hardware_500 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=500, alpha=0.05
    )
    lai_simulation_500 = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1, n_max=500, alpha=0.01
    )

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
    step_simulation_500_error_catcher = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=500,
        alpha=0.05,
        random_seed=step_random_seed,
    )
    step_simulation_500_error_2 = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=500,
        alpha=0.025,
        random_seed=step_random_seed,
    )

    # Run without index permutation
    permutation_idx_hardware = np.arange(50)
    permutation_idx_simulation = np.arange(500)

    # Run tests on each data stream where appropriate

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
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
    )
    cleanupspill_result_lai_200 = lai_hardware_200.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
    )
    cleanupspill_result_lai_500 = lai_hardware_500.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
    )

    cleanupspill_result_savi = savi_hardware.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
    )

    cleanupspill_result_step_50 = step_hardware_50.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
    )
    cleanupspill_result_step_200 = step_hardware_200.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
    )
    cleanupspill_result_step_500 = step_hardware_500.run_on_sequence(
        eval_clean_up_spill[permutation_idx_hardware, 1],
        eval_clean_up_spill[permutation_idx_hardware, 0],
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
        eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
        eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
    )
    spoonontowel_result_savi = savi_simulation.run_on_sequence(
        eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
        eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
    )
    spoonontowel_result_step_500 = step_simulation_500.run_on_sequence(
        eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
        eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
    )
    spoonontowel_result_step_500_error_catcher = (
        step_simulation_500_error_catcher.run_on_sequence(
            eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
            eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
        )
    )
    spoonontowel_result_step_500_error_catcher2 = (
        step_simulation_500_error_2.run_on_sequence(
            eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
            eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
        )
    )

    print()
    print("SPOON ON TOWEL: ")
    print()
    print("Lai-500  time-to-decision: ", spoonontowel_result_lai_500.info["Time"])
    print("STEP-500 time-to-decision: ", spoonontowel_result_step_500.info["Time"])
    print(
        "STEP-500 time-to-decision (error catch): ",
        spoonontowel_result_step_500_error_catcher.info["Time"],
    )
    print(
        "STEP-500 time-to-decision (error catch 2): ",
        spoonontowel_result_step_500_error_catcher2.info["Time"],
    )
    print(
        "SAVI     time-to-decision: ",
        spoonontowel_result_savi.info["result_for_alternative"].info["Time"],
    )

    ##################################
    ### Result 4: EggplantInBasket ###
    ##################################
    eggplantinbasket_result_lai_500 = lai_simulation_500.run_on_sequence(
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
    )
    eggplantinbasket_result_savi = savi_simulation.run_on_sequence(
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
    )
    eggplantinbasket_result_step_500 = step_simulation_500.run_on_sequence(
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
        eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
    )
    eggplantinbasket_result_step_500_error_catcher = (
        step_simulation_500_error_catcher.run_on_sequence(
            eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
            eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
        )
    )
    eggplantinbasket_result_step_500_error_2 = (
        step_simulation_500_error_2.run_on_sequence(
            eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
            eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
        )
    )

    print()
    print("EGGPLANT IN BASKET: ")
    print()
    print("Lai-500  time-to-decision: ", eggplantinbasket_result_lai_500.info["Time"])
    print("STEP-500 time-to-decision: ", eggplantinbasket_result_step_500.info["Time"])
    print(
        "STEP-500 time-to-decision (error catcher): ",
        eggplantinbasket_result_step_500_error_catcher.info["Time"],
    )
    print(
        "STEP-500 time-to-decision (error catcher 2): ",
        eggplantinbasket_result_step_500_error_2.info["Time"],
    )

    print(
        "SAVI     time-to-decision: ",
        eggplantinbasket_result_savi.info["result_for_alternative"].info["Time"],
    )

    # _ = visualize_step_policy_and_trajectory(
    #     500,
    #     0.05,
    #     eggplantinbasket_result_step_500.info["Time"] + 1,
    #     np.concatenate(
    #         (
    #             eval_sim_eggplant_in_basket[:, 1].reshape(-1, 1),
    #             eval_sim_eggplant_in_basket[:, 0].reshape(-1, 1),
    #         ),
    #         axis=1,
    #     ),
    #     task_string="EggplantInBasket",
    # )

    # _ = visualize_step_policy_and_trajectory(
    #     500,
    #     0.01,
    #     eggplantinbasket_result_step_500.info["Time"] + 1,
    #     np.concatenate(
    #         (
    #             eval_sim_eggplant_in_basket[:, 1].reshape(-1, 1),
    #             eval_sim_eggplant_in_basket[:, 0].reshape(-1, 1),
    #         ),
    #         axis=1,
    #     ),
    #     task_string="EggplantInBasket",
    # )

    # _ = visualize_step_policy_and_trajectory(
    #     500,
    #     0.025,
    #     eggplantinbasket_result_step_500.info["Time"] + 1,
    #     np.concatenate(
    #         (
    #             eval_sim_eggplant_in_basket[:, 1].reshape(-1, 1),
    #             eval_sim_eggplant_in_basket[:, 0].reshape(-1, 1),
    #         ),
    #         axis=1,
    #     ),
    #     task_string="EggplantInBasket",
    # )

    ###########################
    ### Result 5: StackCube ###
    ###########################
    stackcube_result_lai_500 = lai_simulation_500.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 1],
        eval_sim_stack_cube[permutation_idx_simulation, 0],
    )
    stackcube_result_savi = savi_simulation.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 1],
        eval_sim_stack_cube[permutation_idx_simulation, 0],
    )
    stackcube_result_step_500 = step_simulation_500.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 1],
        eval_sim_stack_cube[permutation_idx_simulation, 0],
    )
    stackcube_result_step_500_error_catcher = (
        step_simulation_500_error_catcher.run_on_sequence(
            eval_sim_stack_cube[permutation_idx_simulation, 1],
            eval_sim_stack_cube[permutation_idx_simulation, 0],
        )
    )
    stackcube_result_step_500_error_2 = step_simulation_500_error_2.run_on_sequence(
        eval_sim_stack_cube[permutation_idx_simulation, 1],
        eval_sim_stack_cube[permutation_idx_simulation, 0],
    )

    print()
    print("STACK CUBE: ")
    print()
    print("Lai-500  time-to-decision: ", stackcube_result_lai_500.info["Time"])
    print("STEP-500 time-to-decision: ", stackcube_result_step_500.info["Time"])
    print(
        "STEP-500 time-to-decision (error catcher): ",
        stackcube_result_step_500_error_catcher.info["Time"],
    )
    print(
        "STEP-500 time-to-decision (error catcher 2): ",
        stackcube_result_step_500_error_2.info["Time"],
    )
    print(
        "SAVI     time-to-decision: ",
        stackcube_result_savi.info["result_for_alternative"].info["Time"],
    )

    # _ = visualize_step_policy_and_trajectory(
    #     500,
    #     0.05,
    #     stackcube_result_step_500.info["Time"] + 1,
    #     np.concatenate(
    #         (
    #             eval_sim_stack_cube[:, 1].reshape(-1, 1),
    #             eval_sim_stack_cube[:, 0].reshape(-1, 1),
    #         ),
    #         axis=1,
    #     ),
    #     task_string="StackCube",
    # )

    # _ = visualize_step_policy_and_trajectory(
    #     500,
    #     0.01,
    #     stackcube_result_step_500.info["Time"] + 1,
    #     np.concatenate(
    #         (
    #             eval_sim_stack_cube[:, 1].reshape(-1, 1),
    #             eval_sim_stack_cube[:, 0].reshape(-1, 1),
    #         ),
    #         axis=1,
    #     ),
    #     task_string="StackCube",
    # )

    # _ = visualize_step_policy_and_trajectory(
    #     500,
    #     0.025,
    #     stackcube_result_step_500.info["Time"] + 1,
    #     np.concatenate(
    #         (
    #             eval_sim_stack_cube[:, 1].reshape(-1, 1),
    #             eval_sim_stack_cube[:, 0].reshape(-1, 1),
    #         ),
    #         axis=1,
    #     ),
    #     task_string="StackCube",
    # )

    run_multitrial_average = False

    if run_multitrial_average:
        seed_int = 42
        # rng = np.random.default_rng(seed=seed_int)
        n_runs = int(400)
        results_FRT = np.zeros((n_runs, 5))
        results_CUS = np.zeros((n_runs, 5))
        results_SOT = np.zeros((n_runs, 3))
        results_EIB = np.zeros((n_runs, 3))
        results_SC = np.zeros((n_runs, 3))

        for i in tqdm(range(n_runs)):
            rng = np.random.default_rng()
            permutation_idx_hardware = rng.choice(50, 50, replace=False)
            permutation_idx_simulation = rng.choice(500, 500, replace=False)

            ##############################
            ### Result 1: FoldRedTowel ###
            ##############################
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

            foldredtowel_result_step_200 = step_hardware_200.run_on_sequence(
                eval_fold_red_towel[permutation_idx_hardware, 0],
                eval_fold_red_towel[permutation_idx_hardware, 1],
            )
            foldredtowel_result_step_500 = step_hardware_500.run_on_sequence(
                eval_fold_red_towel[permutation_idx_hardware, 0],
                eval_fold_red_towel[permutation_idx_hardware, 1],
            )

            results_FRT[i, 0] = foldredtowel_result_lai_200.info["Time"]
            results_FRT[i, 1] = foldredtowel_result_lai_500.info["Time"]
            results_FRT[i, 2] = foldredtowel_result_step_200.info["Time"]
            results_FRT[i, 3] = foldredtowel_result_step_500.info["Time"]
            results_FRT[i, 4] = foldredtowel_result_savi.info[
                "result_for_alternative"
            ].info["Time"]

            ##############################
            ### Result 2: CleanUpSpill ###
            ##############################
            cleanupspill_result_lai_200 = lai_hardware_200.run_on_sequence(
                eval_clean_up_spill[permutation_idx_hardware, 1],
                eval_clean_up_spill[permutation_idx_hardware, 0],
            )
            cleanupspill_result_lai_500 = lai_hardware_500.run_on_sequence(
                eval_clean_up_spill[permutation_idx_hardware, 1],
                eval_clean_up_spill[permutation_idx_hardware, 0],
            )

            cleanupspill_result_savi = savi_hardware.run_on_sequence(
                eval_clean_up_spill[permutation_idx_hardware, 1],
                eval_clean_up_spill[permutation_idx_hardware, 0],
            )

            cleanupspill_result_step_200 = step_hardware_200.run_on_sequence(
                eval_clean_up_spill[permutation_idx_hardware, 1],
                eval_clean_up_spill[permutation_idx_hardware, 0],
            )
            cleanupspill_result_step_500 = step_hardware_500.run_on_sequence(
                eval_clean_up_spill[permutation_idx_hardware, 1],
                eval_clean_up_spill[permutation_idx_hardware, 0],
            )

            results_CUS[i, 0] = cleanupspill_result_lai_200.info["Time"]
            results_CUS[i, 1] = cleanupspill_result_lai_500.info["Time"]
            results_CUS[i, 2] = cleanupspill_result_step_200.info["Time"]
            results_CUS[i, 3] = cleanupspill_result_step_500.info["Time"]
            results_CUS[i, 4] = cleanupspill_result_savi.info[
                "result_for_alternative"
            ].info["Time"]

            ##############################
            ### Result 3: SpoonOnTowel ###
            ##############################
            spoonontowel_result_lai_500 = lai_simulation_500.run_on_sequence(
                eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
                eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
            )
            spoonontowel_result_savi = savi_simulation.run_on_sequence(
                eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
                eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
            )
            spoonontowel_result_step_500 = step_simulation_500.run_on_sequence(
                eval_sim_spoon_on_towel[permutation_idx_simulation, 1],
                eval_sim_spoon_on_towel[permutation_idx_simulation, 0],
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
                eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
                eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
            )
            eggplantinbasket_result_savi = savi_simulation.run_on_sequence(
                eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
                eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
            )
            eggplantinbasket_result_step_500 = step_simulation_500.run_on_sequence(
                eval_sim_eggplant_in_basket[permutation_idx_simulation, 1],
                eval_sim_eggplant_in_basket[permutation_idx_simulation, 0],
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
                eval_sim_stack_cube[permutation_idx_simulation, 1],
                eval_sim_stack_cube[permutation_idx_simulation, 0],
            )
            stackcube_result_savi = savi_simulation.run_on_sequence(
                eval_sim_stack_cube[permutation_idx_simulation, 1],
                eval_sim_stack_cube[permutation_idx_simulation, 0],
            )
            stackcube_result_step_500 = step_simulation_500.run_on_sequence(
                eval_sim_stack_cube[permutation_idx_simulation, 1],
                eval_sim_stack_cube[permutation_idx_simulation, 0],
            )

            results_SC[i, 0] = stackcube_result_lai_500.info["Time"]
            results_SC[i, 1] = stackcube_result_step_500.info["Time"]
            results_SC[i, 2] = stackcube_result_savi.info[
                "result_for_alternative"
            ].info["Time"]

        n_runs_correction = np.sqrt(float(n_runs))
        # Synthesize mean and standard deviation
        print()
        print("FOLD RED TOWEL (AVERAGE): ")
        print()
        print(
            "Lai-200  time-to-decision (mu, std): (",
            np.mean(results_FRT[:, 0]),
            ", ",
            np.std(results_FRT[:, 0]) / n_runs_correction,
            ")",
        )
        print(
            "Lai-500  time-to-decision (mu, std): (",
            np.mean(results_FRT[:, 1]),
            ", ",
            np.std(results_FRT[:, 1]) / n_runs_correction,
            ")",
        )
        print(
            "STEP-200 time-to-decision (mu, std): (",
            np.mean(results_FRT[:, 2]),
            ", ",
            np.std(results_FRT[:, 2]) / n_runs_correction,
            ")",
        )
        print(
            "STEP-500 time-to-decision (mu, std): (",
            np.mean(results_FRT[:, 3]),
            ", ",
            np.std(results_FRT[:, 3]) / n_runs_correction,
            ")",
        )
        print(
            "SAVI time-to-decision (mu, std): (",
            np.mean(results_FRT[:, 4]),
            ", ",
            np.std(results_FRT[:, 4]) / n_runs_correction,
            ")",
        )

        print()
        print("CLEAN UP SPILL (AVERAGE): ")
        print()
        print(
            "Lai-200  time-to-decision (mu, std): (",
            np.mean(results_CUS[:, 0]),
            ", ",
            np.std(results_CUS[:, 0]) / n_runs_correction,
            ")",
        )
        print(
            "Lai-500  time-to-decision (mu, std): (",
            np.mean(results_CUS[:, 1]),
            ", ",
            np.std(results_CUS[:, 1]) / n_runs_correction,
            ")",
        )
        print(
            "STEP-200 time-to-decision (mu, std): (",
            np.mean(results_CUS[:, 2]),
            ", ",
            np.std(results_CUS[:, 2]) / n_runs_correction,
            ")",
        )
        print(
            "STEP-500 time-to-decision (mu, std): (",
            np.mean(results_CUS[:, 3]),
            ", ",
            np.std(results_CUS[:, 3]) / n_runs_correction,
            ")",
        )
        print(
            "SAVI time-to-decision (mu, std): (",
            np.mean(results_CUS[:, 4]),
            ", ",
            np.std(results_CUS[:, 4]) / n_runs_correction,
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
