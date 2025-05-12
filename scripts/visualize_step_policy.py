"""Method to construct visualizations of the STEP near-optimal decision making policy.

This is primarily a debugging tool, useful for the designer to visually verify that the
policy is incorporating states in a logical / explainable manner.
"""

import argparse
import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from sequentialized_barnard_tests import StepTest
from sequentialized_barnard_tests.base import Decision, Hypothesis


def visualize_step_policy(
    n_max: int,
    alpha: float,
    risk_budget_shape_parameter: float = 0.0,
    use_p_norm: bool = False,
    mirrored: bool = True,
    alternative: Hypothesis = Hypothesis.P0LessThanP1,
):
    """Tool to visualize the compressed STEP policy in an array-like lookup table format (i.e., in decompressed
       form). If there is an associated saved uncompressed policy, then the function will also load the policy
       and store reconstruction error rates to verify the faithful / lossless nature of the reconstruction.

    Args:
        n_max (int): Maximum number of evaluation trials
        alpha (float): Cumulative type-1 error risk limit
        risk_budget_shape_parameter (float, optional): Shape of the risk budget. Defaults to 0.0.
        use_p_norm (bool, optional): Whether to use p_norm or partial zeta function. Defaults to False (i.e., partial zeta function).
        mirrored (bool, optional): Whether to use a mirrored policy. Defaults to True.
        alternative (Hypothesis, optional): The alternative hypothesis under consideration. Defaults to Hypothesis.P0LessThanP1.

    Raises:
        ValueError: Could not find the policy
        ValueError: If the policy has the wrong n_max or was not successfully synthesized.

    Returns:
        bool: Whether a ground-truth uncompressed policy was used for comparison
        ArrayLike: The reconstruction errors against the ground truth uncompressed policy, ELSE 0.
    """
    STEP_test = StepTest(
        alternative, n_max, alpha, risk_budget_shape_parameter, use_p_norm
    )
    STEP_test.load_existing_policy()

    if STEP_test.policy is None:
        raise ValueError(
            "Unable to find a policy with these parameters. Please double check or run appropriate policy synthesis. "
        )

    # Set up and create the directory in which to save the appropriate images.
    policy_id_str = f"n_max_{n_max}_alpha_{alpha}_shape_parameter_{risk_budget_shape_parameter}_pnorm_{use_p_norm}/"

    check_array_base_str = (
        f"sequentialized_barnard_tests/policies/" + policy_id_str + f"array/time_"
    )
    try:
        np.load(check_array_base_str + f"{5}.npy")
        compute_reconstruction_error_flag = True
    except:
        compute_reconstruction_error_flag = False

    if compute_reconstruction_error_flag:
        error_by_timestep = np.zeros(n_max + 1)

    media_save_path = "media/im/policies/" + policy_id_str

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

    # Iterate through loop to generate policy_array and associated images
    for t in tqdm(range(n_max + 1)):
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

        if t >= 1 and compute_reconstruction_error_flag:
            check_policy_array = np.load(check_array_base_str + f"{t}.npy")
            error_by_timestep[t] = np.mean(np.abs(policy_array - check_policy_array))

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
        ax2.set_xlabel("Baseline Performance", fontsize=24)
        ax2.set_ylabel("Test Policy Performance", fontsize=24)
        ax2.tick_params(labelsize=20)
        ax2.text(0.05, 0.95, f"n = {t}", color="#FFFFFF", fontsize=24, weight="heavy")
        ax2.set_aspect("equal")
        ax2.grid(True)
        fig2.savefig(media_save_path + f"{t:03d}.png", dpi=450)

    if compute_reconstruction_error_flag:
        return True, error_by_timestep
    else:
        return False, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script synthesizes a near-optimal STEP policy for a given "
            "{n_max, alpha} combination. The results are saved to a .npy file at "
            "'sequentialized_barnard_tests/policies'. Some parameters of the STEP "
            "policy's synthesis procedure can have important numerical effects "
            "on the resulting efficiency of computation."
        )
    )
    parser.add_argument(
        "-n",
        "--n_max",
        type=int,
        default=200,
        help=(
            "Maximum number of robot policy evals (per policy) in the evaluation procedure. "
            "Defaults to 200."
        ),
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help=(
            "Maximal allowed Type-1 error rate of the statistical testing procedure. "
            "Defaults to 0.05."
        ),
    )
    parser.add_argument(
        "-pz",
        "--log_p_norm",
        type=float,
        default=0.0,
        help=(
            "Rate at which risk is accumulated, reflecting user's belief about underlying "
            "likelihood of different alternatives and nulls being true. If using a p_norm "
            ", this variable is equivalent to log(p). If not using a p_norm, this is the "
            "argument to the zeta function, partial sums of which give the shape of the risk budget."
            "Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "-up",
        "--use_p_norm",
        type=bool,
        default=False,
        help=(
            "Toggle whether to use p_norm or zeta function shape family for the risk budget. "
            "If True, uses p_norm shape; else, uses zeta function shape family. "
            "Defaults to False (zeta function partial sum family)."
        ),
    )
    # TODO: add mirrored, alternative

    args = parser.parse_args()

    policy_id_str = f"n_max_{args.n_max}_alpha_{args.alpha}_shape_parameter_{args.log_p_norm}_pnorm_{args.use_p_norm}/"
    full_load_str = f"sequentialized_barnard_tests/policies/" + policy_id_str
    media_save_path = "media/im/policies/" + policy_id_str
    scripts_save_path = "scripts/im/policies/" + policy_id_str

    if not os.path.isdir(media_save_path):
        os.makedirs(media_save_path)

    if not os.path.isdir(scripts_save_path):
        os.makedirs(scripts_save_path)

    risk_accumulation = np.load(full_load_str + f"risk_accumulation.npy")
    points_array = np.load(full_load_str + f"points_array.npy")

    fig, ax = plt.subplots()
    for i in range(args.n_max + 1):
        if i % 10 == 0:
            ax.plot(points_array, risk_accumulation[i, :])
    ax.set_xlabel("Null Hypothesis (p, p)")
    ax.set_ylabel("Accumulated Risk")
    ax.set_title("Risk Accumulation at 10-step Intervals")

    fig.savefig(scripts_save_path + "risk_accumulation.png")

    compute_error, mean_error_at_each_timestep = visualize_step_policy(
        args.n_max,
        args.alpha,
        args.log_p_norm,
        args.use_p_norm,
    )

    if compute_error:
        fig, ax = plt.subplots()
        ax.plot(mean_error_at_each_timestep)

        fig.savefig(scripts_save_path + "error_in_reconstruction.png")

        print(
            "Error (mu, sigma): (",
            np.mean(mean_error_at_each_timestep),
            ", ",
            np.std(mean_error_at_each_timestep),
            ")",
        )
