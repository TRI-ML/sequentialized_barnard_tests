"""Method to run policy synthesis for STEP procedure.

Policies are stored in sequentialized_barnard_tests/policies.

Example Default Usage (all equivalent, using default params):

    (1) python scripts/synthesize_step_policy.py
    (2) python scripts/synthesize_step_policy.py -n 200 -a 0.05
    (3) python scripts/synthesize_step_policy.py --n_max 200 --alpha 0.05 --n_points 89

Example Non-Default Parameter Usage:

    python scripts/synthesize_step_policy.py -n 400
"""

import argparse
import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.ndimage import convolve
from scipy.optimize import linprog
from tqdm import tqdm

from sequentialized_barnard_tests import StepTest
from sequentialized_barnard_tests.base import Hypothesis
from sequentialized_barnard_tests.utils.utils_step import (
    compress_policy_simple,
    reconstruct_rejection_region,
    run_single_state_assignment,
    synthesize_risk_budget,
)


def verify_type1_error_control(
    n_max: int,
    alpha: float,
    n_points: int,
    lambda_value: float,
    major_axis_length: float,
    risk_budget_shape_parameter: float = 0.0,
    use_p_norm: bool = False,
    custom_differential_risk_budget: ArrayLike = None,
    dead_time: int = None,
    verbose: bool = False,
    mirrored: bool = True,
    alternative: Hypothesis = Hypothesis.P0LessThanP1,
):
    try:
        assert n_max >= 1
        assert 0.0 < alpha and alpha < 1.0
        assert n_points >= 21
        assert lambda_value > 0.0
        assert major_axis_length > 0.0
    except:
        raise ValueError(
            "Invalid argument in set (n_max, alpha, n_points, lambda_value, major_axis_length)"
        )

    ##########
    # HANDLE INITIAL FORMATTING
    # ASSIGN RISK BUDGET PER USER REQUIREMENTS
    ##########

    # Handle dead_time parameter
    if dead_time is None:
        dead_time = np.maximum(4, int(np.floor(np.log2(n_max)) + 1))
    else:
        try:
            assert dead_time >= 1
        except:
            raise ValueError("dead_time must be >= 1")

    if verbose:
        print("Dead time: ", dead_time)

    diff_mass_removal_array, cumulative_mass_removal_array = synthesize_risk_budget(
        custom_differential_risk_budget,
        risk_budget_shape_parameter,
        use_p_norm,
        dead_time,
        n_max,
        alpha,
        verbose,
    )
    # Error handling: confirm that budgets end in the right place
    try:
        assert np.isclose(alpha, cumulative_mass_removal_array[-1])
        assert np.isclose(alpha, np.sum(diff_mass_removal_array))
    except:
        raise ValueError(
            "Inconsistent cumulative and differential mass removal arrays; will lead to unpredictable optimization behavior!"
        )

    ##########
    # HANDLE Kernels, storage matrices, and encoding matrices
    # HANDLE capacity to compress the policy as we go
    ##########
    # TODO: more principled setup than the empirical shape parameters for quadratic_score
    # Compute extremal 0 < p_min, p_max < 1 that contain risk of positive delta
    p_max = np.exp(np.log(1.0 - alpha - 1e-5) / n_max)
    p_min = 1.0 - p_max

    # Assign control points corresponding to worst-case null hypotheses that span [p_min, p_max]
    POINTS_ARRAY = p_min + (p_max - p_min) * np.linspace(0, n_points - 1, n_points) / (
        n_points - 1
    )

    # Error handling -- incorrect span of control points
    try:
        assert np.isclose(POINTS_ARRAY[-1], p_max)
        assert np.isclose(POINTS_ARRAY[0], p_min)
    except:
        raise ValueError(
            "Error in assigning control points of worst-case null hypotheses; extremal values do not match [p_min, p_max]"
        )

    # Assign dynamics kernels on the basis of the worst-case point null hypotheses (control points).
    KERNELS = np.zeros((3, 3, n_points))
    for k in range(n_points):
        null_prob = float(POINTS_ARRAY[k])
        KERNELS[:, :, k] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, (1.0 - null_prob) ** 2, null_prob * (1.0 - null_prob)],
                [0.0, null_prob * (1.0 - null_prob), null_prob**2],
            ]
        )

    # Markovian state transition matrices (O(n_max^2))
    STATE_DIST_PRE = np.zeros((n_max + 1, n_max + 1, n_points))
    STATE_DIST_POST = np.zeros((n_max + 1, n_max + 1, n_points))

    # Initialize state distribution
    STATE_DIST_PRE[0, 0, :] = copy.deepcopy(np.ones(n_points))

    risk_accumulation = np.zeros((n_max + 1, n_points))

    STEP_test = StepTest(
        alternative, n_max, alpha, risk_budget_shape_parameter, use_p_norm
    )
    STEP_test.load_existing_policy()
    if STEP_test.policy is None:
        raise ValueError(
            "Unable to find a policy with these parameters. Please double check or run appropriate policy synthesis. "
        )

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

    for t in tqdm(range(1, n_max + 1)):
        # Delete things we don't need
        try:
            del policy_array
            del decision_array_t
        except:
            pass

        # Propagate the dynamics
        critical_limit = int(np.minimum(n_max + 1, t + 1))

        # Propagate dynamics under each control point (worst-case null)
        for k in range(n_points):
            convolve(
                STATE_DIST_PRE[:critical_limit, :critical_limit, k],
                KERNELS[:, :, k],
                output=STATE_DIST_POST[:critical_limit, :critical_limit, k],
                mode="constant",
                cval=0.0,
            )

        if t > dead_time:
            # Construct policy array
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

            # Delete probability mass in the array
            # for k in range(n_points):
            #     for i in range(critical_limit):
            #         for j in range(critical_limit):
            #             candidate_mass_removal = STATE_DIST_POST[i, j, k]
            #             if policy_array[i, j] > 0.0:
            #                 # Remove mass and accumulate risk
            #                 risk_accumulation[t, k] += candidate_mass_removal * float(
            #                     policy_array[i, j]
            #                 )
            #                 STATE_DIST_POST[i, j, k] *= 1.0 - float(policy_array[i, j])
            #             else:
            #                 # Remove mass, but it is not risk
            #                 STATE_DIST_POST[i, j, k] *= 1.0 - np.abs(
            #                     float(policy_array[i, j])
            #                 )
            for i in range(critical_limit):
                for j in range(critical_limit):
                    candidate_mass_removal_vector = copy.deepcopy(
                        STATE_DIST_POST[i, j, :]
                    )
                    if policy_array[i, j] > 0.0:
                        # Remove mass and accumulate risk
                        risk_accumulation[t, :] += copy.deepcopy(
                            candidate_mass_removal_vector * float(policy_array[i, j])
                        )
                        STATE_DIST_POST[i, j, :] -= copy.deepcopy(
                            candidate_mass_removal_vector * float(policy_array[i, j])
                        )
                    else:
                        # Remove mass, but it is not risk
                        STATE_DIST_POST[i, j, :] -= copy.deepcopy(
                            candidate_mass_removal_vector
                            * np.abs(float(policy_array[i, j]))
                        )

        else:
            pass

        # Copy post to pre in advance of the next step of the loop
        STATE_DIST_PRE = copy.deepcopy(STATE_DIST_POST)

    return risk_accumulation


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
        "-np",
        "--n_points",
        type=int,
        default=89,
        help=(
            "Number of control points used to approximate worst-case Type-1 Error. First "
            "of three numerically important STEP parameters. More n_points adds precision "
            "at the expense of additional computation. In practice, ~50 is often sufficient. "
            "Defaults to 89."
        ),
    )
    parser.add_argument(
        "-l",
        "--lambda_value",
        type=float,
        default=2.1,
        help=(
            "First of two approximate shape parameters which specify a prior over the order "
            "in which states are appended to the optimization scheme. Can be numerically important "
            "in practice. "
            "Defaults to 2.1."
        ),
    )
    parser.add_argument(
        "-m",
        "--major_axis_length",
        type=float,
        default=1.4,
        help=(
            "Second of two approximate shape parameters which specify a prior over the order "
            "in which states are appended to the optimization scheme. Can be numerically important "
            "in practice. "
            "Defaults to 1.4."
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

    args = parser.parse_args()

    if args.n_max == 100:
        lambda_value = 2.1
        major_axis_length = 1.4
    elif args.n_max == 200:
        lambda_value = 2.1
        major_axis_length = 1.15
    elif args.n_max == 300:
        lambda_value = 2.1
        major_axis_length = 1.4
    elif args.n_max == 400:
        lambda_value = 2.1
        major_axis_length = 1.4
    elif args.n_max == 500:
        lambda_value = 2.2
        major_axis_length = 1.35
    else:
        lambda_value = args.lambda_value
        major_axis_length = args.major_axis_length

    RISK_ACCUMULATION = verify_type1_error_control(
        args.n_max,
        args.alpha,
        args.n_points,
        lambda_value,
        major_axis_length,
        args.log_p_norm,
        args.use_p_norm,
    )

    final_risk = np.cumsum(RISK_ACCUMULATION, axis=0)
    fig, ax = plt.subplots()
    for k in range(args.n_points):
        ax.plot(final_risk[:, k])

    policy_id_str = f"n_max_{args.n_max}_alpha_{args.alpha}_shape_parameter_{args.log_p_norm}_pnorm_{args.use_p_norm}/"
    policy_img_save_str = "scripts/im/" + policy_id_str
    if not os.path.isdir(policy_img_save_str):
        os.makedirs(policy_img_save_str)

    fig.savefig(policy_img_save_str + "Sanity_Check_v0.png", dpi=450)

    fig, ax = plt.subplots()
    ax.plot(final_risk[-1, :])
    fig.savefig(policy_img_save_str + "Sanity_Check_v1.png", dpi=450)
