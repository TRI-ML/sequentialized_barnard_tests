"""Method to run policy synthesis for STEP procedure.

Policies are stored in sequentialized_barnard_tests/policies.

Example Default Usage (all equivalent, using default params):

    (1) python sequentialized_barnard_tests/scripts/synthesize_step_policy.py
    (2) python sequentialized_barnard_tests/scripts/synthesize_step_policy.py -n 200 -a 0.05
    (3) python sequentialized_barnard_tests/scripts/synthesize_step_policy.py --n_max 200 --alpha 0.05 --n_points 129

Example Non-Default Parameter Usage:

    python sequentialized_barnard_tests/scripts/synthesize_step_policy.py -n 400
"""

import argparse
import copy
import os
import pickle
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import convolve
from scipy.optimize import linprog
from tqdm import tqdm

from sequentialized_barnard_tests.utils import (
    construct_kl_spaced_points_array_via_binary_expansion,
)
from sequentialized_barnard_tests.utils.utils_step import (
    compress_policy_simple,
    reconstruct_rejection_region,
    run_single_state_assignment,
    synthesize_risk_budget,
)


def run_step_policy_synthesis(
    n_max: int,
    alpha: float,
    n_points: int,
    lambda_value: float,
    major_axis_length: float,
    risk_budget_shape_parameter: float = 0.0,
    use_p_norm: bool = False,
    custom_differential_risk_budget: Optional[ArrayLike] = None,
    dead_time: Optional[int] = None,
    save_policy_array: bool = False,
    save_policy_path: Optional[str] = None,
    verbose: bool = False,
):
    """Procedure to synthesize a near-optimal finite-sample test for the policy comparison problem (assuming p1 > p0 is the alternative). This is the foundation for the
    STEP sequential test (which will utilize said policy in making decisions about newly-sampled data).

    Args:
        n_max (int): Maximum number of trials. Integer >= 1.
        alpha (float): Allowed Type-1 error of the sequential test. Float in (0., 1.)
        n_points (int): Number of control points for controlling Type-1 Error. Higher is more precise at expense of greater computational cost. Must be greater than or equal to 21.
        lambda_value (float): Shape prior of the STEP synthesis procedure. Must be greater than 0.0
        major_axis_length (float): Shape prior of the STEP synthesis procedure. Must be greater than 0.0
        risk_budget_shape_parameter (float, optional): Shape parameter of either partial_zeta or p_norm shapes. If partial_zeta, this is the exact exponent (negative numbers allowed). If p_norm, this is the log-exponent, and thus np.exp will be applied downstream. Defaults to 0.0, where the methods are idential (linear risk budget).
        use_p_norm (bool, optional): Toggle between partial_zeta and p_norm shapes. Defaults to False (partial_zeta shape).
        custom_differential_risk_budget (ArrayLike, optional): If given, sets the exact differential risk budget, OVERRIDING p_norm v.s. partial_zeta selection. If not none, all elements must be nonnegative. Defaults to None.
        dead_time (int, optional): Time to wait before attempting any rejection / acceptance. If None, then logarithmic in n_max. Must be positive; defaults to None.
        save_policy_array (bool, optional): Whether to save ground truth uncompressed policy array. Defaults to False.
        save_policy_path (str, optional): Path to save ground truth uncompressed policy array. Defaults to None.
        verbose (bool, optional): Toggle the printing of progress measures and additional information throughout the synthesis procedure. Defaults to False.

    Raises:
        ValueError: If invalid required arguments
        ValueError: If inconsistent specification of whether to save the uncompressed policy
        ValueError: If invalid specified dead_time
        RuntimeError: If cumulative mass removal arrays do not terminate near alpha (making the procedure either loose, if below alpha, or invalid, if above alpha)
        RuntimeError: If control points are not assigned with proper extremal (min and max) limits

    Returns:
        POLICY_LIST_COMPRESSED (ArrayLike): The compressed representation of the accept/reject comparison policy.
        RISK_ACCUMULATION (ArrayLike): The information concerning tightness of the numerical method (useful for validation and visualization).
        POINTS_ARRAY (ArrayLike): The set of control points, primarily for debugging and estimating added Type-1 Error via TV distance and Pinsker's Inequality.
    """
    # Error handling on required parameters
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

    if save_policy_array and save_policy_path is None:
        raise ValueError(
            "Code is set to save the policy array (save_policy_array is True), but no path has been specified (save_policy_path is None)"
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
        raise RuntimeError(
            "Inconsistent cumulative and differential mass removal arrays; will lead to unpredictable optimization behavior!"
        )

    ##########
    # HANDLE Kernels, storage matrices, and encoding matrices
    # HANDLE capacity to compress the policy as we go
    ##########
    # # Compute extremal 0 < p_min, p_max < 1 that contain risk of positive delta
    # p_max = np.exp(np.log(1.0 - alpha - 1e-5) / n_max)
    # p_min = 1.0 - p_max

    # # Assign control points corresponding to worst-case null hypotheses that span [p_min, p_max]
    # POINTS_ARRAY = p_min + (p_max - p_min) * np.linspace(0, n_points - 1, n_points) / (
    #     n_points - 1
    # )

    p_min = 1.0 / float(n_max)
    p_max = float(n_max - 1) / float(n_max)
    POINTS_ARRAY = construct_kl_spaced_points_array_via_binary_expansion(n_max)

    # Error handling -- incorrect span of control points
    try:
        assert np.isclose(POINTS_ARRAY[-1], p_max)
        assert np.isclose(POINTS_ARRAY[0], p_min)
    except:
        raise RuntimeError(
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

    # Compressed-on-the-fly memory representation
    POLICY_LIST_COMPRESSED = []

    # For post-synthesis verification
    RISK_ACCUMULATION = np.zeros((n_max + 1, n_points))

    policy_array_compressed = compress_policy_simple(
        0, np.zeros((n_max + 1, n_max + 1))
    )
    POLICY_LIST_COMPRESSED.append(policy_array_compressed)
    # Begin loop to synthesize the optimal policy
    for t in tqdm(range(1, n_max + 1), desc="STEP Policy Synthesis"):
        # Don't propagate zeros -- waste of time and effort
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

        # The 'brains' of the algorithm
        if t > dead_time:
            base_accumulated_risk = copy.deepcopy(RISK_ACCUMULATION[t - 1, :])
            critical_risk_target = float(cumulative_mass_removal_array[t])
            (
                current_accumulated_risk,
                CANDIDATE_STATE_ENCODING,
                CARRY_OVER_STATE_ENCODING,
            ) = run_single_state_assignment(
                t,
                base_accumulated_risk,
                critical_risk_target,
                POINTS_ARRAY,
                STATE_DIST_POST,
            )

            DISPOSABLE_CANDIDATE_STATE_ENCODING = copy.deepcopy(
                CANDIDATE_STATE_ENCODING
            )

            ##########
            # Construct LP features
            ##########
            n_nonzero_features = int(np.sum(CANDIDATE_STATE_ENCODING))
            nonzero_indices = np.argwhere(CANDIDATE_STATE_ENCODING)

            assert nonzero_indices.shape[0] == n_nonzero_features
            assert nonzero_indices[0, 1] > nonzero_indices[0, 0]

            FEATURES_BASE = np.zeros((n_points, 2 * n_nonzero_features))
            feature_counter = int(0)
            feature_checksum = int(0)
            try:
                del constraint_list
            except:
                pass

            constraint_list = []

            for k in range(n_nonzero_features):
                idx0 = nonzero_indices[k, 0]
                idx1 = nonzero_indices[k, 1]
                if np.isclose(DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1], 0.0):
                    pass
                else:
                    if idx0 + idx1 == t:
                        FEATURES_BASE[:, feature_counter] = copy.deepcopy(
                            STATE_DIST_POST[idx0, idx1, :]
                        )
                        DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
                        feature_counter += 1
                    else:
                        FEATURES_BASE[:, feature_counter] = copy.deepcopy(
                            STATE_DIST_POST[idx0, idx1, :]
                        )
                        FEATURES_BASE[:, feature_counter + 1] = copy.deepcopy(
                            STATE_DIST_POST[t - idx1, t - idx0, :]
                        )
                        constraint_list.append([feature_counter, feature_counter + 1])
                        DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
                        DISPOSABLE_CANDIDATE_STATE_ENCODING[t - idx1, t - idx0] = 0.0

                        feature_counter += 2

                    feature_checksum += 1

            ##########
            # Set up and run optimization
            ##########
            FEATURES = copy.deepcopy(FEATURES_BASE[:, :feature_counter])
            bounds = (0.0, 1.0)

            b_ub = np.maximum(
                critical_risk_target * np.ones(n_points) - base_accumulated_risk - 1e-8,
                np.zeros(n_points),
            )

            max_feature_counter = feature_counter

            A_eq = np.zeros((len(constraint_list), max_feature_counter))
            for i in range(len(constraint_list)):
                A_eq[i, constraint_list[i][0]] = 1
                A_eq[i, constraint_list[i][1]] = -1

            b_eq = np.zeros(len(constraint_list))
            # Handle edge cases in determining when / how to run the optimization
            if feature_counter == 0:
                pass
            else:
                c_vec = -np.ones(feature_counter)
                try:
                    assert len(c_vec.shape) == 1
                except:
                    print(t)
                    print(c_vec.shape)
                linprog_options = {"disp": False}
                try:
                    res_linprog = linprog(
                        c_vec,
                        FEATURES,
                        b_ub,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        bounds=bounds,
                        options=linprog_options,
                    )
                except:
                    print(c_vec.shape)
                    print(feature_counter)
                    print()

                key_weights = res_linprog.x

            # Reconstruct key_weights in original form
            SECOND_DISPOSABLE_CANDIDATE_STATE_ENCODING = copy.deepcopy(
                CANDIDATE_STATE_ENCODING
            )
            SECOND_FEATURES_BASE = np.zeros((n_points, n_nonzero_features))
            key_weights_rectified = np.zeros(n_nonzero_features)
            counter = 0
            second_feature_counter = int(0)
            for k in range(n_nonzero_features):
                idx0 = nonzero_indices[k, 0]
                idx1 = nonzero_indices[k, 1]
                if np.isclose(
                    SECOND_DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1], 0.0
                ):
                    pass
                else:
                    if idx0 + idx1 == t:
                        key_weights_rectified[second_feature_counter] = key_weights[
                            counter
                        ]
                        SECOND_FEATURES_BASE[:, second_feature_counter] = copy.deepcopy(
                            STATE_DIST_POST[idx0, idx1, :]
                        )
                        SECOND_DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
                        counter += 1
                    else:
                        key_weights_rectified[second_feature_counter] = key_weights[
                            counter
                        ]
                        assert np.isclose(
                            key_weights[counter], key_weights[counter + 1]
                        )
                        SECOND_FEATURES_BASE[:, second_feature_counter] = copy.deepcopy(
                            STATE_DIST_POST[idx0, idx1, :]
                        ) + copy.deepcopy(STATE_DIST_POST[t - idx1, t - idx0, :])
                        SECOND_DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
                        SECOND_DISPOSABLE_CANDIDATE_STATE_ENCODING[
                            t - idx1, t - idx0
                        ] = 0.0
                        counter += 2

                    second_feature_counter += 1

            SECOND_FEATURES = copy.deepcopy(
                SECOND_FEATURES_BASE[:, :second_feature_counter]
            )
            ##########
            # Reconstruct rejection region
            ##########
            STATE_DIST_POST, POLICY_ARRAY = reconstruct_rejection_region(
                t,
                n_max,
                max_feature_counter,
                copy.deepcopy(CANDIDATE_STATE_ENCODING),
                CARRY_OVER_STATE_ENCODING,
                STATE_DIST_POST,
                key_weights_rectified,
                n_nonzero_features,
                nonzero_indices,
            )

            RISK_ACCUMULATION[t, :] = copy.deepcopy(
                base_accumulated_risk
            ) + copy.deepcopy(
                SECOND_FEATURES @ key_weights_rectified[:second_feature_counter]
            )

        else:
            # Need to ensure that this is well-defined for the policy_compression step
            POLICY_ARRAY = np.zeros((n_max + 1, n_max + 1))

        # Copy post to pre in advance of the next step of the loop
        STATE_DIST_PRE = copy.deepcopy(STATE_DIST_POST)

        policy_array_compressed = compress_policy_simple(t, POLICY_ARRAY)
        POLICY_LIST_COMPRESSED.append(policy_array_compressed)

        if save_policy_array:
            np.save(save_policy_path + f"time_{t}.npy", POLICY_ARRAY[: t + 1, : t + 1])

    # Return policy and associated certification / validation information
    return POLICY_LIST_COMPRESSED, RISK_ACCUMULATION, POINTS_ARRAY


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
        default=129,
        help=(
            "Number of control points used to approximate worst-case Type-1 Error. First "
            "of three numerically important STEP parameters. More n_points adds precision "
            "at the expense of additional computation. In practice, ~50 is often sufficient. "
            "Defaults to 129."
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
        type=str,
        default="False",
        help=(
            "Toggle whether to use p_norm or zeta function shape family for the risk budget. "
            "If True, uses p_norm shape; else, uses zeta function shape family. "
            "Defaults to False (zeta function partial sum family)."
        ),
    )
    parser.add_argument(
        "-sa",
        "--save_array",
        type=bool,
        default=False,
        help=(
            "Toggle whether to also save the policy as an explicit (inefficient) numpy array lookup table."
            "If True, the subdirectory 'array/' is made under the policy folder, and the policy is saved in a "
            "sequence of arrays corresponding to each time step. Will require significant memory (~300Mb for Nmax=500)"
            "Defaults to False."
        ),
    )

    args = parser.parse_args()

    if args.use_p_norm[0] == "F" or args.use_p_norm[0] == "f":
        args.use_p_norm = False
    elif args.use_p_norm[0] == "T" or args.use_p_norm[0] == "t":
        args.use_p_norm = True
    else:
        raise ValueError("Invalid input for use_p_norm; must begin with 'T' or 'F'")

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

    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    results_path = f"policies/n_max_{args.n_max}_alpha_{args.alpha}_shape_parameter_{args.log_p_norm}_pnorm_{args.use_p_norm}/"
    full_save_path = os.path.normpath(os.path.join(base_path, results_path))
    if not os.path.isdir(full_save_path):
        os.makedirs(full_save_path)

    special_policy_array_save_path = os.path.join(full_save_path, f"array/")
    if args.save_array:
        if not os.path.isdir(special_policy_array_save_path):
            os.makedirs(special_policy_array_save_path)

    (
        POLICY_LIST_COMPRESSED,
        RISK_ACCUMULATION,
        POINTS_ARRAY,
    ) = run_step_policy_synthesis(
        args.n_max,
        args.alpha,
        args.n_points,
        lambda_value,
        major_axis_length,
        args.log_p_norm,
        args.use_p_norm,
        save_policy_array=args.save_array,
        save_policy_path=special_policy_array_save_path,
    )

    with open(full_save_path + "/" + "policy_compressed.pkl", "wb") as filename:
        pickle.dump(POLICY_LIST_COMPRESSED, filename)

    np.save(full_save_path + "/" + f"risk_accumulation.npy", RISK_ACCUMULATION)
    np.save(full_save_path + "/" + f"points_array.npy", POINTS_ARRAY)

    print(f"STEP policy saved at {full_save_path}.")
