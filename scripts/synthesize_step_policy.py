"""Method to run policy synthesis for STEP procedure.

Policies are stored in sequentialized_barnard_tests/policies.
"""

import copy
import os
import pickle
import sys

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import convolve
from scipy.optimize import linprog
from tqdm import tqdm

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
    custom_differential_risk_budget: ArrayLike = None,
    dead_time: int = None,
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
        verbose (bool, optional): Toggle the printing of progress measures and additional information throughout the synthesis procedure. Defaults to False.

    Raises:
        ValueError: If invalid required arguments
        ValueError: If invalid specified dead_time
        ValueError: If cumulative mass removal arrays do not terminate near alpha (making the procedure either loose, if below alpha, or invalid, if above alpha)
        ValueError: If control points are not assigned with proper extremal (min and max) limits

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

    # Compressed-on-the-fly memory representation
    POLICY_LIST_COMPRESSED = []

    # For post-synthesis verification
    RISK_ACCUMULATION = np.zeros((n_max + 1, n_points))

    # Begin loop to synthesize the optimal policy
    for t in tqdm(range(1, n_max + 1)):
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
            critical_risk_target = cumulative_mass_removal_array[t]
            (
                current_accumulated_risk,
                CANDIDATE_STATE_ENCODING,
                CARRY_OVER_STATE_ENCODING,
            ) = run_single_state_assignment(
                t,
                base_accumulated_risk,
                critical_risk_target,
                lambda_value,
                major_axis_length,
                n_points,
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

            FEATURES_BASE = np.zeros((n_points, n_nonzero_features))
            feature_counter = int(0)
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
                    else:
                        FEATURES_BASE[:, feature_counter] = copy.deepcopy(
                            STATE_DIST_POST[idx0, idx1, :]
                        ) + copy.deepcopy(STATE_DIST_POST[t - idx1, t - idx0, :])
                        DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
                        DISPOSABLE_CANDIDATE_STATE_ENCODING[t - idx1, t - idx0] = 0.0

                    feature_counter += 1

            ##########
            # Set up and run optimization
            ##########
            FEATURES = copy.deepcopy(FEATURES_BASE[:, :feature_counter])
            bounds = (0.0, 1.0)

            b_ub = np.maximum(
                critical_risk_target * np.ones(n_points) - base_accumulated_risk - 1e-6,
                np.zeros(critical_risk_target.shape),
            )

            max_feature_counter = feature_counter

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
                        c_vec, FEATURES, b_ub, bounds=bounds, options=linprog_options
                    )
                except:
                    print(c_vec.shape)
                    print(feature_counter)
                    print()

                key_weights = res_linprog.x

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
                key_weights,
                n_nonzero_features,
                nonzero_indices,
            )

            RISK_ACCUMULATION[t, :] = copy.deepcopy(
                base_accumulated_risk
            ) + copy.deepcopy(FEATURES @ key_weights)

        # Copy post to pre in advance of the next step of the loop
        STATE_DIST_PRE = copy.deepcopy(STATE_DIST_POST)

        policy_array_compressed = compress_policy_simple(t, POLICY_ARRAY)
        POLICY_LIST_COMPRESSED.append(policy_array_compressed)

    # Return policy and associated certification / validation information
    return POLICY_LIST_COMPRESSED, RISK_ACCUMULATION, POINTS_ARRAY


if __name__ == "__main__":
    try:
        n_max = int(sys.argv[1])
    except:
        n_max = int(200)

    try:
        alpha = float(sys.argv[2])
    except:
        alpha = 0.05
    try:
        min_gap = float(sys.argv[3])
    except:
        min_gap = float(0.0)

    try:
        data_dependent_flag = bool(int(sys.argv[4]))
    except:
        data_dependent_flag = False

    try:
        n_points = int(sys.argv[5])
    except:
        n_points = int(49)

    try:
        lambda_value = float(sys.argv[6])
    except:
        lambda_value = 2.1

    try:
        major_axis_length = float(sys.argv[7])
    except:
        major_axis_length = 1.4

    try:
        log_mass_removal_p_norm = np.log(float(sys.argv[8]))
    except:
        log_mass_removal_p_norm = 0.0

    mass_removal_p_norm = np.exp(log_mass_removal_p_norm)

    use_p_norm = False

    (
        POLICY_LIST_COMPRESSED,
        RISK_ACCUMULATION,
        POINTS_ARRAY,
    ) = run_step_policy_synthesis(
        n_max,
        alpha,
        n_points,
        lambda_value,
        major_axis_length,
        log_mass_removal_p_norm,
        use_p_norm=use_p_norm,
    )

    base_path = os.getcwd()
    results_path = f"sequentialized_barnard_tests/policies/n_max_{n_max}_alpha_{alpha}_shape_parameter_{log_mass_removal_p_norm}_pnorm_{use_p_norm}/"
    full_save_path = os.path.join(base_path, results_path)
    if not os.path.isdir(full_save_path):
        os.makedirs(full_save_path)

    try:
        with open(full_save_path + "policy_compressed.pkl", "wb") as filename:
            pickle.dump(POLICY_LIST_COMPRESSED, filename)
    except:
        raise ValueError(
            "Could not find configuration file! Run 'python make_config.py Nmax alpha min_gap random_seed_integer data_dependent_flag' to create it!"
        )
