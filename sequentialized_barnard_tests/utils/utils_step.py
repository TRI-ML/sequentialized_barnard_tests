"""STEP-specific utility functions.

This module defines utility functions that are used solely by STEP procedures.
"""

import copy
import warnings
from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def partial_zeta_function(N: int, p: float, alpha: float):
    """Computes partial sums of the Riemann Zeta function as a shape class
    for the STEP risk budget.

    Args:
        N (int): Number of elements in the series. Must be > 0.
        p (float): Argument to the zeta function. Recommended to lie in [-4, 4] for numerical stability.
        alpha (float): Type-1 Error constraint used to normalize the partial sum to the testing setting.

    Raises:
        ValueError: If N or alpha is invalid.
        warning: If abs(p) > 4.

    Returns:
        (Arraylike): The differential risk accumulation for this shape parameter setting.
    """
    # Error handling on inputs
    try:
        assert N > 0
        assert 0.0 < alpha and alpha < 1.0
    except:
        raise ValueError("Invalid inputs; N must be > 0 and alpha must lie in (0., 1.)")

    if np.abs(p) > 4.0:
        warnings.warn(
            "abs(p) > 4 -- this may lead to numerical issues and is not recommended."
        )

    running_diffs = np.zeros(N + 1)
    partial_sum = 0.0
    for k in range(1, N + 1):
        running_diffs[k] = 1.0 / (k**p)

    partial_sum = np.sum(running_diffs)

    assert np.isclose(np.sum(running_diffs * alpha / partial_sum), alpha)

    return running_diffs * alpha / partial_sum


def p_norm_function(N: int, p: float, alpha: float):
    """Computes values of the function y = x ** p for x in [0., 1.] as a shape class
    for the cumulative STEP risk budget.

    Args:
        N (int): Number of elements in the series. Must be > 0.
        p (float): Exponent in cumulative risk function y = x ** p. Must be >= 0.
        alpha (float): Type-1 Error constraint used to normalize the partial sum to the testing setting.

    Raises:
        ValueError: If N, p, or alpha is invalid.

    Returns:
        (Arraylike): The differential risk accumulation for this shape parameter setting.
    """
    # Error handling on inputs
    try:
        assert N > 0
        assert 0.0 < alpha and alpha < 1.0
        assert p > 0.0
    except:
        raise ValueError(
            "Invalid inputs; N and p must each be > 0 and alpha must lie in (0., 1.)"
        )

    running_accumulation = np.zeros(N + 1)
    for k in range(1, N + 1):
        running_accumulation[k] = (float(k) / float(N)) ** p

    assert np.isclose(alpha * running_accumulation[-1], alpha)

    return np.concatenate((np.zeros(1), np.diff(running_accumulation * alpha)))


def compress_policy_simple(t: int, policy_array_at_time_t: ArrayLike):
    """Simple method to compress a general (t+1, t+1) policy array into a smaller and more
    efficient representation, relying on symmetry and monotonicity properties of the optimal boundary.

    Only compresses in a one-sided fashion due to assumed symmetry of the optimal policy (e.g., that constrained
    alpha and beta, as well as risk budgets, are identical).

    Args:
        t (int): Current time step.
        policy_array_at_time_t (ArrayLike): Full policy array. Size (t+1, t+1)

    Returns:
        policy_compressed (List[ArrayLike]): List of cutoffs by x for the current optimal policy. Encodes optimal actions to take on sequential data.
    """
    try:
        del policy_compressed
    except:
        pass
    policy_compressed = []

    for i in range(t + 1):
        counter_up = i
        while counter_up <= t and policy_array_at_time_t[i, counter_up] <= 1e-6:
            counter_up += 1
        critical_idx_up = counter_up - 1
        counter_non_unity = 0
        while (
            critical_idx_up + counter_non_unity <= t
            and policy_array_at_time_t[i, critical_idx_up + counter_non_unity]
            <= 1.0 - 1e-6
        ):
            counter_non_unity += 1
        counter_non_unity -= 1

        current_policy = np.zeros(counter_non_unity + 1)
        current_policy[0] = critical_idx_up
        for k in range(1, counter_non_unity + 1):
            current_policy[k] = policy_array_at_time_t[i, critical_idx_up + k]

        if critical_idx_up < t:
            assert policy_array_at_time_t[i, critical_idx_up + 1] > 0.0
        else:
            pass

        policy_compressed.append(current_policy)

    return policy_compressed


def synthesize_risk_budget(
    custom_differential_risk_budget: Union[ArrayLike, None],
    risk_budget_shape_parameter: float,
    use_p_norm: bool,
    dead_time: int,
    n_max: int,
    alpha: float,
    verbose: bool,
):
    """Handle the initial set of inputs to the policy synthesis function to ensure clear and repeatable synthesis by accounting for logic of different risk budgets, etc.

    Args:
        custom_differential_risk_budget (ArrayLike): Custom differential (user-specified) risk budget. Must be nonnegative; can be None.
        risk_budget_shape_parameter (float): Shape parameter for both partial_zeta and p_norm risk budget shape families.
        use_p_norm (bool): If custom_differential_risk_budget is None, decide whether to use p_norm or partial_zeta shape family.
        dead_time (int): Number of steps to wait before attempting to find non-trivial decision. Generally logarithmic in n_max; can improve numerical conditioning if set to at least 4.
        n_max (int): Maximum number of comparison evaluations. Must be >= 1.
        alpha (float): Type-1 Error rate constraint. Must lie in (0., 1.)
        verbose (bool): Toggle additional output to the user.

    Raises:
        ValueError: If custom_differential_risk_budget is included but contains a negative element. This is invalid, as risk cannot be undone once taken.

    Returns:
        diff_mass_removal_array (ArrayLike): Differential risk of Type-1 Error. Equals concatenation of (0, np.diff(cumulative_mass_removal_array))
        cumulative_mass_removal_array (ArrayLike): Accumulated risk of Type-1 Error. Last element should be close to alpha
    """
    if custom_differential_risk_budget is not None:
        if np.min(custom_differential_risk_budget) < 0.0:
            raise ValueError(
                "custom differential risk budget is included, but contains a negative entry. Differential risk is always non-negative!"
            )
        try:
            assert np.array(custom_differential_risk_budget).shape[0] == n_max + 1
            custom_differential_risk_budget = np.array(custom_differential_risk_budget)
            custom_differential_risk_budget[0] = 0.0
        except:
            warnings.warn(
                "Length of custom_differential_risk_budget does not match n_max + 1; truncating or extending as appropriate"
            )
            if np.array(custom_differential_risk_budget).shape[0] > n_max + 1:
                custom_differential_risk_budget = np.concatenate(
                    (
                        np.zeros(1),
                        np.array(custom_differential_risk_budget)[1 : n_max + 1],
                    )
                )
            else:
                C = np.array(custom_differential_risk_budget).shape[0]
                custom_differential_risk_budget = np.concatenate(
                    (np.zeros(n_max + 1 - C), np.array(custom_differential_risk_budget))
                )

        if verbose:
            print(
                "Received a valid custom differential risk budget; this takes precedence over p_norm and partial zeta function budgets; use_p_norm flag will be IGNORED!"
            )

        diff_mass_removal_array = (
            custom_differential_risk_budget
            * alpha
            / np.sum(custom_differential_risk_budget)
        )
        cumulative_mass_removal_array = np.cumsum(diff_mass_removal_array)

    else:
        # Use either p_norm or partial zeta function depending on use_p_norm flag
        if use_p_norm:
            # p_norm risk budget shape
            mass_removal_p_norm = np.exp(risk_budget_shape_parameter)
            tmp_diff = p_norm_function(
                n_max + 1 - dead_time, mass_removal_p_norm, alpha
            )

        else:
            # Partial zeta function risk budget shape
            tmp_diff = partial_zeta_function(
                n_max + 1 - dead_time, risk_budget_shape_parameter, alpha
            )

        # Assign cumulative risk budget using cumsum of tmp_diff, with accounting for dead_time
        cumulative_mass_removal_array = np.zeros(n_max + 1)
        cumulative_mass_removal_array[dead_time:] = np.cumsum(tmp_diff[1:])

        # Assign differential risk budget as diff of cumulative risk budget
        diff_mass_removal_array = np.zeros(n_max + 1)
        diff_mass_removal_array[1:] = np.diff(cumulative_mass_removal_array)

    return diff_mass_removal_array, cumulative_mass_removal_array


def quadratic_score(N, lambda_value, major_axis_length, reverse=False):
    """Scoring function to assign the order in which states are considered for rejection. This
    can significantly reduce the size of the linear programs, accelerating policy synthesis.

    Args:
        N (int): Current time step. Number of states is (N+1) ** 2
        lambda_value (float): First shape parameter which determines the quadratic score ordering
        major_axis_length (float): Second shape parameter which determines the quadratic score ordering
        reverse (bool, optional): Toggle the assignment of the null and alternative hypotheses. If False, then alternative is p1 > p0; else alternative is p1 < p0. Defaults to False.

    Returns:
        STATES_SORTED (ArrayLike): States (two-tuples) sorted according to quadratic score distance.
        IDX_SORTED (ArrayLike): Indices of associated states under reshape procedure.
    """
    IDX = np.zeros((N + 1, N + 1, 2))
    for i in range(N + 1):
        for j in range(N + 1):
            IDX[i, j, 0] = i
            IDX[i, j, 1] = j

    STATES = np.copy(IDX) / N

    IDX = IDX.reshape(-1, 2)
    STATES = STATES.reshape(-1, 2)

    minor_axis_length = lambda_value / np.sqrt(N)
    # major_axis_length = minor_axis_length*major_minor_ratio

    SCORES = np.zeros((STATES.shape[0],))

    for k in range(STATES.shape[0]):
        x1 = STATES[k, 0] - 0.5
        y1 = STATES[k, 1] - 0.5
        xp1 = (1.0 / np.sqrt(2.0)) * (x1 + y1)
        yp1 = (1.0 / np.sqrt(2.0)) * (y1 - x1)

        SCORES[k] = (xp1 / major_axis_length) ** 2 + (yp1 / minor_axis_length) ** 2
        if reverse:
            if yp1 >= 0.0:
                SCORES[k] = 0.0
        else:
            if yp1 <= 0.0:
                SCORES[k] = 0.0
        # if yp1 == 0:
        #     SCORES[k] = 1e-4*(np.abs(xp1) + 1e-4)
        # elif yp1 < 0:
        #     SCORES[k] = 0.

    idx_scores = np.argsort(SCORES)[::-1]
    STATES_SORTED = STATES[idx_scores, :]
    IDX_SORTED = IDX[idx_scores, :]

    return STATES_SORTED, IDX_SORTED


def run_single_state_assignment(
    t: int,
    base_accumulated_risk: ArrayLike,
    critical_risk_target: float,
    lambda_value: float,
    major_axis_length: float,
    n_points: int,
    STATE_DIST_POST: ArrayLike,
):
    """Assign states using quadratic score sorted states, accounting for symmetry in the desired policies.

    Args:
        t (int): Current time step.
        base_accumulated_risk (ArrayLike): Accumulated risk for each control point up to the previous time step
        critical_risk_target (float): Current target accumulated risk for each control point
        lambda_value (float): First shape parameter which determines the quadratic score ordering
        major_axis_length (float): Second shape parameter which determines the quadratic score ordering
        n_points (int): Number of control points to effect Type-1 Error control
        STATE_DIST_POST (ArrayLike): Current mass distribution, to decide sufficient added states for good numerical performance.

    Returns:
        current_accumulated_risk (ArrayLike): Accumulated risk for each control point up to the current time step
        CANDIDATE_STATE_ENCODING (ArrayLike): Encoding of non-zero-mass candidate rejection states
        CARRY_OVER_STATE_ENCODING (ArrayLike): Encoding of zero-mass (free) rejection states.
    """
    _, I = quadratic_score(t, lambda_value, major_axis_length)
    nI = I.shape[0]

    CANDIDATE_STATE_ENCODING = np.zeros((t + 1, t + 1))
    CARRY_OVER_STATE_ENCODING = np.zeros((t + 1, t + 1))

    current_accumulated_risk = copy.deepcopy(np.zeros(n_points) + base_accumulated_risk)

    counter = 0
    # Iterate until every null hypothesis has exceeded its risk limit (to promote well-posedness of the optimization)
    while (
        counter == 0 or (np.min(current_accumulated_risk) <= critical_risk_target)
    ) and counter < nI:
        idx0 = int(I[counter, 0])
        idx1 = int(I[counter, 1])
        if idx1 > idx0:
            if np.max(STATE_DIST_POST[idx0, idx1, :]) > 0.0:
                CANDIDATE_STATE_ENCODING[idx0, idx1] = 1.0
                current_accumulated_risk += STATE_DIST_POST[idx0, idx1, :]
            else:
                CARRY_OVER_STATE_ENCODING[idx0, idx1] = 1.0

        counter += 1

    # Continue adding additional states up to three additional semi-diagonals
    critical_delta = idx1 - idx0
    for k in range(counter, counter + 1 + 3 * (t - critical_delta)):
        if k < nI:
            idx0 = int(I[k, 0])
            idx1 = int(I[k, 1])
            if idx1 > idx0:
                if np.max(STATE_DIST_POST[idx0, idx1, :]) > 0.0:
                    CANDIDATE_STATE_ENCODING[idx0, idx1] = 1.0
                    current_accumulated_risk += STATE_DIST_POST[idx0, idx1, :]
                else:
                    CARRY_OVER_STATE_ENCODING[idx0, idx1] = 1.0

    # Enforce symmetry in CANDIDATE_STATE_ENCODING and CARRY_OVER_STATE_ENCODING
    if not np.all(
        np.isclose(
            CANDIDATE_STATE_ENCODING,
            np.transpose(np.flip(CANDIDATE_STATE_ENCODING)),
        )
    ):
        CANDIDATE_STATE_ENCODING = np.maximum(
            CANDIDATE_STATE_ENCODING,
            np.transpose(np.flip(CANDIDATE_STATE_ENCODING)),
        )
        assert np.all(
            np.isclose(
                CANDIDATE_STATE_ENCODING,
                np.transpose(np.flip(CANDIDATE_STATE_ENCODING)),
            )
        )

    if not np.all(
        np.isclose(
            CARRY_OVER_STATE_ENCODING,
            np.transpose(np.flip(CARRY_OVER_STATE_ENCODING)),
        )
    ):
        CARRY_OVER_STATE_ENCODING = np.maximum(
            CARRY_OVER_STATE_ENCODING,
            np.transpose(np.flip(CARRY_OVER_STATE_ENCODING)),
        )
        assert np.all(
            np.isclose(
                CARRY_OVER_STATE_ENCODING,
                np.transpose(np.flip(CARRY_OVER_STATE_ENCODING)),
            )
        )

    return current_accumulated_risk, CANDIDATE_STATE_ENCODING, CARRY_OVER_STATE_ENCODING


def reconstruct_rejection_region(
    t: int,
    n_max: int,
    max_feature_counter: int,
    DISPOSABLE_CANDIDATE_STATE_ENCODING: ArrayLike,
    CARRY_OVER_STATE_ENCODING: ArrayLike,
    STATE_DIST_POST: ArrayLike,
    key_weights: ArrayLike,
    n_nonzero_features: int,
    nonzero_indices: ArrayLike,
):
    """Use the result of the LP optimization to expand the rejection region of the policy.

    Args:
        t (int): Current sequence step
        n_max (int): Maximal number of sequence steps
        max_feature_counter (int): Maximum number of features included based upon candidate states for rejection
        DISPOSABLE_CANDIDATE_STATE_ENCODING (ArrayLike): Modifiable candidate state encoding to update STATE_DIST_POST from optimized rejection region.
        CARRY_OVER_STATE_ENCODING (ArrayLike): State encoding of zero-mass (free) rejection states to fill out Policy encoding.
        STATE_DIST_POST (ArrayLike): Modified state distribution (pre-expansion of new rejection region), to be modified by rejection region expansion.
        key_weights (ArrayLike): Optimized rejection region weights.
        n_nonzero_features (int): Derived parameter to clarify candidate rejection state size.
        nonzero_indices (ArrayLike): Indices of candidate rejection states with non-zero mass (not free rejections).

    Returns:
        STATE_DIST_POST (ArrayLike): Reduced-mass state distribution from expansion of rejection region.
        POLICY_ARRAY (ArrayLike): (Inefficient) decision encoding of the optimized rejection region (to be compressed in the full synthesis loop).
    """
    POLICY_ARRAY = np.zeros((n_max + 1, n_max + 1))
    feature_counter = int(0)
    for k in range(n_nonzero_features):
        idx0 = nonzero_indices[k, 0]
        idx1 = nonzero_indices[k, 1]

        if (
            np.isclose(DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1], 0.0)
            or feature_counter >= max_feature_counter
        ):
            pass
        else:
            if idx0 + idx1 == t:
                # Rejection region
                STATE_DIST_POST[idx0, idx1, :] *= 1.0 - key_weights[feature_counter]
                POLICY_ARRAY[idx0, idx1] = key_weights[feature_counter]

                # Acceptance region
                STATE_DIST_POST[idx1, idx0, :] *= 1.0 - key_weights[feature_counter]
                POLICY_ARRAY[idx1, idx0] = -key_weights[feature_counter]
                DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
            else:
                # Rejection region
                STATE_DIST_POST[idx0, idx1, :] *= 1.0 - key_weights[feature_counter]
                STATE_DIST_POST[t - idx1, t - idx0, :] *= (
                    1.0 - key_weights[feature_counter]
                )
                POLICY_ARRAY[idx0, idx1] = key_weights[feature_counter]
                POLICY_ARRAY[t - idx1, t - idx0] = key_weights[feature_counter]

                # Acceptance region
                STATE_DIST_POST[idx1, idx0, :] *= 1.0 - key_weights[feature_counter]
                STATE_DIST_POST[t - idx0, t - idx1, :] *= (
                    1.0 - key_weights[feature_counter]
                )
                POLICY_ARRAY[idx1, idx0] = -key_weights[feature_counter]
                POLICY_ARRAY[t - idx0, t - idx1] = -key_weights[feature_counter]

                DISPOSABLE_CANDIDATE_STATE_ENCODING[idx0, idx1] = 0.0
                DISPOSABLE_CANDIDATE_STATE_ENCODING[t - idx1, t - idx0] = 0.0
            feature_counter += 1

    n_zero_features = int(np.sum(CARRY_OVER_STATE_ENCODING))
    zero_indices = np.argwhere(CARRY_OVER_STATE_ENCODING)

    for k in range(n_zero_features):
        idx0 = int(zero_indices[k, 0])
        idx1 = int(zero_indices[k, 1])

        # Rejection region
        STATE_DIST_POST[idx0, idx1, :] *= 0.0
        POLICY_ARRAY[idx0, idx1] = 1.0

        # Acceptance region
        STATE_DIST_POST[idx1, idx0, :] *= 0.0
        POLICY_ARRAY[idx1, idx0] = -1.0

    return STATE_DIST_POST, POLICY_ARRAY
