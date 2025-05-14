"""General utility functions.

This module defines utility functions that are shared across sequential methods.
"""

import copy
import warnings

import numpy as np
from numpy.typing import ArrayLike


def bernoulli_kl(p0: float, p1: float) -> float:
    """Compute KL divergence between Bernoulli distributions. Form is
       KL(p1 || p0). Utilizes continuity solution at p1 = 0 and p1 = 1

    Args:
        p0 (float): Baseline (null) mean
        p1 (float): Novel (alt) mean

    Raises:
        ValueError: Invalid prior (p0 = 0 or p0 = 1) yields infinite KL divergence

    Returns:
        float: KL divergence (i.e., score)
    """
    try:
        # Raise error unless p0 in (0, 1) and p1 in [0, 1]
        assert 0.0 < p0 and 1.0 > p0
        assert 0.0 <= p1 and 1.0 >= p1
    except:
        raise ValueError(
            "Invalid p0 or p1; must be in (0, 1) and [0, 1], respectively."
        )

    # Normal setting
    if p0 < 1.0 and p0 > 0.0 and p1 < 1.0 and p1 > 0.0:
        return p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))
    # Use continuous limit of f(x) = x * log(x) if p1 exists on either extreme (limit as x goes to zero is taken to be 0 log0 = 0)
    elif p1 == 0.0 and p0 < 1.0 and p0 > 0.0:
        return (1 - p1) * np.log((1 - p1) / (1 - p0))
    elif p1 == 1.0 and p0 < 1.0 and p0 > 0.0:
        return p1 * np.log(p1 / p0)
    # Raise error if p0 on either extreme
    else:
        raise ValueError("Invalid p0; must be in (0, 1) -- NOT including endpoints!")


def compute_middle_p(p0: float, p1: float) -> float:
    """Binary search to find p_mid that is equidistant from p0 and p1
       under KL divergence distance metric.

    Args:
        p0 (float): Baseline (null) mean
        p1 (float): Novel (alt) mean

    Returns:
        p_mid (float): Midpoint of {p0, p1} in KL distance space
    """
    try:
        assert 0.0 <= p0 and p0 <= 1.0
        assert 0.0 <= p1 and p1 <= 1.0
    except:
        raise ValueError("Invalid p0 or p1; each must be in [0, 1].")

    if p1 > p0:
        p_low = p0
        p_high = p1
    elif p1 == p0:
        return p0
    else:
        p_low = p1
        p_high = p0

    diff_p = p_high - p_low
    while np.abs(diff_p) > 1e-6:
        p_mid = 0.5 * (p_low + p_high)

        kl_diff = bernoulli_kl(p0, p_mid) - bernoulli_kl(p1, p_mid)

        if np.abs(kl_diff) < 1e-8:
            return p_mid
        elif kl_diff < 0.0:
            # Greater gap above -- increase p_mid by raising p_low
            p_low = p_mid
        else:
            # Greater gap below -- decrease p_mid by lowering p_high
            p_high = p_mid

        diff_p = p_high - p_low

    return p_mid


def compute_natural_middle_p(p0: float, p1: float) -> float:
    """Compute p_mid as the interpolant (in natual parameter space)
       of p0 and p1.

    Args:
        p0 (float): Baseline (null) mean. Lies in [0, 1]
        p1 (float): Novel (alt) mean. Lies in [0, 1]

    Raises:
        Warning: The values are ordered in reverse (p0 >= p1). This corresponds
                to a point in the natural null set.

    Returns:
        float: p_mid satisfying the maximal-FPR property given above. Lies
               in [0, 1]
    """
    if p1 <= p0:
        warnings.warn(
            "Currently, p0 >= p1. Returning interpolant, but check the ordering of inputs."
        )

    try:
        assert 0.0 < p0 and p0 < 1.0
        assert 0.0 < p1 and p1 < 1.0
    except:
        # At least one of them is on the boundary
        #
        # If they are both on opposing boundaries, return
        # midpoint (+- inf --> 0 in natural parameter space)
        if np.isclose(p0, 1.0) and np.isclose(p1, 0.0):
            return 0.5
        elif np.isclose(p0, 0.0) and np.isclose(p1, 1.0):
            return 0.5
        else:
            # Exactly one is on the boundary
            #
            # If the one is close to 1., return 1.
            if np.isclose(p0, 1.0) or np.isclose(p1, 1.0):
                return 1.0
            # Else, return 0.
            if np.isclose(p0, 0.0) or np.isclose(p1, 0.0):
                return 0.0

    theta_0 = np.log(p0 / (1.0 - p0))
    theta_1 = np.log(p1 / (1.0 - p1))

    return np.exp(0.5 * (theta_0 + theta_1)) / (1.0 + np.exp(0.5 * (theta_0 + theta_1)))


def bivariate_bernoulli_kl(P0: ArrayLike, P1: ArrayLike) -> float:
    """Compute KL divergence between bivariate Bernoulli distributions. Form is
       KL(P1 || P0). Utilizes continuity solution of 0 for any event in which P_1(event) = 0

    Args:
        P0 (float): Base pair of means (p0_base, p1_base). Each element must be in (0, 1)
        P1 (float): Alt pair of means (p0_alt, p1_alt). Each element must be in [0., 1.]

    Raises:
        ValueError: Invalid prior: p0_base or p1_base NOT IN (0., 1.). This yields infinite KL divergence

    Returns:
        float: KL divergence (i.e., score)
    """
    p0_base = P0[0]
    p1_base = P0[1]

    p0_alt = P1[0]
    p1_alt = P1[1]

    try:
        # Raise error unless base values in (0, 1) and alt values in [0, 1]
        assert 0.0 < p0_base and 1.0 > p0_base
        assert 0.0 < p1_base and 1.0 > p1_base
        assert 0.0 <= p0_alt and 1.0 >= p0_alt
        assert 0.0 <= p1_alt and 1.0 >= p1_alt
    except:
        raise ValueError(
            "Invalid base and alt pairs; base must have each value in (0, 1) and alt must have each value in [0, 1], respectively."
        )

    p_base = np.zeros(4)
    p_alt = np.zeros(4)

    ###
    # Four outcomes: (0, 0), (1, 0), (0, 1), (1, 1)
    ###

    # Case 1: (0, 0)
    p_base[0] = (1.0 - p0_base) * (1.0 - p1_base)
    p_alt[0] = (1.0 - p0_alt) * (1.0 - p1_alt)
    # Case 2: (1, 0)
    p_base[1] = (p0_base) * (1.0 - p1_base)
    p_alt[1] = (p0_alt) * (1.0 - p1_alt)
    # Case 3: (0, 1)
    p_base[2] = (1.0 - p0_base) * (p1_base)
    p_alt[2] = (1.0 - p0_alt) * (p1_alt)
    # Case 4: (1, 1)
    p_base[3] = (p0_base) * (p1_base)
    p_alt[3] = (p0_alt) * (p1_alt)

    total_divergence = 0.0
    # Add each case, using the continuity condition (div = 0) if the alt evaluates to zero
    for i in range(4):
        if p_alt[i] > 0.0:
            total_divergence = total_divergence + (
                p_alt[i] * np.log(p_alt[i] / p_base[i])
            )

    return total_divergence


def compute_bivariate_middle_p(p0: float, p1: float) -> float:
    """Binary search to find (p_mid, p_mid) that is equidistant from (p0, p0) and (p1, p1)
       under KL divergence pseudo-distance metric.

    Args:
        p0 (float): Baseline (null) mean
        p1 (float): Novel (alt) mean

    Returns:
        p_mid (float): Midpoint of {p0, p1} in KL distance space

    Raises:
        ValueError: if p0 or p1 is outside [0., 1.]
    """
    try:
        assert 0.0 <= p0 and p0 <= 1.0
        assert 0.0 <= p1 and p1 <= 1.0
    except:
        raise ValueError("Invalid p0 or p1; each must be in [0, 1].")

    if p1 > p0:
        p_low = p0
        p_high = p1
    elif p1 == p0:
        return p0
    else:
        p_low = p1
        p_high = p0

    diff_p = p_high - p_low
    while np.abs(diff_p) > 1e-10:
        p_mid = 0.5 * (p_low + p_high)

        kl_diff = bivariate_bernoulli_kl(
            [p0, p0], [p_mid, p_mid]
        ) - bivariate_bernoulli_kl([p1, p1], [p_mid, p_mid])

        if np.abs(kl_diff) < 1e-12:
            return p_mid
        elif kl_diff < 0.0:
            # Greater gap above -- increase p_mid by raising p_low
            p_low = p_mid
        else:
            # Greater gap below -- decrease p_mid by lowering p_high
            p_high = p_mid

        diff_p = p_high - p_low

    return p_mid


def construct_kl_spaced_points_array_via_binary_expansion(
    n_max: int, n_points: int = 129
) -> float:
    """Construct approximately equi-KL-spaced POINT_ARRAY using the bivariate middle p computation.

    Args:
        n_max (int): Sets the extreme values for p0 and p1
        n_points (int): Sets the number of points. Must be 2**n + 1

    Returns:
        p_mid (float): Midpoint of {p0, p1} in KL distance space

    Raises:
        ValueError: n_max < 10 OR n_points =/= 2**n + 1 for some integer n
        ValueError: points_array does not have length n_points OR the stitched value is not equal
    """
    try:
        assert n_max >= 10
        log2_value = int(np.floor(np.log2(n_points - 1)))
        assert (n_points - 1) % (2**log2_value) == 0
    except:
        raise ValueError(
            "Invalid n_max or n_points; former must be >= 10, and latter must be of form 2**n + 1"
        )
    p_min = float(1) / float(n_max)
    p_max = float(n_max - 1) / float(n_max)
    list_of_p_values_low = []
    list_of_p_values_high = []
    list_of_p_values_low.append(p_min)
    list_of_p_values_low.append(0.5)
    list_of_p_values_high.append(0.5)
    list_of_p_values_high.append(p_max)

    while len(list_of_p_values_low) < ((n_points - 1) / 2):
        # Expand lower
        current_p_values_low = np.sort(copy.deepcopy(np.array(list_of_p_values_low)))
        for k in range(current_p_values_low.shape[0] - 1):
            tmp_val = compute_bivariate_middle_p(
                current_p_values_low[k], current_p_values_low[k + 1]
            )
            list_of_p_values_low.append(tmp_val)

        # Expand upper
        current_p_values_high = np.sort(copy.deepcopy(np.array(list_of_p_values_high)))
        for k in range(current_p_values_high.shape[0] - 1):
            tmp_val = compute_bivariate_middle_p(
                current_p_values_high[k], current_p_values_high[k + 1]
            )
            list_of_p_values_high.append(tmp_val)

    final_low_values_sorted = np.sort(np.array(list_of_p_values_low))
    final_high_values_sorted = np.sort(np.array(list_of_p_values_high))

    final_full_values_sorted = copy.deepcopy(
        np.concatenate((final_low_values_sorted[:-1], final_high_values_sorted))
    )

    # Stitch the arrays together, with 0.5 being the overlap value
    try:
        assert final_full_values_sorted.shape[0] == n_points
        assert np.isclose(final_low_values_sorted[-1], final_high_values_sorted[0])
    except:
        raise ValueError(
            "Error in points array construction: did not yield n_points length OR the overlap values were mismatched"
        )

    return final_full_values_sorted
