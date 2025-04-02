"""General utility functions.

This module defines utility functions that are shared across sequential methods.
"""

import warnings

import numpy as np


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
