"""Lai-specific utility functions.

This module defines utility functions that are used solely by Lai procedures.
"""

import numpy as np

from sequentialized_barnard_tests.utils.utils_general import bernoulli_kl

# Reference papers:
# Lai (1988): Nearly optimal sequential tests of composite hypotheses
# Lai (1994): Nearly optimal generalized sequential likelihood ratio tests in multivariate exponential families


def calculate_exact_zeta(abs_gap_size: float, p0: float) -> float:
    """Compute offset induced by naive Wald statistic. Two-policy test becomes a
       univariate parametric test of {H0: p < 0.5, H1: p > 0.5 + zeta}

    Args:
        abs_gap_size (float): Magnitude of true (data-generating) gap between p0 and p1. Lies in [0, 1]
        p0 (float): True (data-generating) baseline success rate. Lies in [0, 1]

    Raises:
        ValueError: If abs_gap_size is not in [0, 1 - p0]

    Returns:
        zeta: Value of zeta offset in the univariate test. Lies in [0, 0.5]
    """
    if not (abs_gap_size >= 0.0 and abs_gap_size <= 1.0 - p0):
        raise ValueError("abs_gap_size is not in [0, 1 - p0]")

    return abs_gap_size / (
        2.0 * abs_gap_size + 4.0 * p0 - 4.0 * p0 * p0 - 4.0 * abs_gap_size * p0
    )


def calculate_robust_zeta(abs_gap_size: float) -> float:
    """Compute safe version (underestimate) of zeta, with no requirement for nuisance parameter

    Args:
        abs_gap_size (float): Magnitude of true (data-generating) gap between p0 and p1. Lies in [0, 1]

    Raises:
        ValueError: If abs_gap_size is not in [0, 1].

    Returns:
        zeta: Value of zeta offset in the univariate test. Lies in [0, 0.5]
    """
    if not (abs_gap_size >= 0.0 and abs_gap_size <= 1.0):
        raise ValueError("abs_gap_size is not in [0, 1]")
    return abs_gap_size / (
        2.0 * abs_gap_size + (1.0 - abs_gap_size) * (1.0 - abs_gap_size)
    )


def calculate_gamma(theta0: float, theta1: float, c: float) -> float:
    """Implement Eqn 5.3 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        theta0 (float): Null (scalar) parameter
        theta1 (float): Alt (scalar) parameter
        c (float): Regularizer in the optimization problem (real, > 0)

    Raises:
        ValueError: If c <= 0.

    Returns:
        gamma: Nondimensionalized quantity representing gap in test hypotheses
    """
    if not c > 0:
        raise ValueError("Regularizer c must be positive")

    return abs(theta1 - theta0) / (2.0 * np.sqrt(c))


def compute_h0_star(c: float, n: int) -> float:
    """Implement Eqn 2.12 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence

    Returns:
        h*_0(float): No-gap optimal decision boundary width
    """
    t = c * n
    if t < 0.01:
        val = np.sqrt(
            t
            * (
                2.0 * np.log(1.0 / t)
                + np.log(np.log(1.0 / t))
                + -np.log(4.0 * np.pi)
                - 3.0 * np.exp(-0.016 / np.sqrt(t))
            )
        )
    elif t < 0.1:
        val = 0.39 - (0.015 / np.sqrt(t))
    elif t < 0.8:
        val = np.exp(-0.69 * t - 1)
    else:
        val = (
            0.25
            * (np.sqrt(2.0 / np.pi))
            * ((1.0 / np.sqrt(t)) - (5.0 / (48.0 * np.pi * (t ** (5 / 2)))))
        )

    return val


def compute_hgamma_star_multiplier(c: float, n: int, gamma: float = 0.0) -> float:
    """Implement Eqn 2.13 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence
        gamma (float, optional): Nondimensionalized quantity representing gap in test hypotheses. Defaults to 0.0.

    Returns:
        h*_gamma(float): Multiplier (wrt h*_0) of optimal decision boundary width
    """
    if c * n >= 1.0:
        return np.exp(-gamma * gamma * c * n / 2.0)
    else:
        return np.exp(-gamma * gamma * ((c * n) ** 1.125) / 2.0)


def compute_g0_star(c: float, n: int) -> float:
    """Implements definition given above Eqn 4.2 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence

    Returns:
        g*_0(float): Evaluation of g_0
    """
    t = c * n
    h0_val = compute_h0_star(c, n)
    return (h0_val**2) / (2.0 * t)


def compute_ggamma_star(c: float, n: int, gamma: float = 0.0) -> float:
    """Implements definition given above Eqn 3.3 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence

    Returns:
        g*_gamma(float): Evaluation of g_gamma
    """
    t = c * n
    hstar_val = compute_hgamma_star_multiplier(c, n, gamma) * compute_h0_star(c, n)
    return ((hstar_val + gamma * t) ** 2) / (2.0 * t)


def run_test_step_gamma_uniparameter(
    c: float, n: int, sample_mean: float, bound: float, gamma: float = 0.0
) -> float:
    """Run a single test step ofthe decision making rule in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence
        sample_mean (float): Empirical mean of the induced univariate Bernoulli
        bound (float): Critical Bernoulli parameter (real, float, in [0.5, 1])
        gamma (float, optional): Known or assumed gap in the hypothesis class. Defaults to 0.0.

    Returns:
        decision: Decision at current step: {1: Reject Null, 0: Continue, -1: Accept Null}
    """
    test_val = 0
    if bernoulli_kl(bound, sample_mean) >= (
        compute_ggamma_star(c, n, gamma) / float(n)
    ):
        if sample_mean > bound:  # Reject Null; Accept Alternative
            test_val = 1
        else:  # Accept Null and stop early
            test_val = -1

    return test_val
