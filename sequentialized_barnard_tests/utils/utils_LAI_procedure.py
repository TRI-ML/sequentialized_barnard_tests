import numpy as np


def calc_zeta(abs_gap_size: float) -> float:
    """Compute safe version (underestimate) of zeta, with no requirement for nuisance parameter

    Args:
        abs_gap_size (float): Magnitude of true (data-generating) gap between p0 and p1. Lies in [-1, 1]

    Returns:
        zeta: Value of zeta offset in the univariate test. Lies in [0, 0.5]
    """
    return abs_gap_size / (
        2.0 * abs_gap_size + (1.0 - abs_gap_size) * (1.0 - abs_gap_size)
    )


def calc_exact_zeta(abs_gap_size: float, p0: float) -> float:
    """Compute offset induced by naive Wald statistic. Two-policy test becomes a
       univariate parametric test of {H0: p < 0.5, H1: p > 0.5 + zeta}

    Args:
        abs_gap_size (float): Magnitude of true (data-generating) gap between p0 and p1. Lies in [-1, 1]
        p0 (float): True (data-generating) baseline success rate. Lies in [0, 1]

    Returns:
        float: Value of zeta offset in the univariate test. Lies in [0, 0.5]
    """
    return abs_gap_size / (
        2.0 * abs_gap_size + 4.0 * p0 - 4.0 * p0 * p0 - 4.0 * abs_gap_size * p0
    )


def bernoulli_KL(p0: float, p1: float) -> float:
    """Compute KL divergence between Bernoulli distributions. Form is
       KL(p1 || p0). Uses continuity solution if p1 == {0, 1}

    Args:
        p0 (float): Baseline (null) mean
        p1 (float): Novel (alt) mean

    Raises:
        ValueError: Invalid prior (p0 = 0 or p0 = 1) yields infinite KL divergence

    Returns:
        float: KL divergence (i.e., score)
    """
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
       under KL divergence distance metric

    Args:
        p0 (float): Baseline (null) mean
        p1 (float): Novel (alt) mean

    Returns:
        p_mid (float): Midpoint of {p0, p1} in KL distance space
    """
    p_low = p0
    p_high = p1
    assert p1 > p0

    diff_p = p_high - p_low
    while np.abs(diff_p) > 1e-6:
        p_mid = 0.5 * (p_low + p_high)

        KL_DIFF = bernoulli_KL(p0, p_mid) - bernoulli_KL(p1, p_mid)

        if np.abs(KL_DIFF) < 1e-8:
            return p_mid
        elif KL_DIFF < 0.0:
            # Greater gap above -- increase p_mid by raising p_low
            p_low = p_mid
        else:
            # Greater gap below -- decrease p_mid by lowering p_high
            p_high = p_mid

        diff_p = p_high - p_low

    return p_mid


def compute_maxFPR_p(p0: float, p1: float) -> float:
    """Compute p_mid such that (p_mid, p_mid) induces
       highest FPR against (p0, p1). I.e., the hardest
       point in H0 to distinguish from (p0, p1).

       Corresponds to linear interpolation in natural
       parameter space (if nat. parameters are finite
       for both p0 and p1).

    Args:
        p0 (float): Baseline (null) mean. Lies in (0, 1)
        p1 (float): Novel (alt) mean. Lies in (0, 1)

    Raises:
        ValueError: At least one of the natural parameters is +- infinity;
                    therefore, we cannot resolve the interpolation.

    Returns:
        float: p_mid satisfying the maximal-FPR property given above. Lies
               in (0, 1)
    """
    try:
        assert 0 < p0 and p0 < 1
        assert 0 < p1 and p1 < 1
    except:
        raise ValueError("Must be interior to unit square")

    theta_0 = np.log(p0 / (1.0 - p0))
    theta_1 = np.log(p1 / (1.0 - p1))

    return np.exp(0.5 * (theta_0 + theta_1)) / (1.0 + np.exp(0.5 * (theta_0 + theta_1)))


def calc_gamma(theta0: float, theta1: float, c: float) -> float:
    """Implement Eqn 5.3 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        theta0 (float): Null (scalar) parameter
        theta1 (float): Alt (scalar) parameter
        c (float): Regularizer in the optimization problem (real, > 0)

    Returns:
        gamma: Nondimensionalized quantity representing gap in test hypotheses
    """
    assert c > 0
    return abs(theta1 - theta0) / (2.0 * np.sqrt(c))


def h0_star(c: float, n: int) -> float:
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


def hgamma_star_multiplier(c: float, n: int, gamma: float = 0.0) -> float:
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


def g0_star(c: float, n: int) -> float:
    """Implements definition given above Eqn 4.2 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence

    Returns:
        g*_0(float): Evaluation of g_0
    """
    t = c * n
    h0_val = h0_star(c, n)
    return (h0_val**2) / (2.0 * t)


def ggamma_star(c: float, n: int, gamma: float = 0.0) -> float:
    """Implements definition given above Eqn 3.3 in Lai (1988) -- Nearly Optimal Sequential Tests of Composite Hypotheses

    Args:
        c (float): Optimization problem regularizer term (real, in (0, 1))
        n (int): Step number of the sequence

    Returns:
        g*_gamma(float): Evaluation of g_gamma
    """
    t = c * n
    hstar_val = hgamma_star_multiplier(c, n, gamma) * h0_star(c, n)
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
        decision: Decision at current step: {0: Accept Null, 1: Reject Null, 2: Continue}
    """
    test_val = 2
    if bernoulli_KL(bound, sample_mean) >= (ggamma_star(c, n, gamma) / float(n)):
        if sample_mean > bound:  # Accept Alt
            test_val = 1
        else:
            test_val = 0  # Accept Null

    return test_val
