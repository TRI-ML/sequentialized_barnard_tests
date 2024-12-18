import numpy as np
from binomial_cis import binom_ci  # Joe Vincent's package
from tqdm import tqdm
from utils.fixed_N_binomial import binomial_mean_interval_estimator


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


def evaluate_true_errors(RESULTS_NULL, RESULTS_ALT, A_STAR_INVERSE, B_STAR):
    N_TRIALS = RESULTS_NULL.shape[0]
    assert N_TRIALS == RESULTS_ALT.shape[0]

    Nmax = RESULTS_NULL.shape[1]
    count_FP = 0
    count_FN = 0

    for k in range(N_TRIALS):
        for t in range(Nmax):
            if RESULTS_NULL[k, t, 1] >= B_STAR:
                count_FP += 1
                break
            elif (1.0 / RESULTS_NULL[k, t, 0]) >= A_STAR_INVERSE:
                break
            else:
                pass

        for t in range(Nmax):
            if 1.0 / RESULTS_ALT[k, t, 0] >= A_STAR_INVERSE:
                count_FN += 1
                break
            elif RESULTS_ALT[k, t, 1] >= B_STAR:
                break
            else:
                pass

    return (count_FP / N_TRIALS), (count_FN / N_TRIALS)


def calibrate_sprt(
    Nmax: int,
    alpha: float,
    p0: float,
    p1: float,
    n_trials: int = 5000,
    max_power_nominal: float = 0.999,
    n_parallel_evaluations: int = 45,
) -> tuple[float, float]:

    assert 0.0 < p0 and p0 < 1.0
    assert 0.0 < p1 and p1 < 1.0
    assert Nmax >= 1
    assert n_trials > 100
    assert 0.0 < alpha and 1.0 > alpha
    assert 0.0 < max_power_nominal and 1.0 > max_power_nominal
    assert n_parallel_evaluations >= 1

    # A* and B* each, n_cases times
    NUMBER_OF_ESTIMATED_PARAMETERS = 2 * n_parallel_evaluations

    # Ensure that we only dilute the risk by at most 2%
    INTERVAL_RISK = np.minimum(1e-4, alpha / 50.0)

    # Interval accounts for multiple evaluations to get a net risk of INTERVAL_RISK
    check_interval = binomial_mean_interval_estimator(
        -1, INTERVAL_RISK / NUMBER_OF_ESTIMATED_PARAMETERS, n_trials
    )

    # Ensure that the end result achieves ALPHA_NOMINAL by using union bound (i.e., subtract INTERVAL_RISK from alpha_nominal)
    constant_numerator = int(n_trials * (alpha - INTERVAL_RISK))

    # lb_check gives the practical risk to compare against such that for any particular (p0, p1),
    #    it is true w.h.p. that the chosen A*, B* will have a true FPR less than ALPHA_NOMINAL
    lb_check, _ = check_interval.calc_interval(constant_numerator / (n_trials))

    # Compute p_mid
    p_mid = compute_maxFPR_p(p0, p1)

    RESULTS_NULL = np.zeros((n_trials, Nmax, 2))
    RESULTS_ALT = np.zeros((n_trials, Nmax, 2))

    for k in tqdm(range(n_trials)):
        RESULTS_NULL[k, :, :] = get_max_min_along_trajectory(
            np.random.binomial(1, p_mid, size=(Nmax,)),
            np.random.binomial(1, p_mid, size=(Nmax,)),
            p0,
            p1,
            p_mid,
        )

        results_alt = get_max_min_along_trajectory(
            np.random.binomial(1, p0, size=(Nmax,)),
            np.random.binomial(1, p1, size=(Nmax,)),
            p0,
            p1,
            p_mid,
        )

        RESULTS_ALT[k, :, 0] = 1.0 / results_alt[:, 1]
        RESULTS_ALT[k, :, 1] = 1.0 / results_alt[:, 0]

    MAX_ALT = np.sort(np.max(RESULTS_ALT[:, :, 1], axis=1))
    MAX_NULL = np.sort(np.max(RESULTS_NULL[:, :, 1], axis=1))

    critical_idx_montecarlo = int(np.floor(n_trials * (1.0 - lb_check) + 1))
    critical_idx_low_montecarlo = int(np.floor(n_trials * (max_power_nominal) + 1))
    if critical_idx_low_montecarlo >= n_trials - 1:
        critical_idx_low_montecarlo = int(n_trials - 1)

    B_STAR = MAX_NULL[critical_idx_montecarlo]
    A_STAR_INVERSE = MAX_ALT[critical_idx_low_montecarlo]

    current_FPR, current_FNR = evaluate_true_errors(
        RESULTS_NULL, RESULTS_ALT, A_STAR_INVERSE, B_STAR
    )

    try:
        assert current_FPR <= lb_check
    except:
        while current_FPR > lb_check:
            estimated_gap = int(np.floor((current_FPR - lb_check) * n_trials) + 2)
            critical_idx_montecarlo += estimated_gap
            B_STAR = MAX_NULL[critical_idx_montecarlo]
            current_FPR, current_FNR = evaluate_true_errors(
                RESULTS_NULL, RESULTS_ALT, A_STAR_INVERSE, B_STAR
            )

    print("Beginning while loop!")

    counter_outer = 0
    while (
        current_FPR < (lb_check - (2.0 / n_trials))
        or current_FNR < (1.0 - max_power_nominal) - (1.0 / n_trials)
    ) and counter_outer < 1000:
        counter_outer += 1
        critical_idx_montecarlo -= int(
            np.floor((lb_check - current_FPR) * n_trials) + 1
        )
        B_STAR = MAX_NULL[critical_idx_montecarlo]
        current_FPR, current_FNR = evaluate_true_errors(
            RESULTS_NULL, RESULTS_ALT, A_STAR_INVERSE, B_STAR
        )

        while current_FNR < (1.0 - max_power_nominal) - (1.0 / n_trials):
            critical_idx_low_montecarlo -= 2
            A_STAR_INVERSE = MAX_ALT[critical_idx_low_montecarlo]
            current_FPR, current_FNR = evaluate_true_errors(
                RESULTS_NULL, RESULTS_ALT, A_STAR_INVERSE, B_STAR
            )

    print()
    print("Finished after " + str(counter_outer) + " iterations!")

    A_Star = 1.0 / A_STAR_INVERSE
    B_Star = B_STAR

    return [A_Star, B_Star]


def get_max_min_along_trajectory(sequence_0, sequence_1, p0, p1, p_mid):
    # Determine how many steps are in the trajectory, and verify that the test is valid (Nmax >= T)
    T = len(sequence_0)

    # Store min and max at all times
    return_data = np.ones((T, 2))

    state = 1.0

    # Perform the test through time T
    for t in range(T):
        null_multiplier = 1.0
        if sequence_0[t] >= 0.5:
            # base policy had success
            null_multiplier = p0 / p_mid
        else:
            # base policy had failure
            null_multiplier = (1.0 - p0) / (1.0 - p_mid)

        alt_multiplier = 1.0
        if sequence_1[t] >= 0.5:
            # new policy had success
            alt_multiplier = p1 / p_mid
        else:
            # new policy had failure
            alt_multiplier = (1.0 - p1) / (1.0 - p_mid)

        state *= null_multiplier
        state *= alt_multiplier

        if state <= return_data[t, 0]:
            return_data[t, 0] = state
        else:
            try:
                return_data[t, 0] = return_data[t - 1, 0]
            except:  # t = 0
                pass

        if state >= return_data[t, 1]:
            return_data[t, 1] = state
        else:
            try:
                return_data[t, 1] = return_data[t - 1, 1]
            except:  # t = 0
                pass

    return return_data
