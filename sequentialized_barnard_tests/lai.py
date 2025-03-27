""" Sequential method based on asymptotic solution to optimal stopping PDE

This module defines the sequential test for the 2x2 Bernoulli contingency table as a special
case of the analysis presented in Lai (1988), "Nearly Optimal Sequential Tests of Composite
Hypotheses" (Annals of Statistics 16(2): 856-886 [June, 1988]).
DOI: 10.1214/aos/1176350840
URL: https://projecteuclid.org/journals/annals-of-statistics/volume-16/issue-2/Nearly-Optimal-Sequential-Tests-of-Composite-Hypotheses/10.1214/aos/1176350840.full
"""

from typing import Optional, Union

import numpy as np
from scipy import stats

from sequentialized_barnard_tests.base import (
    Decision,
    Hypothesis,
    SequentialTestBase,
    TestResult,
)
from sequentialized_barnard_tests.utils.utils_lai import (
    calculate_gamma,
    calculate_robust_zeta,
    run_test_step_gamma_uniparameter,
)


class LaiTest(SequentialTestBase):
    """Lai test for comparing two Bernoulli distributions (2x2 Contingency Table).

    This class defines the sequential test for the 2x2 Bernoulli contingency table as a special
    case of the analysis presented in Lai (1988), "Nearly Optimal Sequential Tests of Composite
    Hypotheses" (Annals of Statistics 16(2): 856-886 [June, 1988]).

    Attributes:
        alternative (Hypothesis): Specification of the alternative hypothesis.
        alpha (float): Significance level of the test, lies in (0., 1.).
        calibration_correction (float): Significance level of calibration, lies in (0., alpha).
        minimum_gap (float): Minimal allowed gap, >= 0. Defaults to 0.
        n_max (int): Maximal allowed length of the test trajectory. Must be greater than 0.
        c (float): Regularized cost calibrated to (n_max, alpha). Must lie in (0., 1.)

    """

    def __init__(
        self,
        n_max: int,
        alpha: float,
        alternative: Hypothesis,
        minimum_gap: Optional[float] = 0.0,
        verbose: Optional[bool] = False,
    ) -> None:
        """Initializes the test object.

        Args:
            n_max (int): Maximal trajectory length. Must be greater than 0.
            alpha (float): Significance level of the test. Must lie in (0., 1.)
            alternative (Hypothesis): Specification of the alternative hypothesis.
            minimum_gap (Optional[float], optional): Minimal gap in the alternative space. Nonnegative. Defaults to 0.0 (robust solution).
            verbose (Optional[bool], optional): If True, print the outputs to stdout. Defaults to False.

        Raise:
            ValueError: If the inputs are invalid.
        """

        # Handle erroneous inputs
        try:
            assert n_max > 0
            assert 0.0 < alpha < 1.0
            assert minimum_gap >= 0.0 and minimum_gap <= 1.0
        except:
            raise ValueError(
                "Invalid inputs: MUST HAVE n_max > 0, alpha in (0., 1.), minimum_gap in [0., 1.]"
            )
        # Assign attributes
        self.n_max = n_max
        self.alpha = alpha
        self.alternative = alternative
        self.minimum_gap = minimum_gap

        # Assign derived attributes
        self.calibration_correction = np.minimum(alpha / 50.0, 1e-3)

        # Initialize Lai procedure optimization regularizers
        self.c = None
        self.gamma = None

        # Initialize Lai uniparameter test attributes
        self.zeta = calculate_robust_zeta(minimum_gap)
        self.theta_0 = 0.5
        self.theta_1 = 0.5 + self.zeta

        # Run the reset method by default
        self.reset(verbose)

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: Optional[bool] = False,
    ) -> TestResult:
        """Runs the test procedure on a single pair of Bernoulli data.

        Args:
            datum_0: Bernoulli datum from the first source.
            datum_1: Bernoulli datum from the second source.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Returns:
            TestResult: Result of the hypothesis test.

        Raise:
            ValueError: If the input data take non-Bernoulli values.
        """
        is_bernoulli_0 = datum_0 in [0, 1]
        is_bernoulli_1 = datum_1 in [0, 1]
        if not (is_bernoulli_0 and is_bernoulli_1):
            raise (ValueError("Input data are not interpretable as Bernoulli."))
        if verbose:
            print(
                (
                    "Update the Lai process given new "
                    f"datum_0 == {datum_0} and datum_1 == {datum_1}."
                )
            )
        # Iterate time (total number of pairs seen)
        self.t += 1
        if self.t > self.n_max:
            print("Have exceeded the allowed number of evals; returning FailToDecide")
            decision = Decision.FailToDecide
            info = {"Time": self.t, "State": self.state}
            result = TestResult(decision, info)

            return result

        # Update state {#1s, (#1s + #-1s), #-1s}
        if np.isclose(datum_1, datum_0):
            # Got a zero, so no update
            decision = Decision.FailToDecide
            info = {"Time": self.t, "State": self.state}
            result = TestResult(decision, info)

            return result

        else:
            # Received either 1 or -1
            self.state[1] += 1
            if datum_1 > datum_0:  # Difference of 1
                self.state[0] += 1
            else:  # Difference of -1
                self.state[2] += 1

            if self.alternative == Hypothesis.P0LessThanP1:
                # Test statistic is #1s / (#-1s + #1s)
                statistic = self.state[0] / self.state[1]
            elif self.alternative == Hypothesis.P0MoreThanP1:
                # Test statistic is #-1s / (#-1s + #1s)
                statistic = self.state[2] / self.state[1]

            # Run general procedure using the statistic
            eval_result = run_test_step_gamma_uniparameter(
                self.c,
                self.state[1],
                statistic,
                self.theta_0,
                gamma=self.gamma,
            )

            if eval_result > 0.5:
                # Reject Null and Accept Alternative
                decision = Decision.AcceptAlternative
            elif eval_result < -0.5:
                decision = Decision.AcceptNull
            else:
                decision = Decision.FailToDecide

            info = {"Time": self.t, "State": self.state}
            result = TestResult(decision, info)

            return result

    def reset(
        self,
        verbose: Optional[bool] = False,
    ) -> None:
        """Resets the underlying Lai process.

        Args:
            verbose (Optional[bool], optional): If True, print the outputs to stdout. Defaults to False.
        """
        self.state = np.zeros(3)
        self.t = int(0)

        if self.alternative == Hypothesis.P0MoreThanP1:
            if verbose:
                print("    Null:        P0 <= P1")
                print("    Alternative: P0 >  P1")
        elif self.alternative == Hypothesis.P0LessThanP1:
            if verbose:
                print("    Null:        P0 >= P1")
                print("    Alternative: P0 <  P1")

    def set_c(self, new_c: float) -> None:
        """Set the optimization regularizer c in (0., 1.) to a user-specified value. Doing
        so updates the derived parameter (gamma) in the optimization schema.

        Args:
            new_c (float): Optimization regularizer. Lies in (0., 1.)
        """
        try:
            assert new_c > 0.0 and new_c < 1.0
        except:
            raise ValueError("Invalid value of c")

        self.c = new_c
        self.gamma = calculate_gamma(self.theta_0, self.theta_1, self.c)

    def calibrate_c(self, n_calibration_trajectories: Optional[int] = 10000) -> None:
        """Set the optimization regularizer c in (0., 1.) using Monte Carlo estimation and a
        high-probability upper bound. Doing so updates the derived parameter (gamma) in the
        optimization schema.

        Args:
            n_calibration_trajectories (Optional[int], optional): Number of Monte Carlo trajectories. Must be greater than 100. Defaults to 10000.

        Raise:
            ValueError: If the number of calibration trajectories does not exceed 100.
        """
        try:
            assert n_calibration_trajectories >= 100
        except:
            raise ValueError(
                "Insufficient calibration trajectories. Must have at least 100."
            )
        # Generate calibration trajectories
        trajectories = np.random.binomial(
            1, 0.5, size=(n_calibration_trajectories, self.n_max, 2)
        )

        # Track erroneous decisions
        erroneous_decisions = int(0)

        # High-probability lower bound error count
        target_error_count = stats.binom.ppf(
            self.calibration_correction,
            n_calibration_trajectories,
            self.alpha - self.calibration_correction,
        )

        # Binary search to converge on c
        log_c_max = 0.0
        log_c_min = -16.0

        fpr_error = n_calibration_trajectories
        while np.abs(fpr_error) >= 1.5:
            erroneous_decisions = int(0)
            # Update estimate of c
            log_c_mid = 0.5 * (log_c_max + log_c_min)
            c = np.exp(log_c_mid)
            self.set_c(c)

            for k in range(n_calibration_trajectories):
                # Step through each and figure out FPR
                result = self.run_on_sequence(
                    trajectories[k, :, 0], trajectories[k, :, 1]
                )
                if result.decision == Decision.AcceptAlternative:
                    erroneous_decisions += 1

            fpr_error = erroneous_decisions - target_error_count

            if fpr_error >= 1.5:
                # Empirical FPR is too high --> reduce c
                log_c_max = log_c_mid
            elif fpr_error <= -1.5:
                # Empirical FPR is too low --> increase c
                log_c_min = log_c_mid
            else:
                break

        # Store in self.c and update self.gamma
        self.set_c(c)
