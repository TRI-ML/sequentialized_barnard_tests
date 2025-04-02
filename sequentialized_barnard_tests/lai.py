"""Sequential method based on asymptotic solution to optimal stopping PDE

This module defines the sequential test for the 2x2 Bernoulli contingency table as a special
case of the analysis presented in Lai (1988), "Nearly Optimal Sequential Tests of Composite
Hypotheses" (Annals of Statistics 16(2): 856-886 [June, 1988]).
DOI: 10.1214/aos/1176350840
URL: https://projecteuclid.org/journals/annals-of-statistics/volume-16/issue-2/Nearly-Optimal-Sequential-Tests-of-Composite-Hypotheses/10.1214/aos/1176350840.full
"""

import warnings
from typing import Union

import numpy as np
from scipy import stats
from tqdm import tqdm

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
        n_max (int): Maximal allowed length of the test sequence. Must be greater than 0.
        c (float): Regularized cost calibrated to (n_max, alpha). Must lie in (0., 1.)
    """

    def __init__(
        self,
        alternative: Hypothesis,
        n_max: int,
        alpha: float,
        minimum_gap: float = 0.0,
        calibrate_regularizer: bool = False,
        default_c: float = 4.3321e-05,
        n_calibration_sequences: int = 10000,
        calibration_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """Initializes the test object.

        Args:
            alternative (Hypothesis): Specification of the alternative hypothesis.
            n_max (int): Maximal sequence length. Must be greater than 0.
            alpha (float): Significance level of the test. Must lie in (0., 1.)
            minimum_gap (float, optional): Minimal gap in the alternative
                space. Nonnegative. Defaults to 0.0 (robust solution).
            calibrate_regularizer (bool, optional): Toggle whether to calibrate the
                value of c or set a default value of default_c. Defaults to False.
            default_c (float, optional): If calibrate_regularizer is False, self.c is
                set to default_c. Defaults to 4.3321e-05, which is tuned for
                n_max = 500, alpha = 0.05, and minimum_gap = 0.0.
            n_calibration_sequences (int, optional): If calibrate_regularizer is True,
                self.c is tuned via calibration, where the number of sequences equals
                num_calibration_sequences. Defaults to 10000.
            calibration_seed (int, optional): If calibrating the Lai procedure,
                use this seed in order to generate the sequences. Defaults to 42.
            verbose (bool, optional): If True, print the outputs to stdout.
                Defaults to False.

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

        # Initialize Lai uniparameter test attributes
        self._zeta = calculate_robust_zeta(minimum_gap)
        self._theta_0 = 0.5
        self._theta_1 = 0.5 + self._zeta

        # Initialize Lai procedure optimization regularizers
        self.c = None
        self._gamma = None  # Note that self._gamma is updated automatically whenever self.c is updated.

        # Assign values to these regularizers either via calibration or by explicitly setting c.
        if calibrate_regularizer:
            self.calibrate_c(
                n_calibration_sequences=n_calibration_sequences,
                seed=calibration_seed,
                verbose=verbose,
            )
        else:
            # Calibrated to alpha = 0.05, n_max = 500, minimum_gap = 0.0
            self.set_c(default_c, verbose)

        # Run the reset method by default
        self._state = None
        self._t = None
        self._current_decision = None
        self.reset(verbose)

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
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
        self._t += 1
        if self._t > self.n_max:
            warnings.warn(
                "Have exceeded the allowed number of evals; not updating internal states."
            )
            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

        # Update state {#1s, (#1s + #-1s), #-1s}
        if np.isclose(datum_1, datum_0):
            # Got a zero, so no update
            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

        else:
            eval_result = self._update_state_and_run_lai(datum_0, datum_1)

            if eval_result > 0.5:
                # Reject Null and Accept Alternative
                self._current_decision = Decision.AcceptAlternative
            else:
                self._current_decision = Decision.FailToDecide

            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

    def reset(
        self,
        verbose: bool = False,
    ) -> None:
        """Resets the underlying Lai process.

        Args:
            verbose (bool, optional): If True, print the outputs to stdout.
                Defaults to False.
        """
        self._state = np.zeros(3)
        self._t = int(0)
        self._current_decision = Decision.FailToDecide

        if self.alternative == Hypothesis.P0MoreThanP1:
            if verbose:
                print("    Null:        P0 <= P1")
                print("    Alternative: P0 >  P1")
        elif self.alternative == Hypothesis.P0LessThanP1:
            if verbose:
                print("    Null:        P0 >= P1")
                print("    Alternative: P0 <  P1")

    def set_c(self, new_c: float, verbose: bool = False) -> None:
        """Set the optimization regularizer c in (0., 1.) to a user-specified value.
        Doing so updates the derived parameter (gamma) in the optimization schema.

        Args:
            new_c (float): Optimization regularizer. Lies in (0., 1.)
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        try:
            assert new_c > 0.0 and new_c < 1.0
        except:
            raise ValueError("Regularizer c must be in (0., 1.)")
        if verbose:
            print(f"Setting the regularizer term to {new_c}.")
        self.c = new_c
        self._gamma = calculate_gamma(self._theta_0, self._theta_1, self.c)

    def calibrate_c(
        self,
        n_calibration_sequences: int = 10000,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """Set the optimization regularizer c in (0., 1.) using Monte Carlo estimation
        and a high-probability upper bound. Doing so updates the derived parameter
        (gamma) in the optimization schema.

        Args:
            n_calibration_sequences (int, optional): Number of Monte Carlo
                sequences. Must be greater than 100. Defaults to 10000.
            seed (int, optional): Seed for the numpy random Generator object, from which
                the calibration sequences are drawn. This ensures reproducibility.
                Defaults to 42.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        Raise:
            ValueError: If the number of calibration sequences does not exceed 100.
        """
        try:
            assert n_calibration_sequences >= 100
        except:
            raise ValueError(
                "Insufficient calibration sequences. Must have at least 100."
            )
        if verbose:
            print("Calibrating the regularizer term using Monte Carlo sampling.")
        # Define the Generator object
        rng = np.random.default_rng(seed=seed)

        # Generate calibration sequences
        sequences = rng.binomial(1, 0.5, size=(n_calibration_sequences, self.n_max, 2))

        # Track erroneous decisions
        erroneous_decisions = int(0)

        # High-probability lower bound error count
        target_error_count = stats.binom.ppf(
            self.calibration_correction,
            n_calibration_sequences,
            self.alpha - self.calibration_correction,
        )

        # Binary search to converge on c
        log_c_max = 0.0
        log_c_min = -16.0

        fpr_error = n_calibration_sequences
        # Add termination guarantee via tightness bound on log_c_error
        minimum_log_c_error = 1e-5
        maximum_number_of_iterations = int(
            np.floor(np.log((log_c_max - log_c_min) / minimum_log_c_error) + 1)
        )

        calibration_progress_bar = tqdm(total=maximum_number_of_iterations)
        while (
            np.abs(fpr_error) >= 1.5
            and np.abs(log_c_max - log_c_min) > minimum_log_c_error
        ):
            erroneous_decisions = int(0)
            # Update estimate of c
            log_c_mid = 0.5 * (log_c_max + log_c_min)
            c = np.exp(log_c_mid)
            self.set_c(c, verbose=False)

            for k in range(n_calibration_sequences):
                # Step through each and figure out FPR
                result = self.run_on_sequence(sequences[k, :, 0], sequences[k, :, 1])
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

            calibration_progress_bar.update(1)

        # Store in self.c and update self.gamma
        self.set_c(c, verbose=False)

    def _update_state_and_run_lai(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
    ) -> float:
        """Update the internal state and return the Lai eval result.

        Args:
            datum_0: Bernoulli datum from the first source.
            datum_1: Bernoulli datum from the second source.

        Returns:
            Output of `run_test_step_gamma_uniparameter`.
        """
        self._state[1] += 1
        if datum_1 > datum_0:  # Difference of 1
            self._state[0] += 1
        else:  # Difference of -1
            self._state[2] += 1

        if self.alternative == Hypothesis.P0LessThanP1:
            # Test statistic is #1s / (#-1s + #1s)
            statistic = self._state[0] / self._state[1]
        elif self.alternative == Hypothesis.P0MoreThanP1:
            # Test statistic is #-1s / (#-1s + #1s)
            statistic = self._state[2] / self._state[1]

        # Run general procedure using the statistic
        eval_result = run_test_step_gamma_uniparameter(
            self.c,
            self._state[1],
            statistic,
            self._theta_0,
            gamma=self._gamma,
        )
        return eval_result


class MirroredLaiTest(LaiTest):
    """Mirrored Lai Test for comparing two Bernoulli distribiutions (2x2 Contingency
    Table).

    This class defines the mirrored version of the Lai Test, which has the ability to
    accept the Null Hypothesis in addition to reject it. Otherwise it is the same as
    LaiTest.

    Attributes:
        alternative (Hypothesis): Specification of the alternative hypothesis.
        alpha (float): Significance level of the test, lies in (0., 1.).
        calibration_correction (float): Significance level of calibration, lies in (0., alpha).
        minimum_gap (float): Minimal allowed gap, >= 0. Defaults to 0.
        n_max (int): Maximal allowed length of the test sequence. Must be greater than 0.
        c (float): Regularized cost calibrated to (n_max, alpha). Must lie in (0., 1.)
    """

    def __init__(
        self,
        alternative: Hypothesis,
        n_max: int,
        alpha: float,
        minimum_gap: float = 0.0,
        calibrate_regularizer: bool = False,
        default_c: float = 4.3321e-05,
        n_calibration_sequences: int = 10000,
        calibration_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """Initializes the test object.

        Args:
            alternative (Hypothesis): Specification of the alternative hypothesis.
            n_max (int): Maximal sequence length. Must be greater than 0.
            alpha (float): Significance level of the test. Must lie in (0., 1.)
            minimum_gap (float, optional): Minimal gap in the alternative
                space. Nonnegative. Defaults to 0.0 (robust solution).
            calibrate_regularizer (bool, optional): Toggle whether to calibrate the
                value of c or set a default value of default_c. Defaults to False.
            default_c (float, optional): If calibrate_regularizer is False, self.c is
                set to default_c. Defaults to 4.3321e-05, which is tuned for
                n_max = 500, alpha = 0.05, and minimum_gap = 0.0.
            n_calibration_sequences (int, optional): If calibrate_regularizer is True,
                self.c is tuned via calibration, where the number of sequences equals
                num_calibration_sequences. Defaults to 10000.
            calibration_seed (int, optional): If calibrating the Lai procedure,
                use this seed in order to generate the sequences. Defaults to 42.
            verbose (bool, optional): If True, print the outputs to stdout.
                Defaults to False.

        Raise:
            ValueError: If the inputs are invalid.
        """

        super().__init__(
            alternative,
            n_max,
            alpha,
            minimum_gap,
            calibrate_regularizer,
            default_c,
            n_calibration_sequences,
            calibration_seed,
            verbose,
        )

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
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
        self._t += 1
        if self._t > self.n_max:
            warnings.warn(
                "Have exceeded the allowed number of evals; not updating internal states."
            )
            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

        # Update state {#1s, (#1s + #-1s), #-1s}
        if np.isclose(datum_1, datum_0):
            # Got a zero, so no update
            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

        else:
            eval_result = self._update_state_and_run_lai(datum_0, datum_1)

            if eval_result > 0.5:
                # Reject Null and Accept Alternative
                self._current_decision = Decision.AcceptAlternative
            elif eval_result < -0.5:
                self._current_decision = Decision.AcceptNull
            else:
                self._current_decision = Decision.FailToDecide

            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result
