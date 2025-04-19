"""Sequential method based on Safe Any-time Valid Inference (SAVI)

This module defines the sequential test for 2x2 Bernoulli contingency tables as it
appears in Section 2 of 'Exact Anytime-Valid Confidence Intervals for Contingency Tables
and Beyond' by R. J. Turner and P. D. Grünwald (Statistics & Probability Letters 198,
2023).
"""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import bernoulli, beta

from sequentialized_barnard_tests.base import (
    Decision,
    Hypothesis,
    MirroredTestMixin,
    SequentialTestBase,
    TestResult,
)


class SaviTest(SequentialTestBase):
    """SAVI test for two-by-two contingency tables.

    This class defines the SAVI test for two-by-two contingency tables as it appears
    in Section 2 of 'Exact Anytime-Valid Confidence Intervals for Contingency Tables and
    Beyond' by R. J. Turner and P. D. Grünwald (Statistics & Probability Letters 198,
    2023).

    Attributes:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
        e_value: Current e-value.
        p_value: Current p-value.
    """

    def __init__(
        self,
        alternative: Hypothesis,
        alpha: float,
        verbose: bool = False,
    ) -> None:
        """Initializes the test object.

        Args:
            alternative: Specification of the alternative hypothesis.
            alpha: Significance level of the test.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        self.alternative = alternative
        self.alpha = alpha

        # Logarithm of the current e-value.
        self._log_e_value = None
        # Current p-value that is not clipped at 1.0.
        self._unclipped_p_value = None

        # Beta posterior for Bernoulli parameter P0.
        self._beta_posterior_0 = None
        # Beta posterior for Bernoulli parameter P1.
        self._beta_posterior_1 = None

        # Bayesian estimate of the Bernoulli parameter P0.
        self._estimated_p_0 = None
        # Bayesian estimate of the Bernoulli parameter P1.
        self._estimated_p_1 = None

        # Time state for decision information
        self._t = None

        self.reset(verbose)

    @property
    def e_value(self) -> float:
        """Current e-value of the process."""
        return np.exp(self._log_e_value).item()

    @property
    def p_value(self) -> float:
        """Current p-value of the process."""
        return np.clip(self._unclipped_p_value, 0.0, 1.0).item()

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        """Runs the test of a single pair of Bernoulli data.

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
                    "Update the SAVI process given new "
                    f"datum_0 == {datum_0} and datum_1 == {datum_1}."
                )
            )

        self._t += 1

        if (
            self.alternative == Hypothesis.P0MoreThanP1
            and self._estimated_p_0 <= self._estimated_p_1
        ) or (
            self.alternative == Hypothesis.P0LessThanP1
            and self._estimated_p_0 >= self._estimated_p_1
        ):
            # Theorem 2 in R. J. Turner and P. D. Grünwald (2023).
            optimal_p_0 = self._estimated_p_0
            optimal_p_1 = self._estimated_p_1
        else:
            # solution for equation (11) in R. J. Turner and P. D. Grünwald (2023).
            optimal_p_0 = (self._estimated_p_0 + self._estimated_p_1) / 2
            optimal_p_1 = optimal_p_0

        increment = (
            bernoulli(self._estimated_p_0).logpmf(datum_0)
            + bernoulli(self._estimated_p_1).logpmf(datum_1)
            - bernoulli(optimal_p_0).logpmf(datum_0)
            - bernoulli(optimal_p_1).logpmf(datum_1)
        )
        self._log_e_value += increment

        # Derive a p-value from the e-value following Proposition 12 in
        # `Admissible Anytime-Valid Sequential Inference Must Rely on Nonnegative
        # Martingales` by A. Ramdas et al. (arXiv, 2020).
        self._unclipped_p_value = np.minimum(self._unclipped_p_value, 1 / self.e_value)

        if verbose:
            print(f"  E-value: {self.e_value:.5f}")
            print(f"  P-value: {self.p_value:.5f}")

        # Finally, update the Beta posteriors and the estimates of the Bernoulli
        # parameters.
        a_0, b_0 = self._beta_posterior_0.args
        a_0 += datum_0
        b_0 += 1 - datum_0
        self._beta_posterior_0.args = (a_0, b_0)

        a_1, b_1 = self._beta_posterior_1.args
        a_1 += datum_1
        b_1 += 1 - datum_1
        self._beta_posterior_1.args = (a_1, b_1)

        self._estimate_parameters(verbose)

        # Compute test result.
        if self.p_value <= self.alpha:
            decision = Decision.AcceptAlternative
        else:
            decision = Decision.FailToDecide
        info = {"p_value": self.p_value, "e_value": self.e_value, "Time": self._t}
        result = TestResult(decision, info)

        return result

    def reset(self, verbose: bool = False) -> None:
        """Resets the underlying SAVI process.

        Args:
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        if verbose:
            print("Reset the SAVI process under the following hypotheses:")
        self._log_e_value = 0.0
        self._unclipped_p_value = 1.0
        self._t = int(0)
        if self.alternative == Hypothesis.P0MoreThanP1:
            if verbose:
                print("    Null:        P0 <= P1")
                print("    Alternative: P0 >  P1")
            self._beta_posterior_0 = beta(2, 1)
            self._beta_posterior_1 = beta(1, 2)
        elif self.alternative == Hypothesis.P0LessThanP1:
            if verbose:
                print("    Null:        P0 >= P1")
                print("    Alternative: P0 <  P1")
            self._beta_posterior_0 = beta(1, 2)
            self._beta_posterior_1 = beta(2, 1)
        self._estimate_parameters(verbose)
        if verbose:
            print(f"  E-value: {self.e_value:.5f}")
            print(f"  P-value: {self.p_value:.5f}")

    def _estimate_parameters(self, verbose: bool) -> None:
        if verbose:
            print(("  Estimate the Bernoulli parameters from the Beta " "posteriors:"))
        self._estimated_p_0 = self._beta_posterior_0.mean()
        self._estimated_p_1 = self._beta_posterior_1.mean()
        if verbose:
            print(
                "    Estimated Bernoulli parameters: "
                f"({self._estimated_p_0:.5f}, {self._estimated_p_1:.5f})"
            )


class OracleSaviTest(SaviTest):
    """SAVI test for two-by-two contingency tables with oracle knoweldge of the true
    Bernoulii parameters.

    See SaviTest for detailed descriptions of the test. This oracle is for evaluation of
    maximum power only and should not be used for practical applications.

    Attributs:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
        e_value: Current e-value.
        p_value: Current p-value.
        true_parameters: Knowledge of true Bernoulli parameters (P0, P1).
    """

    def __init__(
        self,
        alternative: Hypothesis,
        alpha: float,
        true_parameters: ArrayLike,
        verbose: bool = False,
    ) -> None:
        """Initializes the test object.

        Args:
            alternative: Specification of the alternative hypothesis.
            alpha: Significance level of the test.
            true_parameters: Knowledge of true Bernoulli parameters (P0, P1).
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Raises:
            ValueError: If true_parameters are not a pair of scalar values in [0, 1].
        """
        if not (
            len(true_parameters) == 2
            and (0.0 <= true_parameters[0] <= 1.0)
            and (0.0 <= true_parameters[1] <= 1.0)
        ):
            raise (
                ValueError("true_parameters have to be a pair of scalars in [0, 1].")
            )
        self.true_parameters = true_parameters
        super().__init__(alternative, alpha, verbose)

    def _estimate_parameters(self, verbose: bool) -> None:
        if verbose:
            print(
                (
                    "  Use the oracle knowledge of Bernoulli parameters for "
                    "the Bayesian estimates:"
                )
            )
        self._estimated_p_0 = self.true_parameters[0]
        self._estimated_p_1 = self.true_parameters[1]
        if verbose:
            print(
                "    Estimated Bernoulli parameters: "
                f"({self._estimated_p_0:.5f}, {self._estimated_p_1:.5f})"
            )


class MirroredSaviTest(MirroredTestMixin, SequentialTestBase):
    """A pair of one-sided SAVI tests with mirrored alternatives.

    In our terminology, a mirrored test is one that runs two one-sided tests
    simultaneously, with the null and the alternaive flipped from each other. This is so
    that it can yield either Decision.AcceptNull or Decision.AcceptAlternative depending
    on the input data, unlike standard one-sided tests that can never 'accept' the null.
    (Those standard tests will at most fail to reject the null, as represented by
    Decision.FailToDecide.)

    For example, if the alternative is Hypothesis.P0MoreThanP1 and the decision is
    Decision.AcceptNull, it should be interpreted as accepting Hypothesis.P0LessThanP1.

    The significance level alpha controls the following two errors simultaneously: (1)
    probability of wrongly accepting the alternative when the null is true, and (2)
    probability of wrongly accepting the null when the alternative is true. Note that
    Bonferroni correction is not needed since the null hypothesis for one test is the
    alternative for the other.

    Attributes:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
        e_value: Current e-value.
        p_value: Current p-value.
    """

    _base_class = SaviTest

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        """Runs the test of a single pair of Bernoulli data.

        Args:
            datum_0: Bernoulli datum from the first source.
            datum_1: Bernoulli datum from the second source.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Returns:
            TestResult: Result of the hypothesis test.
        """
        if verbose:
            print("Test for Alternative:")
        result_for_alternative = self._test_for_alternative.step(
            datum_0, datum_1, verbose
        )
        if verbose:
            print("Test for Null:")
        result_for_null = self._test_for_null.step(datum_0, datum_1, verbose)

        info = {
            "result_for_alternative": result_for_alternative,
            "result_for_null": result_for_null,
        }

        if (not result_for_alternative.decision == Decision.FailToDecide) and (
            result_for_null.decision == Decision.FailToDecide
        ):
            decision = Decision.AcceptAlternative
        elif (not result_for_null.decision == Decision.FailToDecide) and (
            result_for_alternative.decision == Decision.FailToDecide
        ):
            decision = Decision.AcceptNull
        else:
            decision = Decision.FailToDecide

        result = TestResult(decision, info)

        return result

    def reset(self, verbose: bool = False) -> None:
        """Resets the underlying SAVI process.

        Args:
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        if verbose:
            print("Test for Alternative:")
        self._test_for_alternative.reset(verbose)
        if verbose:
            print("Test for Null:")
        self._test_for_null.reset(verbose)


class MirroredOracleSaviTest(MirroredSaviTest):
    """A pair of one-sided oracle SAVI tests with mirrored alternatives.

    In our terminology, a mirrored test is one that runs two one-sided tests
    simultaneously, with the null and the alternaive flipped from each other. This is so
    that it can yield either Decision.AcceptNull or Decision.AcceptAlternative depending
    on the input data, unlike standard one-sided tests that can never 'accept' the null.
    (Those standard tests will at most fail to reject the null, as represented by
    Decision.FailToDecide.)

    For example, if the alternative is Hypothesis.P0MoreThanP1 and the decision is
    Decision.AcceptNull, it should be interpreted as accepting Hypothesis.P0LessThanP1.

    The significance level alpha controls the following two errors simultaneously: (1)
    probability of wrongly accepting the alternative when the null is true, and (2)
    probability of wrongly accepting the null when the alternative is true. Note that
    Bonferroni correction is not needed since the null hypothesis for one test is the
    alternative for the other.

    Attributes:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
        e_value: Current e-value.
        p_value: Current p-value.
        true_parameters: Knowledge of true Bernoulli parameters (P0, P1).
    """

    _base_class = OracleSaviTest
