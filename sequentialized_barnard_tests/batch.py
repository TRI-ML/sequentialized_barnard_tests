"""Batch tests.

This module defines batch methods for hypothesis testing.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import barnard_exact

from sequentialized_barnard_tests.base import Decision, Hypothesis, TestBase, TestResult


class BarnardExactTest(TestBase):
    """Barnard's exact test.

    This class is a wrapper around scipy's implementation of Barnard's exact test.
    For more details, refer to scipy's documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html

    Attributes:
        alpha: Significance level of the test.
        alternative: Specification of the alternative hypothesis.
    """

    def __init__(self, alpha: float, alternative: Hypothesis) -> None:
        """Initializes the test object.

        Args:
            alpha: Significance level of the test.
            alternative: Specification of the alternative hypothesis.
        """
        self.alpha = alpha
        self.alternative = alternative

    def run_on_sequence(
        self, sequence_0: ArrayLike, sequence_1: ArrayLike
    ) -> TestResult:
        """Runs the test on a pair of two Bernoulli sequences.

        Args:
            sequence_0: Sequence of Bernoulli data from the first source.
            sequence_1: Sequence of Bernoulli data from the second source.

        Returns:
            TestResult: Result of the hypothesis test.

        Raises:
            ValueError: If the input sequences are not Bernoulli data.
        """
        sequence_0_is_binary = np.all(
            (np.array(sequence_0) == 0) + (np.array(sequence_0) == 1)
        )
        sequence_1_is_binary = np.all(
            (np.array(sequence_1) == 0) + (np.array(sequence_1) == 1)
        )
        if not (sequence_0_is_binary and sequence_1_is_binary):
            raise (ValueError("Input sequences must be all Bernoulli data."))
        num_successes_0 = np.sum(sequence_0).item()
        num_failures_0 = len(sequence_0) - num_successes_0
        num_successes_1 = np.sum(sequence_1).item()
        num_failures_1 = len(sequence_1) - num_successes_1
        table = [[num_successes_0, num_successes_1], [num_failures_0, num_failures_1]]

        barnard = barnard_exact(
            table,
            alternative=(
                "less" if self.alternative == Hypothesis.P0LessThanP1 else "greater"
            ),
            pooled=(len(sequence_0) == len(sequence_1)),
        )
        if barnard.pvalue <= self.alpha:
            decision = Decision.AcceptAlternative
        else:
            decision = Decision.FailToDecide
        result = TestResult(
            decision,
            {"p_value": barnard.pvalue.item(), "statistic": barnard.statistic.item()},
        )
        return result


class MirroredBarnardExactTest(TestBase):
    """A pair of one-sided Barnard's exact tests with mirrored alternatives.

    It runs two tests, one with `alternative = Hypothesis.P0LessThanP1` and the
    other with `alternative = Hypothesis.P0MoreThanP1`. This is so that it can yield
    either Decision.AcceptNull or Decision.AcceptAlternative depending on the input
    data, unlike standard one-sided tests that can never 'accept' the null. (Those
    standard tests will at most fail to reject the null.) For example, if `alternative =
    Hypothesis.P0MoreThanP1` and the decision is Decision.AcceptNull, it should be
    interpreted as accepting Hypothesis.P0LessThanP1.

    The significance level alpha controls the following two errors simultaneously: (1)
    probability of wrongly accepting alternative when null is true, and (2) probability
    of wrongly accepting null when alternative is true.

    Attributes:
        alpha: Significance level of the test.
        alternative: Specification of the alternative hypothesis.
    """

    def __init__(self, alpha: float, alternative: Hypothesis) -> None:
        """Initializes the test object.

        Args:
            alpha: Significance level of the test.
            alternative: Specification of the alternative hypothesis.
        """
        self._alpha = alpha
        self._alternative = alternative
        if alternative == Hypothesis.P0MoreThanP1:
            null = Hypothesis.P0LessThanP1
        else:
            null = Hypothesis.P0MoreThanP1
        self._null = null

        # Bonferroni correction is not needed because the null hypothesis for one test
        # is the alternative hypothesis for the other in this mirrored setting.
        self._test_for_alternative = BarnardExactTest(
            alpha=alpha, alternative=alternative
        )
        self._test_for_null = BarnardExactTest(alpha=alpha, alternative=null)

    @property
    def alpha(self) -> float:
        """The significance level of the test."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value
        self._test_for_null.alpha = value
        self._test_for_alternative.alpha = value

    @property
    def alternative(self) -> Hypothesis:
        """Specification of the alternative hypothesis."""
        return self._alternative

    @alternative.setter
    def alternative(self, value: Hypothesis) -> None:
        self._alternative = value
        if value == Hypothesis.P0MoreThanP1:
            null = Hypothesis.P0LessThanP1
        else:
            null = Hypothesis.P0MoreThanP1
        self._null = null

        self._test_for_alternative.alternative = self._alternative
        self._test_for_null.alternative = self._null

    def run_on_sequence(
        self, sequence_0: ArrayLike, sequence_1: ArrayLike
    ) -> TestResult:
        """Runs the test on a pair of two Bernoulli sequences.

        Args:
            sequence_0: Sequence of Bernoulli data from the first source.
            sequence_1: Sequence of Bernoulli data from the second source.

        Returns:
            TestResult: Result of the hypothesis test.
        """
        result_for_alternative = self._test_for_alternative.run_on_sequence(
            sequence_0, sequence_1
        )
        result_for_null = self._test_for_null.run_on_sequence(sequence_0, sequence_1)

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
