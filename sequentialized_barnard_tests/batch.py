"""Batch tests.

This module defines batch methods for hypothesis testing.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import barnard_exact

from sequentialized_barnard_tests.base import (
    Decision,
    Hypothesis,
    MirroredTestMixin,
    TestBase,
    TestResult,
)


class BarnardExactTest(TestBase):
    """Barnard's exact test.

    This class is a wrapper around scipy's implementation of Barnard's exact test.
    For more details, refer to scipy's documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html

    Attributes:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
    """

    def __init__(self, alternative: Hypothesis, alpha: float) -> None:
        """Initializes the test object.

        Args:
            alternative: Specification of the alternative hypothesis.
            alpha: Significance level of the test.
        """
        self.alternative = alternative
        self.alpha = alpha

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


class MirroredBarnardExactTest(MirroredTestMixin, TestBase):
    """A pair of one-sided Barnard's exact tests with mirrored alternatives.

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
    """

    _base_class = BarnardExactTest

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
