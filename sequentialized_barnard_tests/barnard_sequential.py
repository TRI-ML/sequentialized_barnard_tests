"""Module implementing a rectified (valid) sequential Barnard procedure.

This function incorporates the base (batch) Barnard procedure into a sequentialized
method using the Bonferroni (union bound) correction.
"""

import warnings
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import barnard_exact

from sequentialized_barnard_tests.base import (
    Decision,
    Hypothesis,
    MirroredTestMixin,
    SequentialTestBase,
    TestResult,
)
from sequentialized_barnard_tests.batch import (
    BarnardExactTest,
    MirroredBarnardExactTest,
)


class BarnardSequentialTest(SequentialTestBase):
    """_summary_

    Args:
        SequentialTestBase (_type_): _description_
    """

    def __init__(
        self,
        alternative: Hypothesis,
        alpha: float,
        n_max: int,
        times_of_evaluation: ArrayLike,
        weights_of_evaluation: Optional[ArrayLike] = None,
    ) -> None:
        """_summary_

        Args:
            alternative (Hypothesis): _description_
            alpha (float): _description_
            n_max (int): _description_
            times_of_evaluation (ArrayLike): _description_
            weights_of_evaluation (Optional[ArrayLike], optional): _description_. Defaults to None.

        Raises:
            ValueError: If alpha, n_max, times_of_evaluation are invalid
        """
        # Handle inputs and check for errors in test attributes
        try:
            assert 0.0 < alpha and alpha < 1.0
            assert n_max >= 1
        except:
            raise ValueError(
                "Invalid inputs: alpha must be in (0., 1.) and n_max must be >= 1"
            )

        # Assign test attributes
        self.alpha = alpha
        self.n_max = n_max
        self.alternative = alternative

        # Handle inputs and check for errors in evaluation protocol attributes
        try:
            assert len(times_of_evaluation) >= 1
            evaluation_times_is_array = True
        except:
            try:
                assert times_of_evaluation >= 1
            except:
                raise ValueError("Scalar times_of_evaluation must be postive integer!")

            evaluation_times_is_array = False

        # Handle additional error cases and assign times_of_evaluation and n_evaluations
        if evaluation_times_is_array:
            times_of_evaluation = np.sort(times_of_evaluation)
            try:
                assert np.min(times_of_evaluation) >= 1
                assert np.min(np.diff(times_of_evaluation)) >= 1
            except:
                raise ValueError(
                    "If an array, times_of_evaluation must be positive and strictly increasing"
                )

            if np.max(times_of_evaluation) <= self.n_max:
                # All values acceptable
                self.times_of_evaluation = times_of_evaluation
                self.n_evaluations = len(self.times_of_evaluation)
            else:
                # Can only accept up to self.n_max
                self.times_of_evaluation = np.where(
                    times_of_evaluation <= self.n_max, times_of_evaluation
                )
                self.n_evaluations = len(self.times_of_evaluation)
        else:
            # Make scalar into an explicit array of times to sample
            self.times_of_evaluation = np.arange(0, n_max, times_of_evaluation)[1:]
            self.n_evaluations = len(self.times_of_evaluation)

        # Given n_evaluations and times_of_evaluation, check the
        # weighting of risk and handle errors
        if weights_of_evaluation is None:
            weights_of_evaluation = np.ones(self.n_evaluations) / self.n_evaluations
        elif len(weights_of_evaluation) < self.n_evaluations:
            warnings.warn(
                "Weights of each test is too short. Reverting to uniform risk budget."
            )
            weights_of_evaluation = np.ones(self.n_evaluations) / self.n_evaluations
        elif len(weights_of_evaluation) == self.n_evaluations:
            if np.min(weights_of_evaluation) > 0:
                weights_of_evaluation *= self.alpha / sum(weights_of_evaluation)
            else:
                warnings.warn(
                    "Weights contain negative elements; unclear how to rectify. Reverting to uniform risk budget."
                )
                weights_of_evaluation = np.ones(self.n_evaluations) / self.n_evaluations
                # weights_of_evaluation += np.min(weights_of_evaluation)
                # weights_of_evaluation *= self.alpha / sum(weights_of_evaluation)
        else:
            warnings.warn(
                f"Weights of each test is too long. Taking only the first self.n_evaluations (here, {self.n_evaluations}) components."
            )
            weights_of_evaluation = weights_of_evaluation[: self.n_evaluations]
            assert len(weights_of_evaluation) == self.n_evaluations
            if np.min(weights_of_evaluation) > 0:
                weights_of_evaluation *= self.alpha / sum(weights_of_evaluation)
            else:
                warnings.warn(
                    "Weights contain negative elements; unclear how to rectify. Reverting to uniform risk budget."
                )
                weights_of_evaluation = np.ones(self.n_evaluations) / self.n_evaluations

        # Assign the risk weighting
        self.weights_of_evaluation = weights_of_evaluation

        # Assign state variables
        self.t = None
        self.sequence = None
        self.reset()

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: Optional[bool] = False,
    ) -> TestResult:
        """_summary_

        Args:
            datum_0 (_type_): _description_
            datum_1 (_type_): _description_
            verbose (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            TestResult: _description_
        """
        is_bernoulli_0 = datum_0 in [0, 1]
        is_bernoulli_1 = datum_1 in [0, 1]
        if not (is_bernoulli_0 and is_bernoulli_1):
            raise (ValueError("Input data are not interpretable as Bernoulli."))
        if verbose:
            print(
                (
                    "Update the Barnard sequential process given new "
                    f"datum_0 == {datum_0} and datum_1 == {datum_1}."
                )
            )

        # Update state (i.e., time)
        self.sequence[self.t, 0] = datum_0
        self.sequence[self.t, 1] = datum_1
        self.t += 1

        # If time matches an element of 'times_of_evaluation,' run a batch test
        run_test = False
        if np.min(np.abs(self.times_of_evaluation - self.t)) < 0.5:
            run_test = True
            idx = np.argmin(np.abs(self.times_of_evaluation - self.t))

        if run_test:
            batch_test = BarnardExactTest(
                self.alternative, self.weights_of_evaluation[idx]
            )
            if self.t < self.n_max:
                result = batch_test.run_on_sequence(
                    self.sequence[: self.t, 0], self.sequence[: self.t, 1]
                )
            else:
                result = batch_test.run_on_sequence(
                    self.sequence[:, 0], self.sequence[:, 1]
                )

            # Append time of decision
            result.info["Time"] = self.t

            return result
        else:
            decision = Decision.FailToDecide
            info = {"Time": self.t}

            result = TestResult(decision, info)
            return result

    def reset(self, verbose: Optional[bool] = False) -> None:
        """_summary_

        Args:
            verbose (Optional[bool], optional): _description_. Defaults to False.
        """
        self.t = int(0)
        self.sequence = np.zeros((self.n_max, 2))


class MirroredBarnardSequentialTest(MirroredTestMixin, SequentialTestBase):
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

    _base_class = BarnardSequentialTest

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
        self._test_for_alternative.reset()
        self._test_for_null.reset()

        if not (len(sequence_0) == len(sequence_1)):
            raise (ValueError("The two input sequences must have the same size."))

        # Solve the problem of reaching the end of the sequence by
        # assigning result to FailToDecide by default
        result = TestResult(decision=Decision.FailToDecide)

        # Run through sequence until a decision is reached or sequence is
        # fully completed.
        for idx in range(len(sequence_0)):
            result_for_alternative, result_for_null = self.step(
                sequence_0[idx], sequence_1[idx]
            )

            # result_for_alternative = self._test_for_alternative.step(
            #     sequence_0[idx], sequence_1[idx], *args, **kwargs
            # )
            # result_for_null = self._test_for_null.step(
            #     sequence_0[idx], sequence_1[idx], *args, **kwargs
            # )

            # Store the info (constituent test results)
            info = {
                "result_for_alternative": result_for_alternative,
                "result_for_null": result_for_null,
            }

            # Assign decision variable
            if (not result_for_alternative.decision == Decision.FailToDecide) and (
                result_for_null.decision == Decision.FailToDecide
            ):
                # If only result_for_alternative, then overall decision is Reject Null / Accept Alternative
                decision = Decision.AcceptAlternative
            elif (not result_for_null.decision == Decision.FailToDecide) and (
                result_for_alternative.decision == Decision.FailToDecide
            ):
                # If only result_for_null, then overall decision is Accept Null / Reject Alternative
                decision = Decision.AcceptNull
            else:
                # Neither (or both) significant: decision is Decision.FailToDecide
                decision = Decision.FailToDecide

            if not decision == Decision.FailToDecide:
                break

        # Assign result
        result = TestResult(decision, info)
        return result

    def step(
        self, datum_0, datum_1, verbose: Optional[bool] = False
    ) -> Tuple[TestResult, TestResult]:
        """_summary_

        Args:
            datum_0 (_type_): _description_
            datum_1 (_type_): _description_
            verbose (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            Tuple[TestResult, TestResult]: _description_
        """
        result_for_alternative = self._test_for_alternative.step(
            datum_0, datum_1, verbose
        )

        result_for_null = self._test_for_null.step(datum_0, datum_1, verbose)

        return [result_for_alternative, result_for_null]
