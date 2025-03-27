"""Base class definitions.

This module defines base classes for hypothesis tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Type, Union

from numpy.typing import ArrayLike


class Decision(Enum):
    """Enum class to represent the decision of a hypothesis test."""

    AcceptNull = 0
    AcceptAlternative = 1
    FailToDecide = 2


@dataclass
class TestResult:
    """Data class defining the result of a hypothesis test.

    A result must contain a decision. Any auxiliary information will be thrown into
    info as an optional dictionary.
    """

    __test__ = False

    decision: Decision
    info: Optional[dict] = None


class TwoSampleBinomialHypothesis(Enum):
    """Enum class to represent the hypothesis of a test that compares parameters of two
    Bernoulli distributions.

    We assume we have two distributions Bernoulli(p_0) and Bernoulli(p_1). A pair of
    data is drawn independently from the two distributions, one at a time. The first
    hypothesis, `P0LessThanP1`, represents `p_0 < p_1`. The second hypothesis,
    `P0MoreThanP1`, represents `p_0 > p_1`. Note that both of these are possible
    alternative hypotheses in a statistical test, and the user will be tasked with
    choosing one as their alternative when instantiating a test.
    """

    P0LessThanP1 = 0
    P0MoreThanP1 = 1


Hypothesis = TwoSampleBinomialHypothesis


class TwoSampleTestBase(ABC):
    """Abstract base class for a two-sample hypothesis test."""

    @abstractmethod
    def run_on_sequence(
        self,
        sequence_0: ArrayLike,
        sequence_1: ArrayLike,
        *args,
        **kwargs,
    ) -> TestResult:
        """Runs the test on a pair of sequential data.

        Args:
            sequence_0: Sequence of data from the first source.
            sequence_1: Sequence of data from the second source.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).


        Returns:
            TestResult: Result of the hypothesis test.
        """
        pass


TestBase = TwoSampleTestBase


class SequentialTwoSampleTestBase(TwoSampleTestBase):
    """Base class for a family of sequential two-sample hypothesis tests."""

    def run_on_sequence(
        self,
        sequence_0: ArrayLike,
        sequence_1: ArrayLike,
        *args,
        **kwargs,
    ) -> TestResult:
        """Runs the test on a pair of sequential data.

        Args:
            sequence_0: Sequence of data from the first source.
            sequence_1: Sequence of data from the second source.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).

        Returns:
            TestResult: Result of the hypothesis test.

        Raises:
            ValueError: If the two sequences do not have the same length.
        """
        self.reset()
        if not (len(sequence_0) == len(sequence_1)):
            raise (ValueError("The two input sequences must have the same size."))

        result = TestResult(decision=Decision.FailToDecide)
        for idx in range(len(sequence_0)):
            result = self.step(sequence_0[idx], sequence_1[idx], *args, **kwargs)
            if not result.decision == Decision.FailToDecide:
                break
        return result

    def reset(self) -> None:
        """Resets internal memory states of the test."""
        pass

    @abstractmethod
    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        *args,
        **kwargs,
    ) -> TestResult:
        """Runs the test on a single pair of data.

        Args:
            datum_0: Datum from the first source.
            datum_1: Datum from the second source.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).

        Returns:
            TestResult: Result of the hypothesis test.
        """
        pass


SequentialTestBase = SequentialTwoSampleTestBase


class MirroredTestMixin:
    """A mixin class to define mirrored hypothesis tests.

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
        It has the same attributes as the underlying base tests.
    """

    _base_class: Type[Union[TestBase, SequentialTestBase]] = (
        None  # To be set by subclasses.
    )

    def __init__(self, alternative: Hypothesis, *args, **kwargs) -> None:
        """Initializes the MirroredTestMixin object.

        It instantiates two private member variables, _test_for_alternative and
        _test_for_null, with flipped alternative hypotheses.

        Args:
            alternative: Specification of the alternative hypothesis.
            *args: Additional positional arguments (if any).
            **kwargs: Additional optional or keyward arguments (if any).

        Raises:
            AttributeError: If alternative is not a valid Hypothesis for a mirrored test.
        """
        if alternative == Hypothesis.P0MoreThanP1:
            null = Hypothesis.P0LessThanP1
        elif alternative == Hypothesis.P0LessThanP1:
            null = Hypothesis.P0MoreThanP1
        else:
            raise (
                AttributeError(f"{alternative} is not a valid value for alternative.")
            )

        self._test_for_alternative = self._base_class(alternative, *args, **kwargs)
        self._test_for_null = self._base_class(null, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Dynamically forward attributes from the underlying base tests.

        Args:
            name: Name of the attribute.

        Raises:
            AttributeError: If the attribute does not exist in the base test class.
        """
        # Dynamically forward attributes from the base test.
        if hasattr(self._test_for_alternative, name):
            return getattr(self._test_for_alternative, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Dynamically set attributes to the underlying base tests.

        Args:
            name: Name of the attribute.
            value: Value of the attribute.

        Raises:
            AttributeError: If the attribute assignment to the base tests fails.
        """
        if name == "alternative":
            # For alternative, make sure _test_for_alternative and _test_for_null has
            # flipped alternative hypothesis.
            if value == Hypothesis.P0MoreThanP1:
                null = Hypothesis.P0LessThanP1
            elif value == Hypothesis.P0LessThanP1:
                null = Hypothesis.P0MoreThanP1
            else:
                raise (AttributeError(f"{value} is not a valid value for alternative."))
            self._test_for_alternative.alternative = value
            self._test_for_null.alternative = null
        else:
            # If setting attributes after initialization, set on both
            # _test_for_alternative and _test_for_null.
            if name not in ["_test_for_alternative", "_test_for_null", "_base_class"]:
                if hasattr(self._test_for_alternative, name):
                    setattr(self._test_for_alternative, name, value)
                    setattr(self._test_for_null, name, value)
                else:
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )
            else:
                super().__setattr__(name, value)
