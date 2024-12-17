"""Base class definitions.

This module defines base classes for hypothesis tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

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
    `P0MoreThanP1`, represents `p_0 > p_1`.
    """

    P0LessThanP1 = 0
    P0MoreThanP1 = 1


Hypothesis = TwoSampleBinomialHypothesis


class TwoSampleTestBase(ABC):
    """Abstract base class for a two-sample hypothesis test."""

    @abstractmethod
    def run_on_sequence(
        self, sequence_0: ArrayLike, sequence_1: ArrayLike
    ) -> TestResult:
        """Runs the test on a pair of sequential data.

        Args:
            sequence_0: Sequence of data from the first source.
            sequence_1: Sequence of data from the second source.

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
            args: Additional positional arguments.
            **kwargs: Additional optional or keyward arguments.

        Returns:
            TestResult: Result of the hypothesis test.

        Raises:
            ValueError: If the two sequences do not have the same length.
        """
        if not (len(sequence_0) == len(sequence_1)):
            raise (ValueError("The two input sequences must have the same size."))

        result = None
        for idx in range(len(sequence_0)):
            result = self.step(sequence_0[idx], sequence_1[idx], *args, **kwargs)
            if not result.decision == Decision.FailToDecide:
                break
        return result

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
            args: Additional positional arguments.
            **kwargs: Additional optional or keyward arguments.

        Returns:
            TestResult: Result of the hypothesis test.
        """
        pass


SequentialTestBase = SequentialTwoSampleTestBase
