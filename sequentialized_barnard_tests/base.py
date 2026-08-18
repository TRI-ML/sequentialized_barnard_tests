"""Base class definitions.

This module re-exports shared test interfaces from statistical_comparison_core.
The canonical definitions now live in statistical_comparison_core.base.
"""

from statistical_comparison_core.base import (
    Decision,
    Hypothesis,
    MirroredTestMixin,
    SequentialTestBase,
    SequentialTwoSampleTestBase,
    TestBase,
    TestResult,
    TwoSampleBinomialHypothesis,
    TwoSampleMeanHypothesis,
    TwoSampleTestBase,
)

__all__ = [
    "Decision",
    "Hypothesis",
    "MirroredTestMixin",
    "SequentialTestBase",
    "SequentialTwoSampleTestBase",
    "TestBase",
    "TestResult",
    "TwoSampleBinomialHypothesis",
    "TwoSampleMeanHypothesis",
    "TwoSampleTestBase",
]
