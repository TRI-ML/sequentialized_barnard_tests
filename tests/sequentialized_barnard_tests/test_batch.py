"""Unit tests for the batch methods.
"""

import pytest

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.batch import (
    BarnardExactTest,
    MirroredBarnardExactTest,
)


@pytest.fixture(scope="module")
def barnard(request):
    test = BarnardExactTest(alpha=0.05, alternative=request.param)
    return test


@pytest.mark.parametrize(
    ("barnard"),
    [(Hypothesis.P0LessThanP1), (Hypothesis.P0MoreThanP1)],
    indirect=["barnard"],
)
def test_barnard_value_error(barnard):
    # Should raise a ValueError if a non-binary sequence is given.
    with pytest.raises(ValueError):
        barnard.run_on_sequence([0, 0], [1.5, -1])
    with pytest.raises(ValueError):
        barnard.run_on_sequence([-1, 0], [1, 1])
    with pytest.raises(ValueError):
        barnard.run_on_sequence([2, 1], [-2, 4])


@pytest.mark.parametrize(
    ("barnard", "sequence_0", "sequence_1", "expected"),
    [
        (Hypothesis.P0LessThanP1, [0, 0], [1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0], [1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1], [0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1], [0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.AcceptAlternative),
    ],
    indirect=["barnard"],
)
def test_barnard(barnard, sequence_0, sequence_1, expected):
    result = barnard.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


def test_mirrored_barnard_properties():
    test = MirroredBarnardExactTest(alpha=0.05, alternative=Hypothesis.P0MoreThanP1)
    assert test.alpha == 0.05
    assert test._test_for_alternative.alpha == test.alpha
    assert test._test_for_null.alpha == test.alpha
    assert test.alternative == Hypothesis.P0MoreThanP1
    assert test._test_for_alternative.alternative == test.alternative
    assert test._test_for_null.alternative == Hypothesis.P0LessThanP1

    new_alpha = 0.2
    test.alpha = new_alpha
    assert test._test_for_alternative.alpha == new_alpha
    assert test._test_for_null.alpha == new_alpha

    new_alternative = Hypothesis.P0LessThanP1
    test.alternative = new_alternative
    assert test._test_for_alternative.alternative == test.alternative
    assert test._test_for_null.alternative == Hypothesis.P0MoreThanP1


@pytest.fixture(scope="module")
def mirrored_barnard(request):
    test = MirroredBarnardExactTest(alpha=0.05, alternative=request.param)
    return test


@pytest.mark.parametrize(
    ("mirrored_barnard", "sequence_0", "sequence_1", "expected"),
    [
        (Hypothesis.P0LessThanP1, [0, 0], [1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0], [1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1], [0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1], [0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.AcceptNull),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.AcceptNull),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.AcceptAlternative),
    ],
    indirect=["mirrored_barnard"],
)
def test_mirrored_barnard(mirrored_barnard, sequence_0, sequence_1, expected):
    result = mirrored_barnard.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected
