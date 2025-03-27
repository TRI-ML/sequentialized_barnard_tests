"""Unit tests for the Barnard sequential (rectified) method.

TODO: Add additional tests to verify unusual specification of the times of evaluation, etc
"""

import numpy as np
import pytest

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.barnard_sequential import (
    BarnardSequentialTest,
    MirroredBarnardSequentialTest,
)

##### Barnard Sequential Test #####


@pytest.fixture(scope="module")
def barnard_sequential(request):
    test = BarnardSequentialTest(
        alternative=request.param, alpha=0.05, n_max=500, times_of_evaluation=10
    )
    return test


@pytest.mark.parametrize(
    ("barnard_sequential"),
    [(Hypothesis.P0LessThanP1), (Hypothesis.P0MoreThanP1)],
    indirect=["barnard_sequential"],
)
def test_barnard_sequential_value_error(barnard_sequential):
    # Should raise a ValueError if non-binary data is given.
    with pytest.raises(ValueError):
        barnard_sequential.step(1.2, 1)
    with pytest.raises(ValueError):
        barnard_sequential.step(1, 1.2)

    # Should raise a ValueError if two sequences do not have the same length.
    with pytest.raises(ValueError):
        barnard_sequential.run_on_sequence([0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        barnard_sequential.run_on_sequence([1, 1, 1], [0, 0])


@pytest.mark.parametrize(
    ("barnard_sequential", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.zeros(31), np.ones(31), Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, np.zeros(31), np.ones(31), Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.ones(31), np.zeros(31), Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, np.ones(31), np.zeros(31), Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["barnard_sequential"],
)
def test_barnard_sequential(barnard_sequential, sequence_0, sequence_1, expected):
    result = barnard_sequential.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


##### Mirrored Barnard Sequential Test #####


@pytest.fixture(scope="module")
def mirrored_barnard_sequential(request):
    test = MirroredBarnardSequentialTest(
        alternative=request.param, alpha=0.05, n_max=500, times_of_evaluation=10
    )
    return test


@pytest.mark.parametrize(
    ("mirrored_barnard_sequential", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.zeros(31), np.ones(31), Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, np.zeros(31), np.ones(31), Decision.AcceptNull),
        (Hypothesis.P0LessThanP1, np.ones(31), np.zeros(31), Decision.AcceptNull),
        (Hypothesis.P0MoreThanP1, np.ones(31), np.zeros(31), Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["mirrored_barnard_sequential"],
)
def test_mirrored_barnard_sequential(
    mirrored_barnard_sequential, sequence_0, sequence_1, expected
):
    result = mirrored_barnard_sequential.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


@pytest.mark.parametrize(
    ("mirrored_barnard_sequential", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 1, 0, 0, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1], Decision.FailToDecide),
        # fmt: on
    ],
    indirect=["mirrored_barnard_sequential"],
)
def test_mirrored_barnard_sequential_special(
    mirrored_barnard_sequential, sequence_0, sequence_1, expected
):
    result = mirrored_barnard_sequential.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected
    assert (
        result.info["result_for_alternative"].info["p_value"]
        < mirrored_barnard_sequential.alpha
    )


@pytest.fixture(scope="module")
def mirrored_barnard_sequential_slow(request):
    test = MirroredBarnardSequentialTest(
        alternative=request.param, alpha=0.05, n_max=500, times_of_evaluation=50
    )
    return test


@pytest.mark.parametrize(
    ("mirrored_barnard_sequential_slow", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.zeros(49), np.ones(49), Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, np.zeros(49), np.ones(49), Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.ones(49), np.zeros(49), Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, np.ones(49), np.zeros(49), Decision.FailToDecide),
        # fmt: on
    ],
    indirect=["mirrored_barnard_sequential_slow"],
)
def test_mirrored_barnard_sequential_slow(
    mirrored_barnard_sequential_slow, sequence_0, sequence_1, expected
):
    result = mirrored_barnard_sequential_slow.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected
