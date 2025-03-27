"""Unit tests for the Lai procedure
"""

import numpy as np
import pytest

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.lai import LaiTest, MirroredLaiTest

##### Lai Test #####


@pytest.fixture(scope="module")
def lai(request):
    test = LaiTest(
        alternative=request.param,
        n_max=500,
        alpha=0.05,
    )
    test.set_c(4.3320915613895993e-05)
    return test


@pytest.mark.parametrize(
    ("lai"),
    [(Hypothesis.P0LessThanP1), (Hypothesis.P0MoreThanP1)],
    indirect=["lai"],
)
def test_lai_input_value_error(lai):
    # Should raise a ValueError if non-binary data is given.
    with pytest.raises(ValueError):
        lai.step(1.2, 1)
    with pytest.raises(ValueError):
        lai.step(1, 1.2)

    # Should raise a ValueError if input sequences do not have the same length
    with pytest.raises(ValueError):
        lai.step([0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        lai.step([1.0, 1.0, 1.0], [0.0, 0.0])


@pytest.mark.parametrize(
    ("lai", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.zeros(15), np.ones(15), Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, np.zeros(15), np.ones(15), Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.ones(15), np.zeros(15), Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, np.ones(15), np.zeros(15), Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["lai"],
)
def test_lai(lai, sequence_0, sequence_1, expected):
    result = lai.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


##### Mirrored Lai Test #####


@pytest.fixture(scope="module")
def mirrored_lai(request):
    test = MirroredLaiTest(
        alternative=request.param,
        n_max=500,
        alpha=0.05,
    )
    test.set_c(5.3077895340120925e-05)
    return test


@pytest.mark.parametrize(
    ("mirrored_lai", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.zeros(15), np.ones(15), Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, np.zeros(15), np.ones(15), Decision.AcceptNull),
        (Hypothesis.P0LessThanP1, np.ones(15), np.zeros(15), Decision.AcceptNull),
        (Hypothesis.P0MoreThanP1, np.ones(15), np.zeros(15), Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["mirrored_lai"],
)
def test_mirrored_lai(mirrored_lai, sequence_0, sequence_1, expected):
    result = mirrored_lai.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected
