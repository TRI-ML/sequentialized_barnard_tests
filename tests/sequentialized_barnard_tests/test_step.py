"""Unit tests for the STEP procedure"""

import os
from pathlib import Path

import numpy as np
import pytest

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.step import MirroredStepTest, StepTest

##### STEP Test #####
paper_data_path = str(
    Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../sequentialized_barnard_tests/eval_data/",
        )
    ).resolve()
)
eval_trajectories = np.load(f"{paper_data_path}/TRI_CLEAN_SPILL_v2.npy")


@pytest.fixture(scope="module")
def step(request):
    test = StepTest(
        alternative=request.param,
        n_max=200,
        alpha=0.05,
    )

    return test


@pytest.mark.parametrize(
    ("step"),
    [(Hypothesis.P0LessThanP1), (Hypothesis.P0MoreThanP1)],
    indirect=["step"],
)
def test_step_input_value_error(step):
    # Should raise a ValueError if non-binary data is given.
    with pytest.raises(ValueError):
        step.step(1.2, 1)
    with pytest.raises(ValueError):
        step.step(1, 1.2)

    # Should raise a ValueError if input sequences do not have the same length
    with pytest.raises(ValueError):
        step.step([0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        step.step([1.0, 1.0, 1.0], [0.0, 0.0])


@pytest.mark.parametrize(
    ("step", "sequence_0", "sequence_1", "expected"),
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
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["step"],
)
def test_step(step, sequence_0, sequence_1, expected):
    result = step.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


@pytest.mark.parametrize(
    ("step", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 22.5),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 22.5),
        # fmt: on
    ],
    indirect=["step"],
)
def test_step_time(step, sequence_0, sequence_1, expected):
    result = step.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(float(result.info["Time"]) - expected) <= 3.0


@pytest.fixture(scope="module")
def step500(request):
    test = StepTest(
        alternative=request.param,
        n_max=500,
        alpha=0.05,
    )

    return test


@pytest.mark.parametrize(
    ("step500", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 33),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 33),
        # fmt: on
    ],
    indirect=["step500"],
)
def test_step500_time(step500, sequence_0, sequence_1, expected):
    result = step500.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 1.5


##### Mirrored STEP Test #####


@pytest.fixture(scope="module")
def mirrored_step(request):
    test = MirroredStepTest(
        alternative=request.param,
        n_max=200,
        alpha=0.05,
    )

    return test


@pytest.mark.parametrize(
    ("mirrored_step", "sequence_0", "sequence_1", "expected"),
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
    indirect=["mirrored_step"],
)
def test_mirrored_step(mirrored_step, sequence_0, sequence_1, expected):
    result = mirrored_step.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


@pytest.mark.parametrize(
    ("mirrored_step", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 25),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 25),
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 25),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 25),
        # fmt: on
    ],
    indirect=["mirrored_step"],
)
def test_mirrored_step_time(mirrored_step, sequence_0, sequence_1, expected):
    result = mirrored_step.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 1.5


@pytest.fixture(scope="module")
def mirrored_step500(request):
    test = MirroredStepTest(
        alternative=request.param,
        n_max=500,
        alpha=0.05,
    )

    return test


@pytest.mark.parametrize(
    ("mirrored_step500", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 33),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 1], eval_trajectories[:, 0], 33),
        (Hypothesis.P0LessThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 33),
        (Hypothesis.P0MoreThanP1, eval_trajectories[:, 0], eval_trajectories[:, 1], 33),
        # fmt: on
    ],
    indirect=["mirrored_step500"],
)
def test_mirrored_step500_time(mirrored_step500, sequence_0, sequence_1, expected):
    result = mirrored_step500.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 1.5
