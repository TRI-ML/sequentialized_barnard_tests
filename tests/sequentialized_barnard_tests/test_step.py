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
            "../eval_data/",
        )
    ).resolve()
)
eval_clean_up_spill = np.load(
    f"{paper_data_path}/TRI_CLEAN_SPILL_v4.npy"
)  # Must be flipped for standard form
eval_fold_red_towel = np.load(
    f"{paper_data_path}/TRI_FOLD_RED_TOWEL.npy"
)  # ALREADY in standard form
eval_sim_spoon_on_towel = np.load(
    f"{paper_data_path}/TRI_SIM_SPOON_ON_TOWEL.npy"
)  # Must be flipped for standard form
eval_sim_eggplant_in_basket = np.load(
    f"{paper_data_path}/TRI_SIM_EGGPLANT_IN_BASKET.npy"
)  # Must be flipped for standard form
eval_sim_stack_cube = np.load(
    f"{paper_data_path}/TRI_SIM_STACK_CUBE.npy"
)  # Must be flipped for standard form


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
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], Decision.AcceptAlternative),
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
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 9),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 9),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 21),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 50),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 50),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 21),
        # fmt: on
    ],
    indirect=["step"],
)
def test_step_time(step, sequence_0, sequence_1, expected):
    result = step.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(float(result.info["Time"]) - expected) <= 0.6


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
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 13),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 13),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 23),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 50),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 50),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 23),
        # fmt: on
    ],
    indirect=["step500"],
)
def test_step500_time(step500, sequence_0, sequence_1, expected):
    result = step500.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


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
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 9),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 9),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 9),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 9),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 21),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 21),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 21),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 21),
        # fmt: on
    ],
    indirect=["mirrored_step"],
)
def test_mirrored_step_time(mirrored_step, sequence_0, sequence_1, expected):
    result = mirrored_step.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


@pytest.fixture(scope="module")
def mirrored_step50(request):
    test = MirroredStepTest(
        alternative=request.param,
        n_max=50,
        alpha=0.05,
    )

    return test


@pytest.mark.parametrize(
    ("mirrored_step50", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 8),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 8),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 8),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 8),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 19),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 19),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 19),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 19),
        # fmt: on
    ],
    indirect=["mirrored_step50"],
)
def test_mirrored_step50_time(mirrored_step50, sequence_0, sequence_1, expected):
    result = mirrored_step50.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


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
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 13),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 13),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 13),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 13),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 23),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 23),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 23),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 23),
        # fmt: on
    ],
    indirect=["mirrored_step500"],
)
def test_mirrored_step500_time(mirrored_step500, sequence_0, sequence_1, expected):
    result = mirrored_step500.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


@pytest.fixture(scope="module")
def mirrored_step500_simulator(request):
    test = MirroredStepTest(
        alternative=request.param,
        n_max=500,
        alpha=0.01,
    )

    return test


@pytest.mark.parametrize(
    ("mirrored_step500_simulator", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_sim_spoon_on_towel[:, 1], eval_sim_spoon_on_towel[:, 0], 36),
        (Hypothesis.P0MoreThanP1, eval_sim_spoon_on_towel[:, 1], eval_sim_spoon_on_towel[:, 0], 36),
        (Hypothesis.P0LessThanP1, eval_sim_spoon_on_towel[:, 0], eval_sim_spoon_on_towel[:, 1], 36),
        (Hypothesis.P0MoreThanP1, eval_sim_spoon_on_towel[:, 0], eval_sim_spoon_on_towel[:, 1], 36),
        (Hypothesis.P0LessThanP1, eval_sim_eggplant_in_basket[:, 1], eval_sim_eggplant_in_basket[:, 0], 131),
        (Hypothesis.P0MoreThanP1, eval_sim_eggplant_in_basket[:, 1], eval_sim_eggplant_in_basket[:, 0], 131),
        (Hypothesis.P0LessThanP1, eval_sim_eggplant_in_basket[:, 0], eval_sim_eggplant_in_basket[:, 1], 131),
        (Hypothesis.P0MoreThanP1, eval_sim_eggplant_in_basket[:, 0], eval_sim_eggplant_in_basket[:, 1], 131),
        (Hypothesis.P0LessThanP1, eval_sim_stack_cube[:, 1], eval_sim_stack_cube[:, 0], 225),
        (Hypothesis.P0MoreThanP1, eval_sim_stack_cube[:, 1], eval_sim_stack_cube[:, 0], 225),
        (Hypothesis.P0LessThanP1, eval_sim_stack_cube[:, 0], eval_sim_stack_cube[:, 1], 225),
        (Hypothesis.P0MoreThanP1, eval_sim_stack_cube[:, 0], eval_sim_stack_cube[:, 1], 225),
        # fmt: on
    ],
    indirect=["mirrored_step500_simulator"],
)
def test_mirrored_step500_time(
    mirrored_step500_simulator, sequence_0, sequence_1, expected
):
    result = mirrored_step500_simulator.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6
