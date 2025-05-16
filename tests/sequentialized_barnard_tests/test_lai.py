"""Unit tests for the Lai procedure"""

import os
from pathlib import Path

import numpy as np
import pytest

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.lai import LaiTest, MirroredLaiTest

##### Lai Test #####
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
def lai(request):
    test = LaiTest(
        alternative=request.param,
        n_max=500,
        alpha=0.05,
        calibrate_regularizer=False,
        use_offline_calibration=False,
    )
    test.set_c(5.3077895340120925e-05)
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


@pytest.fixture(scope="module")
def lai200(request):
    test = LaiTest(
        alternative=request.param,
        n_max=200,
        alpha=0.05,
    )
    test.set_c(0.00014121395942619315)
    return test


@pytest.mark.parametrize(
    ("lai200", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 13),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 13),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 21),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 50),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 50),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 21),
        # fmt: on
    ],
    indirect=["lai200"],
)
def test_lai200_time(lai200, sequence_0, sequence_1, expected):
    result = lai200.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


@pytest.fixture(scope="module")
def lai50(request):
    test = LaiTest(
        alternative=request.param,
        n_max=50,
        alpha=0.05,
    )
    test.set_c(0.000561395711114114)
    return test


@pytest.mark.parametrize(
    ("lai50", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 8),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 8),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 17),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 50),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 50),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 17),
        # fmt: on
    ],
    indirect=["lai50"],
)
def test_lai50_time(lai50, sequence_0, sequence_1, expected):
    result = lai50.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


##### Mirrored Lai Test #####


@pytest.fixture(scope="module")
def mirrored_lai(request):
    test = MirroredLaiTest(
        alternative=request.param,
        n_max=500,
        alpha=0.05,
        calibrate_regularizer=False,
        use_offline_calibration=False,
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


@pytest.fixture(scope="module")
def mirrored_lai200(request):
    test = MirroredLaiTest(
        alternative=request.param,
        n_max=200,
        alpha=0.05,
    )
    test.set_c(0.00014121395942619315)
    return test


@pytest.mark.parametrize(
    ("mirrored_lai200", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 13),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 13),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 13),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 13),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 21),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 21),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 21),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 21),
        # fmt: on
    ],
    indirect=["mirrored_lai200"],
)
def test_mirrored_lai200_time(mirrored_lai200, sequence_0, sequence_1, expected):
    result = mirrored_lai200.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


@pytest.fixture(scope="module")
def mirrored_lai50(request):
    test = MirroredLaiTest(
        alternative=request.param,
        n_max=50,
        alpha=0.05,
    )
    test.set_c(0.000561395711114114)
    return test


@pytest.mark.parametrize(
    ("mirrored_lai50", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 8),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 8),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 8),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 8),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 17),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 17),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 17),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 17),
        # fmt: on
    ],
    indirect=["mirrored_lai50"],
)
def test_mirrored_lai50_time(mirrored_lai50, sequence_0, sequence_1, expected):
    result = mirrored_lai50.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


@pytest.fixture(scope="module")
def mirrored_lai500(request):
    test = MirroredLaiTest(
        alternative=request.param,
        n_max=500,
        alpha=0.01,
        calibrate_regularizer=False,
        use_offline_calibration=False,
    )
    test.set_c(1.013009359863071e-05)
    return test


@pytest.mark.parametrize(
    ("mirrored_lai500", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_sim_spoon_on_towel[:, 1], eval_sim_spoon_on_towel[:, 0], 36),
        (Hypothesis.P0MoreThanP1, eval_sim_spoon_on_towel[:, 1], eval_sim_spoon_on_towel[:, 0], 36),
        (Hypothesis.P0LessThanP1, eval_sim_spoon_on_towel[:, 0], eval_sim_spoon_on_towel[:, 1], 36),
        (Hypothesis.P0MoreThanP1, eval_sim_spoon_on_towel[:, 0], eval_sim_spoon_on_towel[:, 1], 36),
        (Hypothesis.P0LessThanP1, eval_sim_eggplant_in_basket[:, 1], eval_sim_eggplant_in_basket[:, 0], 125),
        (Hypothesis.P0MoreThanP1, eval_sim_eggplant_in_basket[:, 1], eval_sim_eggplant_in_basket[:, 0], 125),
        (Hypothesis.P0LessThanP1, eval_sim_eggplant_in_basket[:, 0], eval_sim_eggplant_in_basket[:, 1], 125),
        (Hypothesis.P0MoreThanP1, eval_sim_eggplant_in_basket[:, 0], eval_sim_eggplant_in_basket[:, 1], 125),
        (Hypothesis.P0LessThanP1, eval_sim_stack_cube[:, 1], eval_sim_stack_cube[:, 0], 417),
        (Hypothesis.P0MoreThanP1, eval_sim_stack_cube[:, 1], eval_sim_stack_cube[:, 0], 417),
        (Hypothesis.P0LessThanP1, eval_sim_stack_cube[:, 0], eval_sim_stack_cube[:, 1], 417),
        (Hypothesis.P0MoreThanP1, eval_sim_stack_cube[:, 0], eval_sim_stack_cube[:, 1], 417),
        # fmt: on
    ],
    indirect=["mirrored_lai500"],
)
def test_mirrored_lai500_time(mirrored_lai500, sequence_0, sequence_1, expected):
    result = mirrored_lai500.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


##### Offline Calibration Test #####
@pytest.mark.parametrize(
    ("alpha", "n_max"),
    [
        # fmt: off
        (0.004, 250),
        (0.004, 1500),
        (0.02, 250),
        (0.02, 1500),
        (0.25, 250),
        (0.25, 1500),
        (0.62, 250),
        (0.62, 1500),
        # fmt: on
    ],
)
def test_offline_calibration(alpha, n_max):
    test_offline = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=n_max,
        alpha=alpha,
        calibrate_regularizer=True,
        use_offline_calibration=True,
    )
    test_online = MirroredLaiTest(
        alternative=Hypothesis.P0LessThanP1,
        n_max=n_max,
        alpha=alpha,
        calibrate_regularizer=True,
        n_calibration_sequences=1000,
        use_offline_calibration=False,
    )
    assert np.abs(test_offline.c - test_online.c) < 1e-3
    # TODO (Haruki): Absolute error is not a good measure of accuracy.
    # Perhaps we should run MC simulation using the estimated `c` and compare the
    # terminal FPR to the requested value of alpha.
