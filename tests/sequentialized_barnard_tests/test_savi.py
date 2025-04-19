"""Unit tests for the SAVI method.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.savi import (
    MirroredOracleSaviTest,
    MirroredSaviTest,
    OracleSaviTest,
    SaviTest,
)

##### SAVI Test #####
paper_data_path = str(
    Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../eval_data/",
        )
    ).resolve()
)
eval_clean_up_spill = np.load(
    f"{paper_data_path}/TRI_CLEAN_SPILL_v2.npy"
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
def savi(request):
    test = SaviTest(alternative=request.param, alpha=0.05)
    return test


@pytest.mark.parametrize(
    ("savi"),
    [(Hypothesis.P0LessThanP1), (Hypothesis.P0MoreThanP1)],
    indirect=["savi"],
)
def test_savi_value_error(savi):
    # Should raise a ValueError if non-binary data is given.
    with pytest.raises(ValueError):
        savi.step(1.2, 1)
    with pytest.raises(ValueError):
        savi.step(1, 1.2)

    # Should raise a ValueError if two sequences do not have the same length.
    with pytest.raises(ValueError):
        savi.run_on_sequence([0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        savi.run_on_sequence([1, 1, 1], [0, 0])


@pytest.mark.parametrize(
    ("savi", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [0, 0, 0, 0], [1, 1, 1, 1], Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, [0, 0, 0, 0], [1, 1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1, 1], [0, 0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1, 1], [0, 0, 0, 0], Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["savi"],
)
def test_savi(savi, sequence_0, sequence_1, expected):
    result = savi.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


@pytest.fixture(scope="module")
def savi_paper_hardware(request):
    test = SaviTest(alternative=request.param, alpha=0.05)
    return test


@pytest.mark.parametrize(
    ("savi_paper_hardware", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 21.5),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 1], eval_clean_up_spill[:, 0], 50),
        (Hypothesis.P0LessThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 50),
        (Hypothesis.P0MoreThanP1, eval_clean_up_spill[:, 0], eval_clean_up_spill[:, 1], 21.5),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 19.5),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 0], eval_fold_red_towel[:, 1], 50),
        (Hypothesis.P0LessThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 50),
        (Hypothesis.P0MoreThanP1, eval_fold_red_towel[:, 1], eval_fold_red_towel[:, 0], 19.5),
        # fmt: on
    ],
    indirect=["savi_paper_hardware"],
)
def test_savi_paper_hardware(savi_paper_hardware, sequence_0, sequence_1, expected):
    result = savi_paper_hardware.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6


##### Oracle SAVI Test #####


@pytest.fixture(scope="module")
def oracle_savi(request):
    print(request)
    test = OracleSaviTest(
        alternative=request.param["alt"],
        alpha=0.05,
        true_parameters=request.param["param"],
    )
    return test


@pytest.mark.parametrize(
    ("oracle_savi"),
    [
        {"alt": Hypothesis.P0LessThanP1, "param": (0.0, 1.0)},
        {"alt": Hypothesis.P0MoreThanP1, "param": (0.0, 1.0)},
    ],
    indirect=["oracle_savi"],
)
def test_oracle_savi_value_error(oracle_savi):
    # Should raise a ValueError if non-binary data is given.
    with pytest.raises(ValueError):
        oracle_savi.step(1.2, 1)
    with pytest.raises(ValueError):
        oracle_savi.step(1, 1.2)

    # Should raise a ValueError if two sequences do not have the same length.
    with pytest.raises(ValueError):
        oracle_savi.run_on_sequence([0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        oracle_savi.run_on_sequence([1, 1, 1], [0, 0])


@pytest.mark.parametrize(
    ("oracle_savi", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        ({"alt": Hypothesis.P0LessThanP1, "param": (0.0, 1.0)}, [0, 0], [1, 1], Decision.FailToDecide),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (0.0, 1.0)}, [0, 0], [1, 1], Decision.FailToDecide),
        ({"alt": Hypothesis.P0LessThanP1, "param": (1.0, 0.0)}, [1, 1], [0, 0], Decision.FailToDecide),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (1.0, 0.0)}, [1, 1], [0, 0], Decision.FailToDecide),
        ({"alt": Hypothesis.P0LessThanP1, "param": (0.0, 1.0)}, [0, 0, 0], [1, 1, 1], Decision.AcceptAlternative),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (0.0, 1.0)}, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        ({"alt": Hypothesis.P0LessThanP1, "param": (1.0, 0.0)}, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (1.0, 0.0)}, [1, 1, 1], [0, 0, 0], Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["oracle_savi"],
)
def test_oracle_savi(oracle_savi, sequence_0, sequence_1, expected):
    result = oracle_savi.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


##### Mirrored SAVI Test #####


def test_mirrored_savi_attribute_assignment():
    test = MirroredSaviTest(alternative=Hypothesis.P0MoreThanP1, alpha=0.05)
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
def mirrored_savi(request):
    test = MirroredSaviTest(alternative=request.param, alpha=0.05)
    return test


@pytest.mark.parametrize(
    ("mirrored_savi", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [0, 0, 0, 0], [1, 1, 1, 1], Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, [0, 0, 0, 0], [1, 1, 1, 1], Decision.AcceptNull),
        (Hypothesis.P0LessThanP1, [1, 1, 1, 1], [0, 0, 0, 0], Decision.AcceptNull),
        (Hypothesis.P0MoreThanP1, [1, 1, 1, 1], [0, 0, 0, 0], Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["mirrored_savi"],
)
def test_mirrored_savi(mirrored_savi, sequence_0, sequence_1, expected):
    result = mirrored_savi.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


@pytest.fixture(scope="module")
def mirrored_savi_paper_sim(request):
    test = MirroredSaviTest(alternative=request.param, alpha=0.01)
    return test


@pytest.mark.parametrize(
    ("mirrored_savi_paper_sim", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_sim_spoon_on_towel[:, 1], eval_sim_spoon_on_towel[:, 0], 32.5),
        (Hypothesis.P0MoreThanP1, eval_sim_spoon_on_towel[:, 1], eval_sim_spoon_on_towel[:, 0], 32.5),
        (Hypothesis.P0LessThanP1, eval_sim_spoon_on_towel[:, 0], eval_sim_spoon_on_towel[:, 1], 32.5),
        (Hypothesis.P0MoreThanP1, eval_sim_spoon_on_towel[:, 0], eval_sim_spoon_on_towel[:, 1], 32.5),
        (Hypothesis.P0LessThanP1, eval_sim_eggplant_in_basket[:, 1], eval_sim_eggplant_in_basket[:, 0], 191.5),
        (Hypothesis.P0MoreThanP1, eval_sim_eggplant_in_basket[:, 1], eval_sim_eggplant_in_basket[:, 0], 191.5),
        (Hypothesis.P0LessThanP1, eval_sim_eggplant_in_basket[:, 0], eval_sim_eggplant_in_basket[:, 1], 191.5),
        (Hypothesis.P0MoreThanP1, eval_sim_eggplant_in_basket[:, 0], eval_sim_eggplant_in_basket[:, 1], 191.5),
        (Hypothesis.P0LessThanP1, eval_sim_stack_cube[:, 1], eval_sim_stack_cube[:, 0], 328.5),
        (Hypothesis.P0MoreThanP1, eval_sim_stack_cube[:, 1], eval_sim_stack_cube[:, 0], 328.5),
        (Hypothesis.P0LessThanP1, eval_sim_stack_cube[:, 0], eval_sim_stack_cube[:, 1], 328.5),
        (Hypothesis.P0MoreThanP1, eval_sim_stack_cube[:, 0], eval_sim_stack_cube[:, 1], 328.5),
        # fmt: on
    ],
    indirect=["mirrored_savi_paper_sim"],
)
def test_mirrored_savi_paper_sim_time(
    mirrored_savi_paper_sim, sequence_0, sequence_1, expected
):
    result = mirrored_savi_paper_sim.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["result_for_alternative"].info["Time"] - expected) <= 0.6


##### Mirrored Oracle SAVI Test #####


def test_mirrored_oracle_savi_attribute_assignment():
    test = MirroredOracleSaviTest(
        alternative=Hypothesis.P0MoreThanP1, alpha=0.05, true_parameters=(0.7, 0.3)
    )
    assert test.alpha == 0.05
    assert test._test_for_alternative.alpha == test.alpha
    assert test._test_for_null.alpha == test.alpha
    assert test.alternative == Hypothesis.P0MoreThanP1
    assert test._test_for_alternative.alternative == test.alternative
    assert test._test_for_null.alternative == Hypothesis.P0LessThanP1
    assert test.true_parameters == (0.7, 0.3)
    assert test._test_for_alternative.true_parameters == test.true_parameters
    assert test._test_for_null.true_parameters == test.true_parameters

    new_alpha = 0.2
    test.alpha = new_alpha
    assert test._test_for_alternative.alpha == new_alpha
    assert test._test_for_null.alpha == new_alpha

    new_alternative = Hypothesis.P0LessThanP1
    test.alternative = new_alternative
    assert test._test_for_alternative.alternative == test.alternative
    assert test._test_for_null.alternative == Hypothesis.P0MoreThanP1

    new_parameters = (0.9, 0.2)
    test.true_parameters = new_parameters
    assert test._test_for_alternative.true_parameters == new_parameters
    assert test._test_for_null.true_parameters == new_parameters


@pytest.fixture(scope="module")
def mirrored_oracle_savi(request):
    print(request)
    test = MirroredOracleSaviTest(
        alternative=request.param["alt"],
        alpha=0.05,
        true_parameters=request.param["param"],
    )
    return test


@pytest.mark.parametrize(
    ("mirrored_oracle_savi", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        ({"alt": Hypothesis.P0LessThanP1, "param": (0.0, 1.0)}, [0, 0], [1, 1], Decision.FailToDecide),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (0.0, 1.0)}, [0, 0], [1, 1], Decision.FailToDecide),
        ({"alt": Hypothesis.P0LessThanP1, "param": (1.0, 0.0)}, [1, 1], [0, 0], Decision.FailToDecide),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (1.0, 0.0)}, [1, 1], [0, 0], Decision.FailToDecide),
        ({"alt": Hypothesis.P0LessThanP1, "param": (0.0, 1.0)}, [0, 0, 0], [1, 1, 1], Decision.AcceptAlternative),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (0.0, 1.0)}, [0, 0, 0], [1, 1, 1], Decision.AcceptNull),
        ({"alt": Hypothesis.P0LessThanP1, "param": (1.0, 0.0)}, [1, 1, 1], [0, 0, 0], Decision.AcceptNull),
        ({"alt": Hypothesis.P0MoreThanP1, "param": (1.0, 0.0)}, [1, 1, 1], [0, 0, 0], Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["mirrored_oracle_savi"],
)
def test_mirrored_oracle_savi(mirrored_oracle_savi, sequence_0, sequence_1, expected):
    result = mirrored_oracle_savi.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected
