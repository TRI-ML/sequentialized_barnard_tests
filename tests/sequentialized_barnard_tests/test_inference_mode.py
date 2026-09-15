"""Tests for the ``inference_mode`` keyword on mirrored SBT tests.

STEP has no p-value, so its mirrored inference mode only controls policy
selection and synthesis alpha, not decision logic or policy format. Lai uses
the same mode vocabulary to choose the effective one-sided alpha used for
regularizer calibration.
"""

import os

import pytest

from sequentialized_barnard_tests import Hypothesis
from sequentialized_barnard_tests.auto import get_mirrored_test, get_test
from sequentialized_barnard_tests.lai import LaiTest, MirroredLaiTest
from sequentialized_barnard_tests.step import (
    _ALLOWED_INFERENCE_MODES,
    MirroredStepTest,
    StepTest,
)


@pytest.fixture
def prevent_synthesis(monkeypatch):
    """Replace ``StepTest.load_existing_policy`` with a no-op that never
    reads or writes disk and never spawns the synthesis subprocess."""

    def fake_load_existing_policy(self, verbose=False):
        # Mirror the real code's policy_path shape so tests can assert on it.
        self.policy_path = os.path.join(
            os.path.dirname(__file__),
            f"policies/n_max_{self.n_max}_alpha_{self._policy_alpha}_shape_parameter_{self.shape_parameter}_pnorm_{self.use_p_norm}/",
            "policy_compressed.pkl",
        )
        self.policy = "fake_policy"
        self.need_new_policy = False
        # Match the real method's reset call at the end.
        self.reset(verbose)

    monkeypatch.setattr(
        StepTest, "load_existing_policy", fake_load_existing_policy
    )


# --- default / comparison ---

def test_default_mode_sets_policy_alpha_equal_to_declared_alpha(prevent_synthesis):
    m = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1, n_max=100, alpha=0.05
    )
    assert m.alpha == pytest.approx(0.05)
    assert m._policy_alpha == pytest.approx(0.05)
    assert m.inference_mode == "comparison"


def test_comparison_mode_matches_default(prevent_synthesis):
    m_default = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1, n_max=100, alpha=0.05
    )
    m_explicit = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=100,
        alpha=0.05,
        inference_mode="comparison",
    )
    assert m_default.alpha == m_explicit.alpha
    assert m_default._policy_alpha == m_explicit._policy_alpha
    assert m_default.inference_mode == m_explicit.inference_mode


# --- ranking halves the policy alpha but keeps the declared alpha ---

def test_ranking_halves_policy_alpha_and_preserves_declared_alpha(
    prevent_synthesis,
):
    m = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=100,
        alpha=0.05,
        inference_mode="ranking",
    )
    assert m.alpha == pytest.approx(0.05)
    assert m._policy_alpha == pytest.approx(0.025)
    assert m.inference_mode == "ranking"


def test_lai_ranking_halves_effective_alpha_and_preserves_declared_alpha():
    m = MirroredLaiTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=800,
        alpha=0.05,
        calibrate_regularizer=False,
        use_offline_calibration=False,
        inference_mode="ranking",
    )
    assert m.alpha == pytest.approx(0.05)
    assert m._test_alpha == pytest.approx(0.025)
    assert m.calibration_correction == pytest.approx(0.025 / 50.0)
    assert m.inference_mode == "ranking"


def test_ranking_uses_policy_alpha_in_production_load_existing_policy(
    monkeypatch,
):
    """Exercise the real ``StepTest.load_existing_policy`` body far enough to
    verify both ``self._policy_alpha`` call sites: the policy directory path
    and the synthesis subprocess ``--alpha`` argument.

    Real synthesis is prevented by injecting a fake ``open`` into the step
    module's globals that raises ``FileNotFoundError`` and a fake
    ``subprocess.run`` that captures the command and raises a
    ``_SynthesisSentinel``. Nothing touches disk.
    """

    class _SynthesisSentinel(RuntimeError):
        pass

    open_paths: list = []
    subprocess_cmds: list = []

    def fake_open(path, *args, **kwargs):
        open_paths.append(path)
        raise FileNotFoundError(path)

    def fake_subprocess_run(cmd, *args, **kwargs):
        subprocess_cmds.append(cmd)
        raise _SynthesisSentinel("synthesis suppressed in test")

    import sequentialized_barnard_tests.step as step_mod

    monkeypatch.setattr(step_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(step_mod.subprocess, "run", fake_subprocess_run)

    # Construction: StepTest.__init__ wraps load_existing_policy in a bare
    # except, so the sentinel is swallowed here even though the production
    # body ran. The captured lists still record what production tried to do.
    m = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=100,
        alpha=0.1,
        inference_mode="ranking",
    )

    # Re-invoke load_existing_policy directly so the sentinel propagates and
    # is assertable, and re-capture both call sites for inspection.
    open_paths.clear()
    subprocess_cmds.clear()
    with pytest.raises(_SynthesisSentinel):
        m.load_existing_policy()

    # First call site (step.py policy_path construction): must use
    # self._policy_alpha = 0.1 / 2.0 = 0.05, not the declared self.alpha = 0.1.
    assert open_paths, "load_existing_policy did not attempt to open a policy file"
    first_path = open_paths[0]
    expected_alpha_segment = f"alpha_{0.1 / 2.0}"
    assert expected_alpha_segment in first_path, (
        f"Expected policy_path to contain {expected_alpha_segment!r}, got {first_path!r}"
    )
    assert "alpha_0.1_" not in first_path

    # Second call site (step.py subprocess --alpha argument): must be
    # str(self._policy_alpha), not str(self.alpha).
    assert subprocess_cmds, "load_existing_policy did not attempt synthesis"
    cmd = subprocess_cmds[0]
    assert "--alpha" in cmd
    alpha_idx = cmd.index("--alpha")
    assert cmd[alpha_idx + 1] == str(0.1 / 2.0)
    assert cmd[alpha_idx + 1] != "0.1"


# --- ranking_no_ties uses full alpha and matches comparison ---

def test_ranking_no_ties_uses_full_alpha(prevent_synthesis):
    m = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=100,
        alpha=0.05,
        inference_mode="ranking_no_ties",
    )
    assert m.alpha == pytest.approx(0.05)
    assert m._policy_alpha == pytest.approx(0.05)
    assert m.inference_mode == "ranking_no_ties"


def test_lai_ranking_no_ties_uses_full_effective_alpha():
    m = MirroredLaiTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=800,
        alpha=0.05,
        calibrate_regularizer=False,
        use_offline_calibration=False,
        inference_mode="ranking_no_ties",
    )
    assert m.alpha == pytest.approx(0.05)
    assert m._test_alpha == pytest.approx(0.05)
    assert m.inference_mode == "ranking_no_ties"


def test_ranking_no_ties_matches_comparison_policy_alpha(prevent_synthesis):
    m_cmp = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=100,
        alpha=0.05,
        inference_mode="comparison",
    )
    m_rnt = MirroredStepTest(
        alternative=Hypothesis.P0MoreThanP1,
        n_max=100,
        alpha=0.05,
        inference_mode="ranking_no_ties",
    )
    assert m_cmp._policy_alpha == m_rnt._policy_alpha
    assert m_cmp.alpha == m_rnt.alpha


# --- invalid inference_mode ---

def test_invalid_inference_mode_raises_and_lists_valid_values():
    with pytest.raises(ValueError) as excinfo:
        MirroredStepTest(
            alternative=Hypothesis.P0MoreThanP1,
            n_max=100,
            alpha=0.05,
            inference_mode="bogus",
        )
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("comparison", "ranking", "ranking_no_ties"):
        assert mode in message


def test_invalid_lai_inference_mode_raises_and_lists_valid_values():
    with pytest.raises(ValueError) as excinfo:
        MirroredLaiTest(
            alternative=Hypothesis.P0MoreThanP1,
            n_max=800,
            alpha=0.05,
            calibrate_regularizer=False,
            use_offline_calibration=False,
            inference_mode="bogus",
        )
    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("comparison", "ranking", "ranking_no_ties"):
        assert mode in message


def test_auto_forwards_inference_mode_to_lai_branch():
    m = get_mirrored_test(
        n_max=800,
        alternative=Hypothesis.P0MoreThanP1,
        alpha=0.05,
        calibrate_regularizer=False,
        use_offline_calibration=False,
        inference_mode="ranking",
    )
    assert isinstance(m, MirroredLaiTest)
    assert m.alpha == pytest.approx(0.05)
    assert m._test_alpha == pytest.approx(0.025)
    assert m.inference_mode == "ranking"


def test_auto_default_critical_n_max_selects_step_at_boundary(
    prevent_synthesis,
):
    m = get_mirrored_test(
        n_max=500,
        alternative=Hypothesis.P0MoreThanP1,
        alpha=0.05,
    )
    assert isinstance(m, MirroredStepTest)


def test_auto_default_critical_n_max_selects_lai_above_boundary():
    m = get_mirrored_test(
        n_max=501,
        alternative=Hypothesis.P0MoreThanP1,
        alpha=0.05,
        calibrate_regularizer=False,
        use_offline_calibration=False,
    )
    assert isinstance(m, MirroredLaiTest)


def test_auto_custom_critical_n_max_selects_regular_lai_branch():
    m = get_test(
        n_max=200,
        alternative=Hypothesis.P0MoreThanP1,
        alpha=0.05,
        critical_n_max=100,
        calibrate_regularizer=False,
        use_offline_calibration=False,
    )
    assert isinstance(m, LaiTest)


def test_auto_custom_critical_n_max_selects_regular_step_branch(
    prevent_synthesis,
):
    m = get_test(
        n_max=200,
        alternative=Hypothesis.P0MoreThanP1,
        alpha=0.05,
        critical_n_max=250,
    )
    assert isinstance(m, StepTest)


def test_auto_custom_critical_n_max_forwards_kwargs_to_mirrored_lai():
    m = get_mirrored_test(
        n_max=200,
        alternative=Hypothesis.P0MoreThanP1,
        alpha=0.05,
        critical_n_max=100,
        calibrate_regularizer=False,
        use_offline_calibration=False,
        inference_mode="ranking",
    )
    assert isinstance(m, MirroredLaiTest)
    assert m._test_alpha == pytest.approx(0.025)


def test_auto_negative_critical_n_max_raises():
    with pytest.raises(ValueError, match="critical_n_max"):
        get_mirrored_test(
            n_max=200,
            alternative=Hypothesis.P0MoreThanP1,
            alpha=0.05,
            critical_n_max=-1,
        )


# --- StepTest is intentionally not extended with inference_mode ---

def test_step_test_does_not_accept_inference_mode_kwarg():
    with pytest.raises(TypeError):
        StepTest(
            alternative=Hypothesis.P0MoreThanP1,
            n_max=200,
            alpha=0.05,
            inference_mode="ranking",
        )


# --- vocabulary drift guard against statistical_comparison_core ---

def test_inference_mode_vocabulary_matches_statistical_comparison_core():
    # SCC's ``MirroredTestMixin._ALLOWED_INFERENCE_MODES`` is private, so we
    # import it only here in the drift-guard test; production code duplicates
    # the strings so SBT does not depend on an SCC private attribute.
    from statistical_comparison_core.base import MirroredTestMixin

    assert (
        _ALLOWED_INFERENCE_MODES
        == MirroredTestMixin._ALLOWED_INFERENCE_MODES
    )
