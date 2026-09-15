"""Unit tests for STEP policy synthesis helpers."""

from types import SimpleNamespace

import numpy as np

from sequentialized_barnard_tests.scripts import synthesize_general_step_policy


def test_policy_lp_solver_returns_default_highs_solution(monkeypatch):
    calls = []
    expected_x = np.array([0.25, 0.75])

    def fake_linprog(*args, **kwargs):
        calls.append(kwargs.get("method", "highs"))
        return SimpleNamespace(
            success=True,
            x=expected_x,
            status=0,
            message="ok",
        )

    monkeypatch.setattr(synthesize_general_step_policy, "linprog", fake_linprog)

    result = synthesize_general_step_policy._solve_policy_linear_program(
        np.array([-1.0, -1.0]),
        np.eye(2),
        np.ones(2),
        np.zeros((0, 2)),
        np.zeros(0),
        (0.0, 1.0),
        {"disp": False},
    )

    assert calls == ["highs"]
    assert np.array_equal(result.x, expected_x)


def test_policy_lp_solver_falls_back_to_highs_ipm(monkeypatch):
    calls = []
    expected_x = np.array([0.25, 0.75])

    def fake_linprog(*args, **kwargs):
        calls.append(kwargs.get("method", "highs"))
        if len(calls) == 1:
            return SimpleNamespace(
                success=False,
                x=None,
                status=4,
                message="numerical failure",
            )
        return SimpleNamespace(
            success=True,
            x=expected_x,
            status=0,
            message="ok",
        )

    monkeypatch.setattr(synthesize_general_step_policy, "linprog", fake_linprog)

    result = synthesize_general_step_policy._solve_policy_linear_program(
        np.array([-1.0, -1.0]),
        np.eye(2),
        np.ones(2),
        np.zeros((0, 2)),
        np.zeros(0),
        (0.0, 1.0),
        {"disp": False},
    )

    assert calls == ["highs", "highs-ipm"]
    assert np.array_equal(result.x, expected_x)
