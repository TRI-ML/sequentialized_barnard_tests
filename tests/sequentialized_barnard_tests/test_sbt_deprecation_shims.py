"""Tests for deprecation shims in sequentialized_barnard_tests.tools.plotting."""

import warnings

import numpy as np

import statistical_comparison_helpers as sch


class TestCLDShim:
    def test_emits_deprecation_warning(self):
        from sequentialized_barnard_tests.tools.plotting import compact_letter_display

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compact_letter_display(
                [("A", "B")], ["A", "B", "C"]
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_output_matches_statistical_comparison_helpers(self):
        from sequentialized_barnard_tests.tools.plotting import compact_letter_display

        pairs = [("A", "B"), ("B", "C")]
        models = ["A", "B", "C", "D"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim_result = compact_letter_display(pairs, models)

        expected = sch.compact_letter_display(pairs, models)
        assert shim_result == expected

    def test_returns_list_of_str(self):
        from sequentialized_barnard_tests.tools.plotting import compact_letter_display

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compact_letter_display([], ["A", "B"])

        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)


class TestBetaPosteriorShim:
    def test_emits_deprecation_warning(self):
        from sequentialized_barnard_tests.tools.plotting import draw_samples_from_beta_posterior

        rng = np.random.default_rng(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            draw_samples_from_beta_posterior(np.array([1, 0, 1]), rng, num_samples=100)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_output_shape(self):
        from sequentialized_barnard_tests.tools.plotting import draw_samples_from_beta_posterior

        rng = np.random.default_rng(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = draw_samples_from_beta_posterior(np.array([1, 0, 1, 1]), rng, num_samples=500)
        assert result.shape == (500,)


class TestPlotShim:
    def test_emits_deprecation_warning(self):
        import matplotlib
        matplotlib.use("Agg")

        from sequentialized_barnard_tests.tools.plotting import plot_model_comparison

        rng = np.random.default_rng(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            warnings.simplefilter("always", DeprecationWarning)
            fig = plot_model_comparison(
                ["A", "B"],
                [np.array([1, 0, 1]), np.array([0, 1, 0])],
                ["a", "b"],
                rng,
            )
            assert any(
                issubclass(x.category, DeprecationWarning)
                and "sequentialized_barnard_tests.tools.plotting.plot_model_comparison" in str(x.message)
                for x in w
            )

        import matplotlib.pyplot as plt
        plt.close(fig)
