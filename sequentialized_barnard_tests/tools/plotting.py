from typing import Dict, List, Optional, Tuple, Union

import warnings

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.step import MirroredStepTest

import statistical_comparison_helpers as _sch


def compact_letter_display(
    significant_pair_list: List[Tuple[str, str]],
    sorted_model_list: List[str],
) -> List[str]:
    """Generates Compact Letter Display (CLD) given a list of significant
    pairs and a list of models.

    .. deprecated::
        Use ``statistical_comparison_helpers.compact_letter_display`` instead.
    """
    warnings.warn(
        "sequentialized_barnard_tests.tools.plotting.compact_letter_display is "
        "deprecated. Use statistical_comparison_helpers.compact_letter_display instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _sch.compact_letter_display(significant_pair_list, sorted_model_list)


def compare_success_and_get_cld(
    model_name_list: List[str],  # [model_0, ...]
    success_array_list: List[np.ndarray],  # [success_array_for_model_0, ...]
    global_confidence_level: float,
    max_sample_size_per_model: int,
    shuffle: bool,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Compares multiple success arrays and returns their Compact Letter Display (CLD)
    representation based on pairwise tests with STEP.

    Args:
        model_name_list: A list of model names.
        success_array_list: A list of binary arrays indicating success/failure
            for each model.
        global_confidence_level: The desired global confidence level for the
            multiple comparisons.
        max_sample_size_per_model: The maximum sample size to use for comparison
            (per model). You must set this number based on your experimental budget
            before initiating your statistical analysis.
        shuffle: Whether to shuffle the True/False ordering of each success array
            before comparison. Set it to False if each True/False outcome is
            independent within each array. Set to True if, for example, each array is a
            concatenation of results from multiple tasks and you want to measure the
            aggregate multi-task performance.
        rng: Optional random number generator instance for shuffling. Only used if
            shuffle is True.
        verbose: Whether to print detailed output. Defaults to True.
    Returns:
        A dictionary mapping model names to their CLD letters.
    """
    if shuffle and rng is None:
        raise ValueError("rng must be provided when shuffle is True.")
    num_models = len(model_name_list)
    # Set up the sequential statistical test.
    global_alpha = 1 - global_confidence_level
    num_comparisons = num_models * (num_models - 1) // 2
    individual_alpha = global_alpha / num_comparisons
    individual_confidence_level = 1 - individual_alpha
    if verbose:
        print("Statistical Test Specs:")
        print("  Method: STEP")
        print(f"  Global Confidence: {round(global_confidence_level, 5)}")
        print(f"    ({round(individual_confidence_level, 5)} per comparison)")
        print(f"  Maximum Sample Size per Model: {max_sample_size_per_model}\n")
    test = MirroredStepTest(
        alternative=Hypothesis.P0LessThanP1,
        alpha=individual_alpha,
        n_max=max_sample_size_per_model,
    )
    test.reset()

    # Prepare success array per model.
    success_array_dict = dict()  # model_name -> success_array
    for idx in np.arange(num_models):
        model = model_name_list[idx]
        success_array = success_array_list[idx]
        if shuffle:
            rng.shuffle(success_array)
        success_array_dict[model] = success_array

    # Run pairwise comparisons.
    comparisons_dict = dict()  # (model_name_a, model_name_b) -> Decision
    for idx_a in np.arange(num_models):
        for idx_b in np.arange(idx_a + 1, num_models):
            model_a = model_name_list[idx_a]
            model_b = model_name_list[idx_b]
            array_a = success_array_dict[model_a]
            array_b = success_array_dict[model_b]
            len_common = min(len(array_a), len(array_b))
            array_a = array_a[:len_common]
            array_b = array_b[:len_common]
            # Run the test.
            test_result = test.run_on_sequence(array_a, array_b)
            comparisons_dict[(model_a, model_b)] = test_result.decision

    # Compact Letter Display algorithm to summarize results
    input_list_to_cld = list()
    for key, val in comparisons_dict.items():
        if val != Decision.FailToDecide:
            input_list_to_cld.append(key)
    models_sorted_by_success_rates = [
        model
        for model, _ in sorted(
            success_array_dict.items(),
            key=lambda kv_pair: (np.mean(kv_pair[1]) if len(kv_pair[1]) else 0.0),
            reverse=True,
        )
    ]
    letters_list = _sch.compact_letter_display(
        input_list_to_cld, models_sorted_by_success_rates
    )
    if verbose:
        print("Statistical Test Results (Compact Letter Display):")
    str_padding = max([len(model) for model in models_sorted_by_success_rates])
    return_dict = dict()
    for letters, model in zip(letters_list, models_sorted_by_success_rates):
        return_dict[model] = letters
        num_successes = np.sum(success_array_dict[model])
        num_trials = len(success_array_dict[model])
        if len(success_array_dict[model]) == 0:
            empirical_success_rate = 0.0
        else:
            empirical_success_rate = np.mean(success_array_dict[model])
        if verbose:
            print(
                f"  CLD for {model:<{str_padding}}: {letters}\n"
                f"    Success Rate {num_successes} / {num_trials} = "
                f"{round(empirical_success_rate, 3)}",
            )

    # Ranks are determined if each policy has a unique single letter.
    all_order_determined = all([len(letters) == 1 for letters in letters_list]) and len(
        set(letters_list)
    ) == len(model_name_list)
    if verbose:
        if all_order_determined:
            print(
                (
                    "All models separated with global confidence of "
                    f"{round(global_confidence_level, 5)}."
                )
            )
        else:
            print(
                (
                    "Not all models were separated with global confidence of "
                    f"{round(global_confidence_level, 5)}. Models that share "
                    "a same letter are not separated from each other with "
                    "statistical significance. For more information on how to "
                    "interpret the letters, see: "
                    "https://en.wikipedia.org/wiki/Compact_letter_display.\n"
                )
            )
    return return_dict


def draw_samples_from_beta_posterior(
    success_array: np.ndarray,
    rng: np.random.Generator,
    num_samples: int = 10000,
    alpha_prior: float = 1,
    beta_prior: float = 1,
) -> np.ndarray:
    """Draw samples from the beta posterior distribution given a success array.

    .. deprecated::
        Use ``statistical_comparison_helpers.draw_samples_from_beta_posterior`` instead.
    """
    warnings.warn(
        "sequentialized_barnard_tests.tools.plotting.draw_samples_from_beta_posterior "
        "is deprecated. Use statistical_comparison_helpers.draw_samples_from_beta_posterior instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _sch.draw_samples_from_beta_posterior(
        success_array, rng, num_samples=num_samples,
        alpha_prior=alpha_prior, beta_prior=beta_prior,
    )


def plot_model_comparison(
    model_name_list: List[str],
    success_arrays: List[np.ndarray],
    cld_letters: List[str],
    rng: np.random.Generator,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    add_legend: bool = False,
    unit_width: int = 6,
    height: int = 4,
    dpi: int = 100,
) -> Union[None, plt.Figure]:
    """Makes a violin plot of success rate estimates with corresponding CLD letters.

    .. deprecated::
        Use ``statistical_comparison_helpers.plot_model_comparison`` instead.
    """
    warnings.warn(
        "sequentialized_barnard_tests.tools.plotting.plot_model_comparison is "
        "deprecated. Use statistical_comparison_helpers.plot_model_comparison instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from statistical_comparison_helpers.plotting import plot_model_comparison as _plot
    from statistical_comparison_helpers.posterior import Binary

    return _plot(
        model_name_list,
        success_arrays,
        cld_letters,
        rng,
        score=Binary(),
        plot_mode="posterior",
        output_path=output_path,
        title=title,
        add_legend=add_legend,
        unit_width=unit_width,
        height=height,
        dpi=dpi,
    )
