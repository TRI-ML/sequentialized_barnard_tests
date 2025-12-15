from typing import Dict, List, Optional, Tuple, Union

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.step import MirroredStepTest


def compact_letter_display(
    significant_pair_list: List[Tuple[str, str]],
    sorted_model_list: List[str],
) -> List[str]:
    """Generates Compact Letter Display (CLD) given a list of significant
    pairs and a list of models. CLD is Based on "An Algorithm for a
    Letter-Based Representation of All-Pairwise Comparisons" by Piepho
    (2004).

    Args:
        significant_pair_list: A list containing tuples of model names that
            were deemed significantly different by each A/B test.
        sorted_model_list: A list of model names sorted by performance in
            descending order.

    Returns:
        A list of letters representing CLD for the corresponding models.
    """
    num_models = len(sorted_model_list)

    # Map model names to indices.
    model_to_index = {model: idx for idx, model in enumerate(sorted_model_list)}
    # Convert significant pairs from names to indices.
    significant_index_pairs = [
        (model_to_index[m1], model_to_index[m2]) for m1, m2 in significant_pair_list
    ]

    # --- Inner helper to remove redundant columns ---
    def remove_redundant_columns(matrix):
        changed = True
        while changed:
            changed = False
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i != j:
                        indices_i = {idx for idx, char in enumerate(matrix[i]) if char}
                        indices_j = {idx for idx, char in enumerate(matrix[j]) if char}
                        if indices_i.issubset(indices_j):
                            matrix.pop(i)
                            changed = True
                            break
                if changed:
                    break
        return matrix

    # --- Main algorithm ---
    # Start with a single column of 'a's for all models.
    letter_matrix = [["a"] * num_models]

    # For each significant pair, update the letter matrix.
    for model_idx1, model_idx2 in significant_index_pairs:
        while any(col[model_idx1] and col[model_idx2] for col in letter_matrix):
            for col_index, letter_column in enumerate(letter_matrix):
                if letter_column[model_idx1] and letter_column[model_idx2]:
                    new_column = letter_column.copy()
                    new_column[model_idx1] = ""
                    letter_column[model_idx2] = ""
                    letter_matrix[col_index] = letter_column
                    letter_matrix.append(new_column)
                    letter_matrix = remove_redundant_columns(letter_matrix)
                    break  # re-check with the while condition

    # --- Reassign letters based on sorted columns ---
    def first_nonempty_position(column):
        for pos, char in enumerate(column):
            if char:
                return pos
        return len(column)

    letter_matrix.sort(key=first_nonempty_position)

    for idx, column in enumerate(letter_matrix):
        replacement_letter = chr(ord("a") + idx)
        letter_matrix[idx] = [replacement_letter if char else "" for char in column]

    # --- Build final CLD output for each model ---
    final_display = []
    for model_idx in range(num_models):
        letters = "".join(
            letter_matrix[col_idx][model_idx]
            for col_idx in range(len(letter_matrix))
            if letter_matrix[col_idx][model_idx]
        )
        final_display.append(letters)

    return final_display


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
    letters_list = compact_letter_display(
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
    These samples can be used to estimate the posterior distribution of the
    success rate of a Bernoulli process. Note that the default prior parameters
    of (1, 1) correspond to a uniform prior.

    Args:
        success_array: A binary array with True/False indicating success/failure.
        rng: A numpy random Generator instance.
        num_samples: Optional number of samples to draw. Defaults to 10000.
        alpha_prior: Optional alpha parameter of the beta prior. Defaults to 1.
        beta_prior: Optional beta parameter of the beta prior. Defaults to 1.

    Returns:
        Samples drawn from the beta posterior distribution.
    """
    n_trials = len(success_array)
    n_successes = np.sum(success_array)
    n_failures = n_trials - n_successes
    posterior = stats.beta(alpha_prior + n_successes, beta_prior + n_failures)
    return posterior.rvs(num_samples, random_state=rng)


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
    """Makes a violin plot of success rate estimates with corresponding CLD letters
    for policy comparison.

    Args:
        model_name_list: A list of model names.
        success_arrays: A list of arrays indicating success/failure for each model.
        cld_letters: A list of CLD letters corresponding to each model.
        rng: A numpy random Generator instance for posterior sampling.
        output_path: Optional file path to save the plot. If None, the plot will not
            be saved but returned as a matplotlib Figure object. Defaults to None.
        title: Optional title for the plot. Defaults to None.
        add_legend: Whether to show legend on the plot. Defaults to False.
        unit_width: Figure width per model. Defaults to 6.
        height: Figure height. Defaults to 4.
        dpi: Resolution of the saved plot. Defaults to 100.

    Returns:
        If output_path is None, returns a matplotlib Figure object containing
        the plot. Otherwise, saves the plot to the specified path and returns None.
    """
    num_models = len(model_name_list)

    posterior_samples = []
    means = []

    for success_array in success_arrays:
        samples = draw_samples_from_beta_posterior(success_array, rng)
        posterior_samples.append(samples)
        means.append(np.mean(samples))

    fig, ax = plt.subplots(figsize=(max(unit_width, num_models), height), dpi=dpi)

    cmap = get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_models)]

    parts = ax.violinplot(
        posterior_samples,
        positions=np.arange(num_models),
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    # Add CLD labels
    for i, (x, y, label) in enumerate(zip(np.arange(num_models), means, cld_letters)):
        ax.text(
            x + 0.15,
            y + 0.03,
            label,
            fontsize=12,
            fontweight="bold",
            color="black",
            verticalalignment="center",
            zorder=4,
        )

    ax.set_xticks(np.arange(num_models))
    ax.set_xticklabels(model_name_list, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success Rate")
    if title is not None:
        ax.set_title(title)
    if add_legend:
        ax.legend(parts["bodies"], model_name_list, loc="best")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved a PNG plot to {output_path}")
    else:
        return fig
