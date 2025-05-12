import numpy as np
from matplotlib import pyplot as plt

from sequentialized_barnard_tests.utils.utils_general import (
    bivariate_bernoulli_kl,
    construct_kl_spaced_points_array_via_binary_expansion,
)

if __name__ == "__main__":
    points_array = construct_kl_spaced_points_array_via_binary_expansion(500)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(points_array, np.arange(1, points_array.shape[0] + 1), "r*", markersize=10)
    ax.set_xlabel("Null hypothesis (p, p)")
    ax.set_ylabel("Count of null hypothesis")
    ax.set_title("Distribution of Null Hypotheses, KL-Equispaced")

    fig.savefig("scripts/im/spacing/null_hypotheses_on_real_line.png")

    kl_divergences = np.zeros(points_array.shape[0] - 1)
    for k in range(points_array.shape[0] - 1):
        kl_divergences[k] = 0.5 * (
            bivariate_bernoulli_kl(
                [points_array[k], points_array[k]],
                [points_array[k + 1], points_array[k + 1]],
            )
            + bivariate_bernoulli_kl(
                [points_array[k + 1], points_array[k + 1]],
                [points_array[k], points_array[k]],
            )
        )

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.plot(points_array[:-1], kl_divergences, "r*", markersize=10)
    ax2.set_xlabel("Null hypothesis (p, p)")
    ax2.set_ylabel("KL Divergence to Next Null")
    ax2.set_title("Verifying Equispaced KL-Divergence")

    fig2.savefig("scripts/im/spacing/KL_divergences_in_order.png")
