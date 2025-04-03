"""calibrate_lai_offline.py

This script computes the regularization parameter `c` for LaiTest and MirroredLaiTest
for various values of n_max (i.e., maximum sequence length) and significance levels.

Example Usage:
    python scripts/calibrate_lai_offline.py \
        --n_mc_samples 1000 \
"""

import argparse
import os
import time
from typing import List

import numpy as np

from sequentialized_barnard_tests import Hypothesis
from sequentialized_barnard_tests.lai import MirroredLaiTest


def main(
    n_mc_samples: int,
    n_max_list: List[int],
    alpha_list: List[float],
) -> None:
    """Main function that runs calibration of the Lai test under different values of
    n_max and alpha. The results are saved as a .npy file.

    The saved data is a dict that contains each (alpha, n_max) as the key and the
    corresponding regularizer `c` as the value.

    Args:
        n_mc_samples: Number of Monte Carlo samples for each calibration.
        n_max_list: A list of n_max (i.e., maximum sequence length).
        alpha_list: A list of alpha (i.e., significance level).
        seed: Random seed.
    """
    calibration_results_dict = dict()
    # Run calibration for each test case.
    num_cases = len(n_max_list) * len(alpha_list)
    counter = 0
    start_time = time.time()
    for alpha in alpha_list:
        for n_max in n_max_list:
            counter += 1
            test_case = (alpha, n_max)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(
                (
                    f"(Elapsed: {elapsed}) Case {counter}/{num_cases}: "
                    f"(n_max, alpha) == {(n_max, alpha)}"
                )
            )
            lai_test = MirroredLaiTest(
                alternative=Hypothesis.P0LessThanP1,
                n_max=n_max,
                alpha=alpha,
                calibrate_regularizer=True,
                use_offline_calibration=False,
                n_calibration_sequences=n_mc_samples,
            )
            calibration_results_dict[test_case] = lai_test.c

    # Save results.
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "../sequentialized_barnard_tests/data",
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lai_calibration_data.npy")
    with open(save_path, "wb") as file:
        np.save(file, calibration_results_dict)
    print(f"Calibration file saved to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script computes the regularization parameter `c` for LaiTest and "
            "MirroredLaiTest for various values of n_max (i.e., maximum sequence "
            "length) and significance levels. The results are saved to a .npy file at "
            "'sequentialized_barnard_tests/data'."
        )
    )
    parser.add_argument(
        "--n_mc_samples",
        type=int,
        default=10000,
        help=(
            "Number of Monte Carlo simulations for calibrating the statistical test. "
            "Defaults to 10000."
        ),
    )
    args = parser.parse_args()
    alpha_list = [
        0.001,
        0.003,
        0.005,
        0.007,
        0.009,
        0.01,
        0.03,
        0.05,
        0.07,
        0.09,
        0.1,
        0.3,
        0.5,
        0.55,
        0.7,
        0.9,
    ]
    n_max_list = [10, 20, 40, 80, 100, 200, 400, 800, 1000, 2000, 4000]
    main(args.n_mc_samples, n_max_list, alpha_list)
