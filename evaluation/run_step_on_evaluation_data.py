import argparse
import json
import os
from datetime import datetime

import numpy as np

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.step import MirroredStepTest

# from ..sequentialized_barnard_tests.lai import MirroredLaiTest
# from ..sequentialized_barnard_tests.savi import MirroredOracleSaviTest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script runs a pre-computed STEP policy on real data. The data "
            "is assumed to be stored as '<base_path>/data/<project_folder>/<file_name>.npy.' "
            "Further, the data is assumed to be of shape (N, 2) corresponding to the success-"
            "failure values observed in evaluation trials. Input parameters determine the nature "
            "of the STEP test, certain save options, and the data project to access. "
        )
    )
    parser.add_argument(
        "-p",
        "--data_folder",
        type=str,
        default="example_clean_spill",
        help=(
            "Relative path added to <base_path>/data/ which specifies the desired "
            "evaluation data folder, from which to run the STEP test. Defaults to ''."
        ),
    )
    parser.add_argument(
        "-f",
        "--data_file",
        type=str,
        default="TRI_CLEAN_SPILL_v4.npy",
        help=(
            "Relative path added to <base_path>/data/<project_folder>/ which specifies the desired "
            "evaluation data on which to run the STEP test. Defaults to ''."
        ),
    )
    parser.add_argument(
        "-n",
        "--n_max",
        type=int,
        default=200,
        help=(
            "Maximum number of robot policy evals (per policy) in the evaluation procedure. "
            "Defaults to 200."
        ),
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help=(
            "Maximal allowed Type-1 error rate of the statistical testing procedure. "
            "Defaults to 0.05."
        ),
    )
    parser.add_argument(
        "-pz",
        "--log_p_norm",
        type=float,
        default=0.0,
        help=(
            "Rate at which risk is accumulated, reflecting user's belief about underlying "
            "likelihood of different alternatives and nulls being true. If using a p_norm "
            ", this variable is equivalent to log(p). If not using a p_norm, this is the "
            "argument to the zeta function, partial sums of which give the shape of the risk budget."
            "Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "-up",
        "--use_p_norm",
        type=bool,
        default=False,
        help=(
            "Toggle whether to use p_norm or zeta function shape family for the risk budget. "
            "If True, uses p_norm shape; else, uses zeta function shape family. "
            "Defaults to False (zeta function partial sum family)."
        ),
    )
    parser.add_argument(
        "-so",
        "--save_output",
        type=bool,
        default=True,
        help=(
            "Toggle whether save the evaluation output in a timestamped config file, as opposed to only printing to terminal. "
            "If True, the evaluation result is saved in a json file in the same directory as the data. "
            "Defaults to True."
        ),
    )
    parser.add_argument(
        "-uda",
        "--use_default_alternative",
        type=bool,
        default=True,
        help=(
            "Determines the alternative to use in the testing procedure. If True, uses the alternative p0 < p1. "
            "Otherwise, uses the alternative p0 > p1. Defaults to True."
        ),
    )

    # Parse the input arguments
    args = parser.parse_args()

    # Define the base path so that the file can be run from any directory
    # base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Define the data folder path and data file path
    data_folder_path = os.path.join(base_path, "data", args.data_folder)
    data_file_path = os.path.join(data_folder_path, args.data_file)

    # Assign step test alternative hypothesis
    if args.use_default_alternative:
        alt = Hypothesis.P0LessThanP1
        alt_str = "P0 < P1"
        null_str = "P0 >= P1"
    else:
        alt = Hypothesis.P0MoreThanP1
        alt_str = "P0 > P1"
        null_str = "P0 <= P1"
    # Initialize the STEP test
    mirrored_step_test = MirroredStepTest(
        alternative=alt,
        n_max=args.n_max,
        alpha=args.alpha,
        shape_parameter=args.log_p_norm,
        use_p_norm=args.use_p_norm,
    )

    # Load data and evaluate
    data = np.load(data_file_path)

    result = mirrored_step_test.run_on_sequence(data[:, 0], data[:, 1])

    decision_str = "No decision"
    if result.decision == Decision.AcceptAlternative:
        decision_str = alt_str
    elif result.decision == Decision.AcceptNull:
        decision_str = null_str
    else:
        pass

    # Print to terminal
    print(
        f"Evaluation result for data stored at: {os.path.join(data_folder_path, args.data_file)}"
    )
    print(f"Decision: {result.decision} --> we conclude that {decision_str}")
    print(f"Time of decision: {result.info['Time']}")

    # Save to json file
    if args.save_output:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        evaluation_dict = {
            "result": [
                {
                    "Hypothesis": alt_str,
                    "Decision": decision_str,
                    "Time": result.info["Time"],
                }
            ],
            "method": "STEP",
            "params": [
                {
                    "n_max": args.n_max,
                    "alpha": args.alpha,
                }
            ],
            "p0_hat": np.mean(data[:, 0]),
            "p1_hat": np.mean(data[:, 1]),
            "N": data.shape[0],
        }

        with open(
            data_folder_path + "/" + f"evaluation_result_{formatted_time}.json", "w"
        ) as fp:
            json.dump(evaluation_dict, fp)
