"""Near-optimal sequential method using optimization-based
approximate solution to the exact optimal stopping PDE.

This module defines the sequential test for the 2x2 Bernoulli
contingency table, as presented and implemented in the paper
Snyder, et. al., (2025), presented at Robotics: Science and Systems 2025.
DOI:
URL: https://www.arxiv.org/abs/2503.10966
"""

import os
import pickle
import subprocess
import sys
import warnings
from typing import Optional, Union

import numpy as np

from sequentialized_barnard_tests.base import (
    Decision,
    Hypothesis,
    SequentialTestBase,
    TestResult,
)


class StepTest(SequentialTestBase):
    """STEP test for comparing two Bernoulli distributions (2x2 Contingency Table).

    This class defines the sequential test for the 2x2 Bernoulli contingency table as a the
    near-optimal, optimization-based solution to the optimal stopping PDE. This procedure
    synthesizes an evaluation policy which defines near-optimal decision making over the
    realization of the natural filtration (rollout results) in the evaluation regime.

    Attributes:
        alternative (Hypothesis): the alternative hypothesis for the statistical test
        n_max (int): the maximal number of trials willing to be evaluated per robot policy
        alpha (float): the maximal allowed Type-1 Error of the evaluation procedure
        shape_parameter (float): the shape parameter of the risk budget.
        use_p_norm (bool): whether to use p_norm shape (True) or partial sums of the zeta function (False).
        policy (List[ArrayLike]): the evaluation decision-making algorithm. Length n_max, each element is an associated array.
        need_new_policy (bool): indicator that a policy has not been previously synthesized for these test parameters.
    """

    def __init__(
        self,
        alternative: Hypothesis,
        n_max: int,
        alpha: float,
        shape_parameter: float = 0.0,
        use_p_norm: bool = False,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the test object.

        Args:
            alternative (Hypothesis): Specification of the alternative hypothesis.
            n_max (int): Maximal sequence length. Must be greater than 0.
            alpha (float): Significance level of the test. Must lie in (0., 1.)
            shape_parameter (float, optional): Shape parameter of the risk budget. Defaults to 0.0.
            use_p_norm (bool, optional): Toggle use of p_norm vs zeta function shape families. Defaults to False.
            random_seed (int, optional): Seed for internal randomness of the test. Defaults to None.
            verbose (bool, optional): If True, print the outputs to stdout. Defaults to False. Defaults to False.

        Raises:
            ValueError: If the n_max and alpha inputs are invalid
        """
        # Handle erroneous inputs
        try:
            assert n_max > 0
            assert 0.0 < alpha < 1.0
        except:
            raise ValueError("Invalid inputs: MUST HAVE n_max > 0, alpha in (0., 1.)")

        # Assign hypothesis
        self.alternative = alternative

        # Assign attributes
        self.n_max = n_max
        self.alpha = alpha
        self.shape_parameter = shape_parameter
        self.use_p_norm = use_p_norm

        # Try to assign policy based on attributes
        try:
            self.load_existing_policy(verbose)
            self.need_new_policy = False
        except:
            self.policy = None
            self.need_new_policy = True

        self._state = None
        self._t = None
        self._current_decision = None
        self.random_seed = random_seed

        # Random generator object initialized and reset in self.reset
        self.reset(verbose)

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        """Runs the test procedure on a single pair of Bernoulli data.

        Args:
            datum_0: Bernoulli datum from the first source.
            datum_1: Bernoulli datum from the second source.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Returns:
            TestResult: Result of the hypothesis test.

        Raise:
            ValueError: If the input data take non-Bernoulli values.
        """
        is_bernoulli_0 = datum_0 in [0, 1]
        is_bernoulli_1 = datum_1 in [0, 1]
        if not (is_bernoulli_0 and is_bernoulli_1):
            raise (ValueError("Input data are not interpretable as Bernoulli."))
        if verbose:
            print(
                (
                    "Update the STEP process given new "
                    f"datum_0 == {datum_0} and datum_1 == {datum_1}."
                )
            )
        if self.policy is None:
            # warnings.warn(f"{self.policy_path}")
            self.load_existing_policy()
            if self._t >= 1:
                warnings.warn(
                    "No policy assigned, so will default to Fail to Decide. Ensure "
                    "that a policy is successfully loaded before stepping through "
                    "the data!"
                )
            if self.policy is None:
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

        # Iterate time state
        self._t += 1

        # Handle case in which we have exceeded n_max
        if self._t > self.n_max:
            warnings.warn(
                "Have exceeded the allowed number of evals; not updating internal states."
            )

            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

        # Update the state to incorporate new data.
        # NOTE: the state is the running SUM, not the running mean!
        self._state[0] += int(datum_0)
        self._state[1] += int(datum_1)

        x = int(self._state[0])
        y = int(self._state[1])

        if y > x:
            #     and self.alternative == Hypothesis.P0LessThanP1) or (
            #     x > y and self.alternative == Hypothesis.P0MoreThanP1
            # ):
            x_absolute = x
            y_absolute = y

            # New policy > old policy (empirically)
            # Therefore, look only to REJECT in standard setting

            # Extract relevant component of policy
            decision_array = self.policy[int(self._t)][x_absolute]

            # Number of non-zero / non-unity policy bins at this x and t
            L = decision_array.shape[0] - 1

            # Highest value of y for which we CONTINUE [i.e., policy = 0]
            critical_zero_y = int(decision_array[0])

            if y_absolute <= critical_zero_y:  # Current state cannot be significant
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            elif (
                y_absolute > critical_zero_y + L
            ):  # Current state is definitely significant
                if self.alternative == Hypothesis.P0LessThanP1:
                    self._current_decision = Decision.AcceptAlternative
                else:
                    self._current_decision = Decision.FailToDecide
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            else:  # Current state is in probabilistic regime
                # random_scalar = np.random.rand(
                #     1
                # )  # TODO: add some kind of seeding procedure to ensure repeatibility
                random_scalar = self.rng.random(1)
                comparator_rv = decision_array[y_absolute - critical_zero_y]
                if (
                    random_scalar <= comparator_rv
                ):  # Then we have probabilistically rejected
                    if self.alternative == Hypothesis.P0LessThanP1:
                        self._current_decision = Decision.AcceptAlternative
                    else:
                        self._current_decision = Decision.FailToDecide
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)
                else:  # Then we have probabilistically continued
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)

                return result

        elif x > y:
            x_absolute = y
            y_absolute = x

            # New policy > old policy (empirically)
            # Therefore, look only to REJECT in reverse setting

            # Extract relevant component of policy
            decision_array = self.policy[int(self._t)][x_absolute]

            # Number of non-zero / non-unity policy bins at this x and t
            L = decision_array.shape[0] - 1

            # Highest value of y for which we CONTINUE [i.e., policy = 0]
            critical_zero_y = int(decision_array[0])

            if y_absolute <= critical_zero_y:  # Current state cannot be significant
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            elif (
                y_absolute > critical_zero_y + L
            ):  # Current state is definitely significant
                if self.alternative == Hypothesis.P0MoreThanP1:
                    self._current_decision = Decision.AcceptAlternative
                else:
                    self._current_decision = Decision.FailToDecide
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            else:  # Current state is in probabilistic regime
                # random_scalar = np.random.rand(
                #     1
                # )  # TODO: add some kind of seeding procedure to ensure repeatibility
                random_scalar = self.rng.random(1)
                comparator_rv = decision_array[y_absolute - critical_zero_y]
                if (
                    random_scalar <= comparator_rv
                ):  # Then we have probabilistically rejected
                    if self.alternative == Hypothesis.P0MoreThanP1:
                        self._current_decision = Decision.AcceptAlternative
                    else:
                        self._current_decision = Decision.FailToDecide
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)
                else:  # Then we have probabilistically continued
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)

                return result
        else:
            # Cannot reject because delta is exactly 0; can only continue!
            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

    def reset(
        self,
        verbose: bool = False,
    ) -> None:
        """Resets the underlying STEP process, including setting the random
        generator object seed.

        Args:
            verbose (bool, optional): If True, print the outputs to stdout.
                Defaults to False.
        """
        self._state = np.zeros(2)
        self._t = int(0)
        self._current_decision = Decision.FailToDecide

        if self.random_seed is not None:
            self.rng = np.random.default_rng(seed=self.random_seed)
        else:
            self.rng = np.random.default_rng()

        if self.alternative == Hypothesis.P0MoreThanP1:
            if verbose:
                print("    Null:        P0 <= P1")
                print("    Alternative: P0 >  P1")
        elif self.alternative == Hypothesis.P0LessThanP1:
            if verbose:
                print("    Null:        P0 >= P1")
                print("    Alternative: P0 <  P1")

    def load_existing_policy(
        self,
        verbose: bool = False,
    ) -> None:
        """Loads a policy based on the STEP test attributes.

        Args:
            verbose (bool, optional): If True, print the outputs to stdout.
                Defaults to False.
        """
        # Determine the path to the synthesis script
        script_dir = os.path.join(os.path.dirname(__file__), "scripts")
        synth_script = os.path.join(script_dir, "synthesize_general_step_policy.py")

        policy_path = os.path.join(
            os.path.dirname(__file__),
            f"policies/n_max_{self.n_max}_alpha_{self.alpha}_shape_parameter_{self.shape_parameter}_pnorm_{self.use_p_norm}/",
            "policy_compressed.pkl",
        )

        try:
            with open(policy_path, "rb") as filename:
                self.policy = pickle.load(filename)
            self.need_new_policy = False
        except Exception as e:
            print(f"Policy not found at {policy_path}. Attempting synthesis...")
            # Build command to synthesize policy
            cmd = [
                sys.executable,
                synth_script,
                "--n_max",
                str(self.n_max),
                "--alpha",
                str(self.alpha),
                "--n_points",
                "129",  # default value, could be parameterized
                "--lambda_value",
                "2.1",  # default, could be parameterized
                "--major_axis_length",
                "1.4",  # default, could be parameterized
                "--log_p_norm",
                str(self.shape_parameter),
                "--use_p_norm",
                str(self.use_p_norm),
            ]
            print(f"Running synthesis command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=script_dir, capture_output=False)
            # Try loading again
            try:
                with open(policy_path, "rb") as filename:
                    self.policy = pickle.load(filename)
                self.need_new_policy = False
                print("Policy synthesis and loading successful.")
            except Exception as e2:
                warnings.warn(
                    f"Unable to synthesize or load policy at {policy_path}. Error: {e2}"
                )
                self.policy = None
                self.need_new_policy = True

        self.policy_path = policy_path

        # Because a policy has been (attempted to be) loaded, run self.reset.
        self.reset(verbose)


class MirroredStepTest(StepTest):
    """Mirrored STEP test for comparing two Bernoulli distributions (2x2 Contingency Table).

    This class defines the sequential test for the 2x2 Bernoulli contingency table as a the
    near-optimal, optimization-based solution to the optimal stopping PDE. This procedure
    synthesizes an evaluation policy which defines near-optimal decision making over the
    realization of the natural filtration (rollout results) in the evaluation regime.

    Attributes:
        alternative (Hypothesis): the alternative hypothesis for the statistical test
        n_max (int): the maximal number of trials willing to be evaluated per robot policy
        alpha (float): the maximal allowed Type-1 Error of the evaluation procedure
        shape_parameter (float): the shape parameter of the risk budget.
        use_p_norm (bool): whether to use p_norm shape (True) or partial sums of the zeta function (False).
        policy (List[ArrayLike]): the evaluation decision-making algorithm. Length n_max, each element is an associated array.
        need_new_policy (bool): indicator that a policy has not been previously synthesized for these test parameters.
    """

    def __init__(
        self,
        alternative: Hypothesis,
        n_max: int,
        alpha: float,
        shape_parameter: float = 0.0,
        use_p_norm: bool = False,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """Initializes the test object.

        Args:
            alternative (Hypothesis): Specification of the alternative hypothesis.
            n_max (int): Maximal sequence length. Must be greater than 0.
            alpha (float): Significance level of the test. Must lie in (0., 1.)
            shape_parameter (float, optional): Shape parameter of the risk budget. Defaults to 0.0.
            use_p_norm (bool, optional): Toggle use of p_norm vs zeta function shape families. Defaults to False.
            random_seed (int, optional): Seed for internal randomness of the test. Defaults to None.
            verbose (bool, optional): If True, print the outputs to stdout. Defaults to False. Defaults to False.
        """
        super().__init__(
            alternative,
            n_max,
            alpha,
            shape_parameter,
            use_p_norm,
            random_seed,
            verbose,
        )

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        """Runs the test procedure on a single pair of Bernoulli data.

        Args:
            datum_0: Bernoulli datum from the first source.
            datum_1: Bernoulli datum from the second source.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Returns:
            TestResult: Result of the hypothesis test.

        Raise:
            ValueError: If the input data take non-Bernoulli values.
        """
        is_bernoulli_0 = datum_0 in [0, 1]
        is_bernoulli_1 = datum_1 in [0, 1]
        if not (is_bernoulli_0 and is_bernoulli_1):
            raise (ValueError("Input data are not interpretable as Bernoulli."))
        if verbose:
            print(
                (
                    "Update the STEP process given new "
                    f"datum_0 == {datum_0} and datum_1 == {datum_1}."
                )
            )
        if self.policy is None:
            self.load_existing_policy()
            if self._t >= 1:
                warnings.warn(
                    "No policy assigned, so will default to Fail to Decide. Ensure "
                    "that a policy is successfully loaded before stepping through "
                    "the data!"
                )

            if self.policy is None:
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

        # Iterate time state
        self._t += 1

        # Handle case in which we have exceeded n_max
        if self._t > self.n_max:
            warnings.warn(
                "Have exceeded the allowed number of evals; not updating internal states."
            )

            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result

        # Update the state to incorporate new data.
        # NOTE: the state is the running SUM, not the running mean!
        self._state[0] += int(datum_0)
        self._state[1] += int(datum_1)

        x = int(self._state[0])
        y = int(self._state[1])

        if y > x:
            #     and self.alternative == Hypothesis.P0LessThanP1) or (
            #     x > y and self.alternative == Hypothesis.P0MoreThanP1
            # ):
            x_absolute = x
            y_absolute = y

            # New policy > old policy (empirically)
            # Therefore, look only to REJECT in standard setting

            # Extract relevant component of policy
            try:
                decision_array = self.policy[int(self._t)][x_absolute]
            except:
                print("Time: ", self._t)
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result
            # Number of non-zero / non-unity policy bins at this x and t
            L = decision_array.shape[0] - 1

            # Highest value of y for which we CONTINUE [i.e., policy = 0]
            critical_zero_y = int(decision_array[0])

            if y_absolute <= critical_zero_y:  # Current state cannot be significant
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            elif (
                y_absolute > critical_zero_y + L
            ):  # Current state is definitely significant
                if self.alternative == Hypothesis.P0LessThanP1:
                    self._current_decision = Decision.AcceptAlternative
                else:
                    self._current_decision = Decision.AcceptNull
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            else:  # Current state is in probabilistic regime
                # random_scalar = np.random.rand(
                #     1
                # )  # TODO: add some kind of seeding procedure to ensure repeatibility
                random_scalar = self.rng.random(1)
                comparator_rv = decision_array[y_absolute - critical_zero_y]
                if (
                    random_scalar <= comparator_rv
                ):  # Then we have probabilistically rejected
                    if self.alternative == Hypothesis.P0LessThanP1:
                        self._current_decision = Decision.AcceptAlternative
                    else:
                        self._current_decision = Decision.AcceptNull
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)
                else:  # Then we have probabilistically continued
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)

                return result

        elif x > y:
            x_absolute = y
            y_absolute = x

            # New policy > old policy (empirically)
            # Therefore, look only to REJECT in reverse setting

            # Extract relevant component of policy
            decision_array = self.policy[int(self._t)][x_absolute]

            # Number of non-zero / non-unity policy bins at this x and t
            L = decision_array.shape[0] - 1

            # Highest value of y for which we CONTINUE [i.e., policy = 0]
            critical_zero_y = int(decision_array[0])

            if y_absolute <= critical_zero_y:  # Current state cannot be significant
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            elif (
                y_absolute > critical_zero_y + L
            ):  # Current state is definitely significant
                if self.alternative == Hypothesis.P0MoreThanP1:
                    self._current_decision = Decision.AcceptAlternative
                else:
                    self._current_decision = Decision.AcceptNull
                info = {"Time": self._t, "State": self._state}
                result = TestResult(self._current_decision, info)

                return result

            else:  # Current state is in probabilistic regime
                # random_scalar = np.random.rand(
                #     1
                # )  # TODO: add some kind of seeding procedure to ensure repeatibility
                random_scalar = self.rng.random(1)
                comparator_rv = decision_array[y_absolute - critical_zero_y]
                if (
                    random_scalar <= comparator_rv
                ):  # Then we have probabilistically rejected
                    if self.alternative == Hypothesis.P0MoreThanP1:
                        self._current_decision = Decision.AcceptAlternative
                    else:
                        self._current_decision = Decision.AcceptNull
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)
                else:  # Then we have probabilistically continued
                    info = {"Time": self._t, "State": self._state}
                    result = TestResult(self._current_decision, info)

                return result
        else:
            # Cannot reject because delta is exactly 0; can only continue!
            info = {"Time": self._t, "State": self._state}
            result = TestResult(self._current_decision, info)

            return result
