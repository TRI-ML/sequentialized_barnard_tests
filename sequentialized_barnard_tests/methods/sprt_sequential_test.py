import numpy as np
from base import Decision, SequentialTestBase, TestResult
from numpy.typing import ArrayLike
from utils.utils_SPRT_oracle import (
    bernoulli_KL,
    calc_exact_zeta,
    calibrate_sprt,
    compute_maxFPR_p,
    compute_middle_p,
)


class SPRT_Sequential_Test(SequentialTestBase):
    def __init__(self, Nmax: int, alpha: float, p0: float, p1: float):
        """Initialize a near-optimal finite-time 2SPRT with access to privileged information (p0, p1).

        Args:
            Nmax (int): Maximum number of trials for this testing procedure. Must be positive
            alpha (float): Tolerated level of the risk of error. Lies in [0, 1]
            p0 (float): Specifies true generating mean of base policy. Lies in [0, 1].
            p1 (float): Specifies true generating mean of new policy. Lies in [0, 1].
            data_dependent (bool, optional): Specifies whether to use data-dependent (stronger, but unrealizable) calibration. Defaults to False.

        Raises:
            NotImplementedError: If the policy cannot be found, the test class will not instantiate.
        """

        assert alpha > 0.0 and Nmax > 0
        assert p0 > 0.0 and p1 > 0.0
        assert p0 < 1.0 and p1 < 1.0

        self.Nmax = Nmax
        self.alpha = alpha

        # Assign the values of the true point null and point alternative hypotheses
        self.p0 = p0
        self.p1 = p1
        self.p_mid = compute_maxFPR_p(p0, p1)

        # Get the associated values of (a_star, b_star) based on {p0, p1, Nmax, and alpha}
        self.a_star = None
        self.b_star = None

        # Get appropriate calibration and assign self.a_star, self.b_star (LR cutoffs)
        try:
            self.calibrate_AB()
        except:
            print(
                "Parameters do not match any pre-calibrated case, and calibration is not implemented at this time."
            )
            print(
                "Setting to the reasonable but arbitrary values of A* = 1/20, B* = 20"
            )
            self.set_arbitrary_Astar(0.05)
            self.set_arbitrary_Bstar(20.0)

        # Initialize test time and 2SPRT state
        self.test_time = 0
        self.test_state = 1.0  # Note that the state here is likelihood ratio

    def step(self, datum_0: float, datum_1: float):
        """Iterates near-optimal test using compressed policy representation

        Args:
            new_datum (np.array(2)): New result for each policy rollout, in the order [base_policy_result, new_policy_result].

        Returns:
            test_completed (bool): Whether the test has terminated or not
        """
        # Iterate time
        self.test_time += 1

        assert np.isclose(np.abs(datum_0 - 0.5), 0.5)
        assert np.isclose(np.abs(datum_1 - 0.5), 0.5)

        base_multiplier = 1.0
        if datum_0 >= 0.5:
            # base policy had success
            base_multiplier = self.p0 / self.p_mid
        else:
            # base policy had failure
            base_multiplier = (1.0 - self.p0) / (1.0 - self.p_mid)

        alt_multiplier = 1.0
        if datum_1 >= 0.5:
            # new policy had success
            alt_multiplier = self.p1 / self.p_mid
        else:
            # new policy had failure
            alt_multiplier = (1.0 - self.p1) / (1.0 - self.p_mid)

        self.test_state *= base_multiplier
        self.test_state *= alt_multiplier

        result = TestResult
        result.decision = Decision.FailToDecide

        result.info["time_of_decision"] = self.test_time

        if self.test_state >= self.b_star:
            # REJECT
            result.decision = Decision.AcceptAlternative
        elif self.test_state <= self.a_star:
            # ACCEPT
            result.decision = Decision.AcceptNull
        else:
            # No significant result...
            pass

        return result

    def run_on_sequence(
        self, sequence_0: ArrayLike, sequence_1: ArrayLike
    ) -> TestResult:
        """Run the Lai procedure on a sequence of data from p0 and p1. Modifies abstract class
        definition to account for the desire for a float representation of the sequence results.

        Args:
            sequence_0 (Arraylike): Sequence of i.i.d. bernoulli trials of Policy 0
            sequence_1 (Arraylike): Sequence of i.i.d. bernoulli trials of Policy 1

        Returns:
            result (TestResult): Test result class with decision and optional information
        """
        self.reset()
        if not (len(sequence_0) == len(sequence_1)):
            raise (ValueError("The two input sequences must have the same size."))
        elif not (len(sequence_0) <= self.Nmax):
            raise (ValueError("The sequences cannot exceed Nmax in length."))

        result = None
        for idx in range(len(sequence_0)):
            result = self.step(float(sequence_0[idx]), float(sequence_1[idx]))
            if not result.decision == Decision.FailToDecide:
                break

        return result

    def calibrate_AB(
        self, n_trials=5000, max_power_nominal=0.999, n_parallel_evaluations=45
    ):
        calibrated_parameters = calibrate_sprt(
            self.Nmax,
            self.alpha,
            self.p0,
            self.p1,
            n_trials=n_trials,
            max_power_nominal=max_power_nominal,
            n_parallel_evaluations=n_parallel_evaluations,
        )
        self.a_star = calibrated_parameters[0]
        self.b_star = calibrated_parameters[1]

    def set_arbitrary_Astar(self, new_a: float):
        assert new_a < 1.0 and new_a > 0.0
        self.a_star = new_a

    def set_arbitrary_Bstar(self, new_b: float):
        assert new_b > 1.0
        self.b_star = new_b

    def reset(self):
        self.test_time = 0
        self.test_state = 1.0  # Note that the state here is likelihood ratio
