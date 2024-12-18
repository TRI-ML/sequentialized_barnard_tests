import numpy as np
from binomial_cis import binom_ci  # Joe Vincent's package
from methods.fixed_N_base import mean_interval_estimator, mean_test_evaluator


# Binomial Mean Hypothesis Test -- derived from Joe Vincent's Binomial CIs
class binomial_mean_test_evaluator(mean_test_evaluator):
    def __init__(self, mu_0, side, delta, N):
        super().__init__(mu_0, side, delta, N)
        # NOTE: Null Hypothesis: mu = mu_0

    def run_test(self, xbar, mu_0=None):
        if mu_0 is None:
            mu_0 = self.mu_0

        number_of_ones = int(xbar * self.N + 1e-8)

        # Verify that xbar is sensible as a mean of a binomial
        try:
            assert np.allclose(number_of_ones, xbar * self.N, atol=1e-04)
        except:
            print(
                "Encountered an exception to the float-int conversion. Results follow: "
            )
            print()
            print("Number of ones: ", number_of_ones)
            print("Float approximation: ", xbar * self.N)

        if self.side < -0.5:
            # Alternative: mu < mu_0
            # implies use upper bound on empirical mean and compare with mu_0
            lb = 0.0
            ub = binom_ci(number_of_ones, self.N, self.delta, "ub")
            if ub < mu_0:
                return 1  # Reject the null and accept alternative: mu < mu_0
        elif self.side > 0.5:
            ub = 1.0
            lb = binom_ci(number_of_ones, self.N, self.delta, "lb")
            if lb > mu_0:
                return 1  # Reject the null and accept alternative: mu > mu_0
        else:
            lb, ub = binom_ci(number_of_ones, self.N, self.delta, "lb,ub")
            if lb > mu_0 or ub < mu_0:
                return 1  # Reject the null and accept alternative: mu =/= mu_0

        return 0  # Fail to reject the null as we have exhausted all cases of rejection
        # Important note: "exhausted all cases of rejection" is about enumeration in code; for any given usage, there is only ONE relevant case depending on the particular choice of alternative. This is NOT multiple testing!


# Binomial (Mean) Confidence Intervals -- Joe Vincent
class binomial_mean_interval_estimator(mean_interval_estimator):
    def __init__(self, side, delta, N):
        super().__init__(side, delta, N)

    def calc_interval(self, xbar):
        number_of_ones = int(xbar * self.N + 1e-8)

        # Verify that xbar is sensible as a mean of a binomial
        try:
            assert np.allclose(number_of_ones, xbar * self.N, atol=1e-04)
        except:
            print(
                "Encountered an exception to the float-int conversion. Results follow: "
            )
            print()
            print("Number of ones: ", number_of_ones)
            print("Float approximation: ", xbar * self.N)

        if self.side < -0.5:
            lb = binom_ci(number_of_ones, self.N, self.delta, "lb", verbose=False)
            ub = 1.0
        elif self.side > 0.5:
            lb = 0.0
            ub = binom_ci(number_of_ones, self.N, self.delta, "ub", verbose=False)
        else:
            lb, ub = binom_ci(
                number_of_ones, self.N, self.delta, "lb,ub", verbose=False
            )

        return lb, ub
