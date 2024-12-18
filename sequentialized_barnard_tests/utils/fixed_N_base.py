import numpy as np


class mean_test_evaluator:
    def __init__(self, mu_0, side, delta, N):
        self.mu_0 = mu_0
        self.side = side
        self.delta = delta
        self.N = N

    def run_test(self, xbar, mu_0=None):
        pass


class mean_interval_estimator:
    def __init__(self, side, delta, N):
        self.side = side
        self.delta = delta
        self.N = N

    def calc_interval(self, xbar):
        pass
