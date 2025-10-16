"""
Factory function for automatic selection of STEP or Lai based on n_max.
"""

from sequentialized_barnard_tests.step import MirroredStepTest, StepTest
from sequentialized_barnard_tests.lai import MirroredLaiTest, LaiTest


def get_test(n_max: int, alternative, alpha: float, verbose: bool = False, **kwargs):
    """
    Factory function to select StepTest or LaiTest based on n_max.
    Uses LaiTest for n_max > 500, otherwise StepTest.

    Shared arguments:
        n_max (int): Maximal sequence length.
        alternative: Specification of the alternative hypothesis.
        alpha (float): Significance level of the test.
        verbose (bool, optional): If True, print outputs to stdout.
    Additional arguments for each class can be passed as keyword arguments.
    """
    if n_max > 500:
        if verbose:
            print("Using LaiTest for n_max > 500")
        return LaiTest(alternative, n_max, alpha, verbose=verbose, **kwargs)
    else:
        if verbose:
            print("Using StepTest for n_max <= 500")
        return StepTest(alternative, n_max, alpha, verbose=verbose, **kwargs)


def get_mirrored_test(
    n_max: int, alternative, alpha: float, verbose: bool = False, **kwargs
):
    """
    Factory function to select MirroredStepTest or MirroredLaiTest based on n_max.
    Uses MirroredLaiTest for n_max > 500, otherwise MirroredStepTest.

    Shared arguments:
        n_max (int): Maximal sequence length.
        alternative: Specification of the alternative hypothesis.
        alpha (float): Significance level of the test.
        verbose (bool, optional): If True, print outputs to stdout.
    Additional arguments for each class can be passed as keyword arguments.
    """
    if n_max > 500:
        if verbose:
            print("Using MirroredLaiTest for n_max > 500")
        return MirroredLaiTest(alternative, n_max, alpha, verbose=verbose, **kwargs)
    else:
        if verbose:
            print("Using MirroredStepTest for n_max <= 500")
        return MirroredStepTest(alternative, n_max, alpha, verbose=verbose, **kwargs)
