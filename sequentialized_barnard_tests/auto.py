"""
Factory function for automatic selection of STEP or Lai based on n_max.
"""

from sequentialized_barnard_tests.lai import MirroredLaiTest, LaiTest
from sequentialized_barnard_tests.step import MirroredStepTest, StepTest


def _validate_critical_n_max(critical_n_max: int) -> None:
    if critical_n_max < 0:
        raise ValueError("critical_n_max must be nonnegative.")


def get_test(
    n_max: int,
    alternative,
    alpha: float,
    verbose: bool = False,
    *,
    critical_n_max: int = 500,
    **kwargs,
):
    """
    Factory function to select StepTest or LaiTest based on n_max.
    Uses LaiTest for n_max > critical_n_max, otherwise StepTest.

    Shared arguments:
        n_max (int): Maximal sequence length.
        alternative: Specification of the alternative hypothesis.
        alpha (float): Significance level of the test.
        verbose (bool, optional): If True, print outputs to stdout.
        critical_n_max (int, optional): Largest n_max that should use STEP.
            Values above this threshold use Lai. Defaults to 500; set to 0
            to route all positive n_max values to LaiTest.
    Additional arguments for each class can be passed as keyword arguments.
    """
    _validate_critical_n_max(critical_n_max)
    if n_max > critical_n_max:
        if verbose:
            print(f"Using LaiTest for n_max > {critical_n_max}")
        return LaiTest(alternative, n_max, alpha, verbose=verbose, **kwargs)
    else:
        if verbose:
            print(f"Using StepTest for n_max <= {critical_n_max}")
        return StepTest(alternative, n_max, alpha, verbose=verbose, **kwargs)


def get_mirrored_test(
    n_max: int,
    alternative,
    alpha: float,
    verbose: bool = False,
    *,
    critical_n_max: int = 500,
    **kwargs,
):
    """
    Factory function to select MirroredStepTest or MirroredLaiTest based on n_max.
    Uses MirroredLaiTest for n_max > critical_n_max, otherwise MirroredStepTest.

    Shared arguments:
        n_max (int): Maximal sequence length.
        alternative: Specification of the alternative hypothesis.
        alpha (float): Significance level of the test.
        verbose (bool, optional): If True, print outputs to stdout.
        critical_n_max (int, optional): Largest n_max that should use STEP.
            Values above this threshold use Lai. Defaults to 500; set to 0
            to route all positive n_max values to MirroredLaiTest.
    Additional arguments for each class can be passed as keyword arguments.
    """
    _validate_critical_n_max(critical_n_max)
    if n_max > critical_n_max:
        if verbose:
            print(f"Using MirroredLaiTest for n_max > {critical_n_max}")
        return MirroredLaiTest(alternative, n_max, alpha, verbose=verbose, **kwargs)
    else:
        if verbose:
            print(f"Using MirroredStepTest for n_max <= {critical_n_max}")
        return MirroredStepTest(alternative, n_max, alpha, verbose=verbose, **kwargs)
