from setuptools import find_packages, setup

setup(
    name="sequentialized_barnard_tests",
    version="0.0.4",
    description="Sequential statistical hypothesis testing for two-by-two contingency tables.",
    authors=["David Snyder", "Haruki Nishimura"],
    author_emails=["dasnyder@princeton.edu", "haruki.nishimura@tri.global"],
    license="MIT",
    packages=find_packages(),
    package_data={
        "sequentialized_barnard_tests": [
            "scripts/synthesize_general_step_policy.py",
            "data/lai_calibration_data.npy",
            "policies/n_max_100_alpha_0.05_shape_parameter_0.0_pnorm_False/policy_compressed.pkl",
            "policies/n_max_200_alpha_0.05_shape_parameter_0.0_pnorm_False/policy_compressed.pkl",
            "policies/n_max_500_alpha_0.05_shape_parameter_0.0_pnorm_False/policy_compressed.pkl",
        ],
    },
    install_requires=[
        "binomial_cis",
        "matplotlib",
        "numpy>=1.20",
        "pytest",
        "scipy",
        "tqdm",
    ],
)
