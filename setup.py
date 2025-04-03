from setuptools import find_packages, setup

setup(
    name="sequentialized_barnard_tests",
    version="0.0.1",
    description="Sequential statistical hypothesis testing for two-by-two contingency tables.",
    authors=["David Snyder", "Haruki Nishimura"],
    author_emails=["dasnyder@princeton.edu", "haruki.nishimura@tri.global"],
    license="MIT",
    packages=find_packages(),
    package_data={
        "sequentialized_barnard_tests": [
            "data/lai_calibration_data.npy",
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
