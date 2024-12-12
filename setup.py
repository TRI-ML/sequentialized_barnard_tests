from setuptools import find_packages, setup

setup(
    name="sequentialized_barnard_tests",
    version="0.0.1",
    description="Sequential statistical hypothesis testing for two-by-two contingency tables.",
    authors=["David Snyder", "Haruki Nishimura"],
    author_emails=["dasnyder@princeton.edu", "haruki.nishimura@tri.global"],
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy>=1.20",
        "scipy",
        "tqdm",
    ],
)
