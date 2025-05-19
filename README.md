# sequentialized_barnard_tests
A collection of sequential statistical hypothesis testing methods for two-by-two contingency tables.

## Development Plan
This codebase will be developed into a standalone pip package, with planned release date June 2025.

Current features:
    -- Fully automated STEP policy synthesis
    -- Baseline sequential method implementations of (SAVI)[https://www.sciencedirect.com/science/article/pii/S0167715223000597?via%3Dihub] and (Lai)[https://projecteuclid.org/journals/annals-of-statistics/volume-16/issue-2/Nearly-Optimal-Sequential-Tests-of-Composite-Hypotheses/10.1214/aos/1176350840.full]
    -- Validation tools
        -- STEP policy visualization
        -- Verification of Type-1 Error control
    -- Unit tests
        -- Method functionality
        -- Recreate results from our (paper)[https://www.arxiv.org/abs/2503.10966]

Features in development:
    -- Demonstration scripts for each stage of the STEP evaluation pipeline
    -- Approximately optimal risk budget estimation tool based on evaluator priors on $$(p_0, p_1)$$
    -- Fundamental limit estimator for guiding evaluation effort
        -- Determine if a particular effect size is plausibly discoverable given the evaluation budget
    -- Baseline implementation of a sequential Barnard procedure which controls Type-1 error

## Installation Instructions
An example of the environment setup is shown below.
```bash
$ cd <some_directory>
$ git clone git@github.com:TRI-ML/sequentialized_barnard_tests.git
$ virtualenv --python=python3.10 <env_name>
$ source <env_name>/bin/activate
$ cd sequentialized_barnard_tests
$ pip install -r requirements.txt
$ pip install -e .
$ pre-commit install
```

## Overview and Quick Start Guide

Scripts to generate and visualize STEP policies are included under:
```bash
$ scripts/
```

Any resulting visualizations are stored in:
```bash
$ media/
```

The evaluation environment for real data is included in:
```bash
$ evaluation/
```

and data is stored in:
```bash
$ data/
```

Quick-start scripts are included under:
```bash
$ quick_start/
```
