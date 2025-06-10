# sequentialized_barnard_tests
A collection of sequential statistical hypothesis testing methods for two-by-two contingency tables.

## Development Plan
This codebase will be developed into a standalone pip package, with planned release date June 2025.

Current features:
- Fully automated STEP policy synthesis
- Baseline sequential method implementations of [SAVI](https://www.sciencedirect.com/science/article/pii/S0167715223000597?via%3Dihub) and [Lai](https://projecteuclid.org/journals/annals-of-statistics/volume-16/issue-2/Nearly-Optimal-Sequential-Tests-of-Composite-Hypotheses/10.1214/aos/1176350840.full) procedures.
- Validation tools
    - STEP policy visualization
    - Verification of Type-1 Error control
- Unit tests
    - Method functionality
    - Recreate results from our [paper](https://www.arxiv.org/abs/2503.10966).

Features in development:
- Demonstration scripts for each stage of the STEP evaluation pipeline
- Approximately optimal risk budget estimation tool based on evaluator priors on $$(p_0, p_1)$$
- Fundamental limit estimator for guiding evaluation effort
    - Determine if a particular effect size is plausibly discoverable given the evaluation budget
- Baseline implementation of a sequential Barnard procedure which controls Type-1 error

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

# Code Overview and Quick Start Guides

## Overview
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

and the associated evaluation data is stored in:
```bash
$ data/
```

Quick-start scripts are included under:
```bash
$ quick_start/
```

## Quick Start Guide: Making a STEP Policy for Specific \{n_max, alpha\}

### (1) Understanding the Accepted Shape Parameters
In order to synthesize a STEP Policy for specific values of n_max and alpha, one additional set of parametric decisions will be required. The user will need to set the risk budget shape, which is specified by choice of function family (p-norm and zeta-function) and particular shape parameter. The shape parameter is real-valued; used directly for zeta functions and exponentiated for p-norms.

**For p-norms**\
$$\text{Input: } \lambda \in \mathbb{R}$$
$$\text{Accumulated Budget}(n) = (\frac{n}{n_{max}})^{\exp{\lambda}}$$

**For zeta function**\
$$\text{Input: } \lambda \in \mathbb{R}$$
$$\text{Accumulated Budget}(n) = \frac{\alpha}{Z(n_{max})} \cdot \sum_{i=1}^n (\frac{1}{i})^{\lambda}$$

### (1A) Arbitrary Risk Budgets
Generalizing the accepted risk budgets to arbitrary monotonic sequences $`\{0, \epsilon_1 > 0, \epsilon_2 > \epsilon_1, ..., \alpha\}`$ is in the development pipeline, but is *not handled at present in the code*.

### (2) Running STEP Policy Synthesis
Having decided an appropriate form for the risk budget shape, policy synthesis is straightforward to run. From the base directory, the general command would be:

```bash
$ python scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha} -pz {shape_parameter} -up {use_p_norm_shape_family}
```

Often, the user may not have a clear idea as to which shape family they would prefer. In that case, we recommend the easiest option of using a uniform risk budget, which corresponds to a parameter of 0 for both shape families. This can be run with the explicit arguments or with the implicit defaults; thus, the following commands are EQUIVALENT:

```bash
$ python scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha}
```
```bash
$ python scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha} -pz {0} -up {True}
```
```bash
$ python scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha} -pz {0} -up {False}
```

### (2A) Disclaimers
Running the policy synthesis will save a durable policy to the user's local machine. This policy can be reused for all future settings requiring the same \{n_max, alpha\} combination. For n_max < 500, the amount of required memory is < 5Mb. The policy is saved under:
```bash
$ sequentialized_barnard_tests/policies/
```

At present, we have not tested extensively beyond n_max=500. Going beyond this limit may lead to issues, and the likelihood will grow the larger n_max is set to be. The code will also require increasing amounts of RAM as n_max is increased.

## Quick Start Guide: Evaluation on Real Data

We now assume that a STEP policy has been constructed for the target problem. This can either be one of the default policies, or a newly constructed one following the recipe in the preceding section.

### (1) Formatting the Real Data
The data should be formatted into a numpy array of shape $(N, 2)$. The user should create a new project directory and save the data within it:
```bash
$ mkdir data/{new_project_dir}
$ cp path/to/{my_data_file.npy} data/{new_project_dir}/{my_data_file.npy}
```

See the example structure for an illustration of the desired format.

### (2) Running the Evaluation
Then, the user need simply run the evaluation script, which sets the project directory and file and then carries through the policy synthesis arguments:
```bash
$ python evaluation/run_step_on_evaluation_data.py -p "{new_project_dir}" -f "{my_data_file.npy}" -n {n_max} -a {alpha} -pz {shape_parameter} -up {use_p_norm_shape_family}
```

This will print the evaluation result to the terminal, as well as save key information in a timestamped json file.

To illustrate this via an evaluation on the default data. The following commands are EQUIVALENT:
```bash
$ python evaluation/run_step_on_evaluation_data.py -p "example_clean_spill" -f "TRI_CLEAN_SPILL_v4.npy"
```
```bash
$ python evaluation/run_step_on_evaluation_data.py -p "example_clean_spill" -f "TRI_CLEAN_SPILL_v4.npy" -n {200} -a {0.05} -pz {0.0} -up {False}
```
