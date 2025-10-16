# License statement

The code is provided under a Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Under the license, the code is provided royalty free for non-commercial purposes only. The code may be covered by patents and if you want to use the code for commercial purposes, please contact us for a different license.

# sequentialized_barnard_tests
A collection of sequential statistical hypothesis testing methods for two-by-two contingency tables.

[[Paper](https://www.arxiv.org/abs/2503.10966)]&nbsp;&nbsp;[[Website](https://tri-ml.github.io/step/)]
<!-- ## Development Plan
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
- Baseline implementation of a sequential Barnard procedure which controls Type-1 error -->

## Installation Instructions \[Standard\]
The basic environmental setup is shown below. A virtual / conda environment may be constructed; however, the requirements are quite lightweight and this is probably not needed.
```bash
$ pip install sequentialized_barnard_tests
```

## Installation Instructions \[Dev\]
For potential contributors and developers, we recommend a virtualenv:
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

We assume that any specified virtual / conda environment has been activated for all subsequent code snippets.

# Quick Start Guides
## Convenience: Automatic Test Instantiation

For convenience, you can automatically select between STEP and Lai (a baseline method) depending on the value of `n_max` using the factory function in `auto.py`:

```python
from sequentialized_barnard_tests import get_mirrored_test
test = get_mirrored_test(n_max, alternative, alpha, verbose=True, ...)
```
If `n_max > 500`, this will instantiate a `MirroredLaiTest`, which is a computationally efficient baseline with comparable performance to `MirroredStepTest` for a large-enough sample size; otherwise, it will use the more powerful `MirroredStepTest` which can take longer to synthesize the decision rule. All shared and class-specific arguments can be passed as keyword arguments.

## Example Usage

Below is a minimum example code with different policy evaluation data, leading to three distinct evaluation results.

### Case 1: Test yields `AcceptAlternative`
```python
from sequentialized_barnard_tests import get_mirrored_test, Hypothesis

n_max = 100  # maximum sample size is 100 (per policy)
alternative = Hypothesis.P0LessThanP1  # we want to test if "success rate of the first policy < success rate of the second policy"
alpha = 0.05  # false positive rate is 5%

test = get_mirrored_test(n_max=n_max, alternative=alternative, alpha=alpha)

success_array_policy_0 = [False] * 10  # the first policy failed 10 times
success_array_policy_1 = [True] * 10  # the second policy succeeded 10 times 

result = test.run_on_sequence(success_array_policy_0, success_array_policy_1)
decision = result.decision
print(decision)  # AcceptAlternative: success rate of the first policy < success rate of the second policy with 95% confidence
```

### Case 2: Test yields `AcceptNull`
```python
from sequentialized_barnard_tests import get_mirrored_test, Hypothesis

n_max = 100  # maximum sample size is 100 (per policy)
alternative = Hypothesis.P0LessThanP1  # we want to test if "success rate of the first policy < success rate of the second policy"
alpha = 0.05  # false positive rate is 5%

test = get_mirrored_test(n_max=n_max, alternative=alternative, alpha=alpha)

success_array_policy_0 = [True] * 10  # the first policy succeeded 10 times
success_array_policy_1 = [False] * 10  # the second policy failed 10 times 

result = test.run_on_sequence(success_array_policy_0, success_array_policy_1)
decision = result.decision
print(decision)  # AcceptNull: success rate of the first policy > success rate of the second policy with 95% confidence
```

Note: `AcceptNull` is a valid decision only for "mirrored" tests.
In our terminology, a mirrored test is one that runs two one-sided tests
simultaneously, with the null and the alternaive flipped from each other.
(Because of the monotonicity of the test statistic, mirrored tests suffer no penalty for
running two tests simultaneously, and therefore essentially dominate one-sided tests.)
In the example above, the alternative is `Hypothesis.P0LessThanP1` and the decision is
`Decision.AcceptNull`, which should be interpreted as accepting `Hypothesis.P0MoreThanP1`.
If you rather want a more conventional one-sided test, you can instantiate one by calling
`get_test` instead of `get_mirrored_test`.

### Case 3: Test yields `FailToDecide`
```python
from sequentialized_barnard_tests import get_mirrored_test, Hypothesis

n_max = 100  # maximum sample size is 100 (per policy)
alternative = Hypothesis.P0LessThanP1  # we want to test if "success rate of the first policy < success rate of the second policy"
alpha = 0.05  # false positive rate is 5%

test = get_mirrored_test(n_max=n_max, alternative=alternative, alpha=alpha)

success_array_policy_0 = [True, False, False, True]  # the first policy succeeded 2 out of 4 times
success_array_policy_1 = [False, True, True, True]  # the second policy succeeded 3 out of 4 times

result = test.run_on_sequence(success_array_policy_0, success_array_policy_1)
decision = result.decision
print(decision)  # FailToDecide: difference was not statistically separable; user can collect 100 - 4 = 96 more rollouts for each policy to re-run the test.
```

## Key Notes for Understanding the Core Ideas of STEP Code

We include key notes for understanding the core ideas of the STEP code. Quick-start resources are included in both shell script and notebook form.

### (1A) Understanding the Accepted Shape Parameters
In order to synthesize a STEP Policy for specific values of n_max and alpha, one additional set of parametric decisions will be required. The user will need to set the risk budget shape, which is specified by choice of function family (p-norm vs zeta-function) and particular shape parameter. The shape parameter is real-valued; it is used directly for zeta functions and is exponentiated for p-norms.

**For p-norms**
- $$\text{Shape Parameter: } \lambda \in \mathbb{R}$$
- $$\text{Accumulated Risk Budget}(n) = \alpha \cdot (\frac{n}{n_{max}})^{\exp{(\lambda)}}$$

**For zeta function**
- $$\text{Shape Parameter: } \lambda \in \mathbb{R}$$
- $$\text{Accumulated Risk Budget}(n) = \frac{\alpha}{Z(n_{max})} \cdot \sum_{i=1}^n (\frac{1}{i})^{\lambda}$$
- $$Z(n_{max}) = \sum_{i=1}^{n_{max}} (\frac{1}{i})^{\lambda}$$

The user may confirm that in each case, evaluating the accumulated risk budget at $`n=n_{max}`$ returns precisely $\alpha$.


### (1B) Shape Parameters in Code
In the codebase, the value of $\lambda$ is set in the \{shape_parameter\} variable. This variable is real-valued.\
\
The family shape is set by the \{use_p_norm\} variable. This variable is Boolean.
- If it is True, then p-norm family is used.
- If it is False, the zeta-function family is used.


### (1C) Arbitrary Risk Budgets
Generalizing the accepted risk budgets to arbitrary monotonic sequences $`\{0, \epsilon_1 > 0, \epsilon_2 > \epsilon_1, ..., \epsilon_{n_{max}} = \alpha\}`$ is in the development pipeline, but is **not handled at present in the code**.


### (2A) Running STEP Policy Synthesis
Having decided an appropriate form for the risk budget shape, policy synthesis is straightforward to run. From the base directory, the general command would be:

```bash
$ python sequentialized_barnard_tests/scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha} -pz {shape_parameter} -up {use_p_norm}
```

Note: This script will be called automatically upon instantiation of a test object, if the corresponding polciy file is missing from `sequentialized_barnard_tests/policies/`.

### (2B) What If I Don't Know the Right Risk Budget?
We recommend using the default linear risk budget, which is the shape *used in the paper*. This corresponds to \{shape_parameter\}$`= 0.0`$ for each shape family. Thus, *each of the following commands constructs the same policy*:

```bash
$ python sequentialized_barnard_tests/scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha}
```
```bash
$ python sequentialized_barnard_tests/scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha} -pz {0.0} -up "True"
```
```bash
$ python sequentialized_barnard_tests/scripts/synthesize_general_step_policy.py -n {n_max} -a {alpha} -pz {0.0} -up "False"
```

Note: For \{shape_parameter\} $`\neq 0`$, the shape families differ. Therefore, the choice of \{use_p_norm\} *will affect the STEP policy*.

### (2C) Disclaimers
- Running the policy synthesis will save a durable policy to the user's local machine. This policy can be reused for all future settings requiring the same \{n_max, alpha\} combination. For \{n_max\} $`< 500`$, the amount of required memory is < 5Mb. The policy is saved under:
```bash
$ sequentialized_barnard_tests/policies/
```

- At present, we have not tested extensively beyond \{n_max\}$`=500`$. Going beyond this limit may lead to issues, and the likelihood will grow the larger \{n_max\} is set to be. The code will also require increasing amounts of RAM as \{n_max\} is increased.

## Script-Based Evaluation on Real Data

We now assume that a STEP policy has been constructed for the target problem. This can either be one of the default policies, or a newly constructed one following the recipe in the preceding section.

### (1) Formatting the Real Data
The data should be formatted into a numpy array of shape $(N, 2)$. The user should create a new project directory and save the data within it:
```bash
$ mkdir data/{new_project_dir}
$ cp path/to/{my_data_file.npy} data/{new_project_dir}/{my_data_file.npy}
```

We give an example that *would have generated* the included data:
```bash
$ mkdir data/example_clean_spill
$ cp some/path/to/TRI_CLEAN_SPILL_v4.npy data/example_clean_spill/TRI_CLEAN_SPILL_v4.npy
```

### (2) Running the Evaluation
Then, the user need simply run the evaluation script, which requires the project directory and file in addition to the policy synthesis arguments:
```bash
$ python evaluation/run_step_on_evaluation_data.py -p "{new_project_dir}" -f "{my_data_file.npy}" -n {n_max} -a {alpha} -pz {shape_parameter} -up "{use_p_norm}"
```

This will print the evaluation result to the terminal, as well as save key information in a timestamped json file.

We illustrate this via an evaluation on the default data:
```bash
$ python evaluation/run_step_on_evaluation_data.py -p "example_clean_spill" -f "TRI_CLEAN_SPILL_v4.npy" -n {200} -a {0.05} -pz {0.0} -up "False"
```

# Code Overview
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

# Citation

```
@inproceedings{snyder2025step,
title = {Is Your Imitation Learning Policy Better Than Mine? Policy Comparison with Near-Optimal Stopping},
author = {Snyder, David and Hancock, Asher James and Badithela, Apurva and Dixon, Emma and Miller, Patrick and Ambrus, Rares Andrei and Majumdar, Anirudha and Itkina, Masha and Nishimura, Haruki},
booktitle={Proceedings of the Robotics: Science and Systems Conference (RSS)},
year = {2025},
}
```
