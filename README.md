# MATLAB Implementation of the AR3 Tensor Optimization Method

The AR3 method is a regularized third-order tensor method for unconstrained and smooth continuous optimization problems.

A detailed description of the algorithm and extensive numerical comparisons between different algorithmic variants and with classical second-order methods can be found in the paper

> C. Cartis, R. A. Hauser, Y. Liu, K. Welzel, W. Zhu, Efficient Implementation of Third-order Tensor Methods with Adaptive Regularization for Unconstrained Optimization, 2025, https://arxiv.org/abs/2501.00404.

The results of all experiments conducted to generate the plots in the paper can be found at https://wandb.ai/ar3-project/all_experiments.

The code is written for MATLAB on Linux and includes a MATLAB wrapper, written by Nick Gould, for code that evaluates third derivatives of the test problems in the More, Garbow, Hillstrom (MGH) test set, written by Birgin et al. (https://github.com/johngardenghi/mgh).
It is recommended to install the Optimization Toolbox and the Deep Learning Toolbox for MATLAB for full functionality.


## Folder structure

* `experiments`: contains scripts to test behaviour and performance of AR2 and AR3
* `libraries`: contains the necessary libraries
  * `libraries/ar3` contains the implementation of AR2 and AR3
  * `libraries/mgh` contains the wrapper for the Fortran code for the MGH test functions
  * `libraries/test_function` contains various additional test functions

## Repository setup

Follow these steps to run this code on your own.

### Clone

Clone this git repository onto your local computer using `git clone`.
All directory paths will be relative to this base folder from now on.

### Compile the MGH MATLAB wrapper (optional)

The `libraries/mgh` folder contains `mgh.mexa64` which makes it possible to call the Fortran routines implementing evaluations of the MGH function and derivatives from within MATLAB.
To compile this wrapper yourself, use `git submodule update --init` to initialize the submodule that points to https://github.com/johngardenghi/mgh and call `./compile.sh` from within the `libraries/mgh` folder.

### Set up a Python virtual environment

This project makes use of [Weights and Biases](https://wandb.ai) to store the results of individual runs.
To use its Python library we need to install it in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install wandb
```

### Open MATLAB

Start MATLAB inside the `experiments/ExperimentProject` folder and run `setup` before any other commands.
This will add all relevant files to the MATLAB path and makes sure that the Python executable from our virtual environment is used.

The `run_experiment.m` script can be used to run the optimization algorithm and various information about each iteration to an Excel spreadsheet.
It also provides reasonable defaults and indicates the different options that exist for the algorithm parameters.
Try running it to see whether everything is set up correctly.

At the heart of this script lies the `training` function.
It performs a single optimization run and takes a struct of parameters as input.
The output is a vector of structs that represents the history of the optimization run.
The kth entry records the values of multiple quantities of interest (e.g. function value, gradient norm, regularization parameter) at the kth iteration.

A second option exists to run a whole sweep of experiments using MATLAB's built-in Experiment Manager.
Open the `ExperimentProject.prj` file in the MATLAB IDE, then from the "Apps" start "Experiment Manager".
It will prompt "Do you want to open the current project in Experiment Manager?".
Select "Yes".
Open the supplied "TestExperiment", change it as needed and run using the "Run" button at the top.
The Experiment Manager will call the training function with every combination of parameter choices specified in the experiment, provides progress bars and is robust against individual optimization runs failing.

To sync the results of the Experiment Manager runs to the Weights and Biases servers, use the `upload_to_wandb` script on the corresponding folder.
For example:

```Matlab
upload_to_wandb Results/TestExperiment_Result4_20240612T174612/
```


## Code style

To ensure a consistent style of the MATLAB code we use the MISS_HIT style checker (https://misshit.org/).
There is a pre-commit hook to automatically apply the changes suggested by MISS_HIT and enforce its style guidelines.
It needs a working Python installation and can then be activated by running

```bash
pip install pre-commit
pre-commit install
```

inside the project directory.
Afterwards, run `pre-commit run --all` to check the code.
