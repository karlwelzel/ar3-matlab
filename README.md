# MATLAB Implementation of the AR3 Tensor Optimization Method

The AR3 method is a regularized third-order tensor method for unconstrained and smooth continuous optimization problems.

A detailed description of the algorithm and extensive numerical comparisons between different algorithmic variants and with classical second-order methods can be found in the paper

> C. Cartis, R. A. Hauser, Y. Liu, K. Welzel, W. Zhu, Efficient Implementation of Third-order Tensor Methods with Adaptive Regularization for Unconstrained Optimization, 2025, https://arxiv.org/abs/2501.00404.

The results of all experiments conducted to generate the plots in the paper can be found at https://wandb.ai/ar3-project/all_experiments.

The code is written for MATLAB on Linux and includes a MATLAB wrapper, written by Nick Gould, for evaluating third derivatives of the test problems in the More, Garbow, Hillstrom (MGH) test set, written by Birgin et al. (https://github.com/johngardenghi/mgh).
It is recommended to install the Optimization Toolbox and the Deep Learning Toolbox for MATLAB for full functionality.


## Folder structure

* `experiments`: contains scripts to execute AR2 and AR3 algorithms
  * `experiments/ExperimentProject` contains scripts to construct and test AR2 and AR3 algorithms
  * `experiments/GeneratingPlots` contains Matlab Experiment Manager settings, all recorded runs and python codes to regenerate all plots
* `libraries`: contains the necessary libraries
  * `libraries/ar3` contains the implementation of AR2 and AR3
  * `libraries/mgh` contains the wrapper for the Fortran code for the MGH test functions
  * `libraries/test_function` contains various additional test functions

## Repository setup

Follow these steps to run this code on your own machine.

### Clone

Clone this git repository onto your local computer using `git clone`.
All directory paths will be relative to this base folder from now on.

### Compile the MGH MATLAB wrapper (optional)

The `libraries/mgh` folder contains `mgh.mexa64` which makes it possible to call the Fortran routines implementing evaluations of the MGH function and derivatives from within MATLAB.
To compile this wrapper yourself, use `git submodule update --init` to initialize the submodule that points to https://github.com/johngardenghi/mgh and run
```bash
gfortran -O3 -c -o set_precision.o fortran/set_precision.f90
gfortran -O3 -c -o mgh.o fortran/mgh.90 -lm
mex -largeArrayDims set_precision.o mgh.o mgh.F90
```
from within the `libraries/mgh` folder.

Alternatively, if you have downloaded the code as a ZIP file, copy the contents of the https://github.com/johngardenghi/mgh repository in the `libraries/mgh/fortran` folder before following the instructions above.

### Set up a Python virtual environment

This project makes use of [Weights and Biases](https://wandb.ai) to store the results of individual runs.
To use its Python library we need to install it in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install wandb==0.17.9
```

For compatibility between MATLAB and Python versions check https://www.mathworks.com/support/requirements/python-compatibility.html.
You might need to use a different Python version that one pre-installed with your OS.

### Open MATLAB

Start MATLAB inside the `ExperimentProject` folder and run `setup` before issuing any other commands.
This will add all relevant files to the MATLAB path and ensure the Python executable from the virtual environment is used.

Use the script `ExperimentProject/run_experiment.m` to run the optimization algorithm and log various information about each iteration to an Excel spreadsheet.

The script also provides reasonable defaults and indicates available options for algorithm parameters.
A flowchart illustrating the overall structure of the implementation can be found in `ExperimentProject/AR3_Matlab_Flowchart.pdf`.

At the heart of this script lies the `ExperimentProject/training` function.
It performs a single optimization run and accepts a struct of parameters as input.
The output is a vector of structs that records the history of the optimization run.
The kth entry records the values of multiple quantities of interest (e.g. function value, gradient norm, regularization parameter) at the kth iteration.

A second option exists to run a whole sweep of experiments using MATLAB's built-in Experiment Manager.
Open the `ExperimentProject/ExperimentProject.prj` file in the MATLAB IDE, then launch "Experiment Manager" from the "Apps" tab.
When prompted "Do you want to open the current project in Experiment Manager?", select "Yes".
Open the supplied "TestExperiment", change it as needed and click "Run".
The Experiment Manager will call the training function for every combination of parameter values specified in the experiment.
It provides progress tracking and is robust against individual optimization runs failing.

To sync the results of the Experiment Manager runs to the Weights and Biases servers, use the `upload_to_wandb` script on the corresponding folder.
For example:

```Matlab
upload_to_wandb Results/TestExperiment_Result4_20240612T174612/
```

### Rerun all experiments

To rerun all experiments that went into the graphs in the paper, first copy over each ".mat" file under `GeneratingPlots` to `ExperimentProject`, add them to the project, open the "Experiment Manager" as explained above and run each experiment.
This will generate 3,215 runs in total, each with a different combination of parameters.

Using the `upload_to_wandb` script as explained above you can upload the results to the weights and biases servers using the project and group name specified in the experiment settings.
Afterwards, you can view and access the raw data using their cloud platform.


### Regenerate all plots

To regenerate all convergence dot plots and performance profile plots from the paper, you can use the raw data for our runs, which is available publicly at https://wandb.ai/ar3-project/all_experiments.

After activating your local Python installation, run
```bash
pip install wandb==0.17.9 matplotlib pandas seaborn
wandb login
```
and then start the relevant plotting script, for example `python "wandb2convergence_plot benchmark.py"`.
To speed up the plotting, the raw data is cached in the `.bin` files.
The WandB integration is still necessary to apply the proper filters and download metadata for each run.

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
