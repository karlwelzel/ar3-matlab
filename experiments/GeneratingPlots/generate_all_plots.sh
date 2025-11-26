#!/usr/bin/bash

# Use the correct Python interpreter with necessary packages installed.
# For Windows (e.g., Git Bash):
#   PYTHON="C:/ProgramData/anaconda3/python.exe"

# For Linux, activate the local virtual environment (optional if PYTHON is set directly)
source ../../.venv/bin/activate
PYTHON="../../.venv/bin/python"

echo "Using Python: $PYTHON"

# Generate performance profile plots for different experimental settings
"$PYTHON" "wandb2performance_profiles sigma0.py"
"$PYTHON" "wandb2performance_profiles interpolation.py"
"$PYTHON" "wandb2performance_profiles prerejection.py"
"$PYTHON" "wandb2performance_profiles updates.py"
"$PYTHON" "wandb2performance_profiles theta.py"
"$PYTHON" "wandb2performance_profiles theta_GN.py"
"$PYTHON" "wandb2performance_profiles theta_GN_benchmark.py"
"$PYTHON" "wandb2performance_profiles 2vs3.py"
"$PYTHON" "wandb2performance_profiles benchmark.py"
"$PYTHON" "wandb2performance_profiles GLRT.py"

# Generate convergence plots for the same set of experiments
"$PYTHON" "wandb2convergence_plot sigma0.py"
"$PYTHON" "wandb2convergence_plot interpolation.py"
"$PYTHON" "wandb2convergence_plot prerejection.py"
"$PYTHON" "wandb2convergence_plot updates.py"
"$PYTHON" "wandb2convergence_plot theta.py"
"$PYTHON" "wandb2convergence_plot theta_GN.py"
"$PYTHON" "wandb2convergence_plot benchmark.py"

# Generate special plots
"$PYTHON" "local_minimizer_discontinuity.py"
"$PYTHON" "illustration_ARp.py"
