#!/usr/bin/bash

# Use the correct Python interpreter with necessary packages installed.
# For Windows (e.g., Git Bash):
#   PYTHON="C:/ProgramData/anaconda3/python.exe"

# For Linux, activate the local virtual environment (optional if PYTHON is set directly)
source ../../.venv/bin/activate
PYTHON="../../.venv/bin/python"

echo "Using Python: $PYTHON"

# Create output folders if they don't already exist
for folder in 2vs3 benchmark interpolation prerejection sigma0 theta theta_GN updates; do
    if [ ! -d "$folder" ]; then
        mkdir "$folder"
        echo "Created folder: $folder"
    fi
done

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

# Generate convergence plots for the same set of experiments
"$PYTHON" "wandb2convergence_plot sigma0.py"
"$PYTHON" "wandb2convergence_plot interpolation.py"
"$PYTHON" "wandb2convergence_plot prerejection.py"
"$PYTHON" "wandb2convergence_plot updates.py"
"$PYTHON" "wandb2convergence_plot theta.py"
"$PYTHON" "wandb2convergence_plot theta_GN.py"
"$PYTHON" "wandb2convergence_plot benchmark.py"
