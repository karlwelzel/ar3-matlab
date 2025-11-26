import json
import os

import matplotlib.pyplot as plt
import wandb

from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots


def set_plot_asthetics():
    """
    Simple PGF + LaTeX setup that does NOT import ../settings/math.tex.
    This avoids the pdflatex error you are seeing.
    """
    plt.switch_backend("pgf")
    plt.rc(
        "pgf",
        texsystem="pdflatex",
        rcfonts=False,
        preamble=r"""
            \usepackage{amsmath,amssymb}
            \usepackage{bm}
        """,
    )
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Computer Modern Roman")
    plt.rc("lines", markersize=2.0, linewidth=1.0)
    plt.rc("savefig", dpi=300)


groups = ["Exp_Benchmark_8"]
histories = cache_run_histories(groups)

print("Cached groups:", list(histories.keys()))
if "Exp_Benchmark_8" in histories:
    print(
        "Number of problems in Exp_Benchmark_8 cache:",
        len(histories["Exp_Benchmark_8"]),
    )
else:
    print("WARNING: 'Exp_Benchmark_8' not found in histories dict.")

# api = wandb.Api(timeout=10000)
api = wandb.Api()
wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={
        "group": "Exp_Benchmark_8",
    },
)

print("Number of runs fetched from W&B:", len(wandb_runs))
if len(wandb_runs) == 0:
    raise RuntimeError(
        "No runs found for group 'Exp_Benchmark_8'. "
        "Check the project path and group name in W&B."
    )

inner_inner_solvers = {run.config.get("inner_inner_solver", None) for run in wandb_runs}
inner_inner_stop_rules = {
    run.config.get("inner_inner_stop_rule", None) for run in wandb_runs
}
print("Distinct inner_inner_solver values:", inner_inner_solvers)
print("Distinct inner_inner_stop_rule values:", inner_inner_stop_rules)

method_parameters = [
    "inner_inner_solver",
    "inner_inner_stop_rule",
]


def method_sort_key(method: str) -> tuple[int, str]:
    """
    Total ordering on (inner_inner_solver, inner_inner_stop_rule):

        1. MCMR + ARP_Theory
        2. MCMR + First_Order
        3. GLRT + ARP_Theory
        4. GLRT + First_Order
        5+. anything else (at the end, just in case)
    """
    method_dict = json.loads(method)
    solver = method_dict.get("inner_inner_solver", "")
    stop_rule = method_dict.get("inner_inner_stop_rule", "")

    if solver == "MCMR" and stop_rule == "ARP_Theory":
        idx = 1
    elif solver == "MCMR" and stop_rule == "First_Order":
        idx = 2
    elif solver == "GLRT" and stop_rule == "ARP_Theory":
        idx = 3
    elif solver == "GLRT" and stop_rule == "First_Order":
        idx = 4
    else:
        idx = 5

    return (idx, method)


categorized_runs = categorize_runs(
    wandb_runs=wandb_runs,
    dump_type="gpp",
    histories=histories,
    method_parameters=method_parameters,
    ignore=[
        "wandb_project",
        "wandb_group",
        "update_sigma0",
        "update_type",
        "update_use_prerejection",
        "update_decrease_measure",
        "stop_rule",
        "stop_tolerance_g",
        "stop_max_iterations",
        "inner_solver",
        "inner_stop_rule",
        "inner_stop_theta",
        "inner_stop_tolerance_g",
        "inner_stop_max_iterations",
        "inner_update_type",
        "inner_update_decrease_measure",
        "inner_inner_stop_tolerance_g",
        "p",
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)

print("sorted_methods:", categorized_runs.sorted_methods)
print("Number of problem setups:", len(categorized_runs.all_problem_setups))

if not categorized_runs.sorted_methods:
    raise RuntimeError(
        "categorized_runs.sorted_methods is empty. "
        "Likely either histories are missing/empty for Exp_Benchmark_8, "
        "or none of the runs have the required method parameters set."
    )

new_labels: list[str] = []
for method in categorized_runs.sorted_methods:
    method_dict = json.loads(method)
    solver = method_dict.get("inner_inner_solver", "UNKNOWN")
    stop_rule = method_dict.get("inner_inner_stop_rule", "UNKNOWN")

    # Renaming labels "Exact"/"Inexact"
    if solver == "MCMR":
        solver_label = "MCM"
    else:
        solver_label = solver
    if stop_rule == "First_Order":
        acc_label = "Exact"
    elif stop_rule == "ARP_Theory":
        acc_label = "Inexact"
    else:
        acc_label = stop_rule

    # Example: \textsf{MCM, Exact}, \textsf{GLRT, Inexact}
    label = rf"\textsf{{{solver_label}, {acc_label}}}"
    new_labels.append(label)

print("Legend labels (in order):", new_labels)

# Generate performance profile plots (PGF + PNG)
os.makedirs("GLRT", exist_ok=True)
filename_prefix = "GLRT/performance_profile GLRT"

set_plot_asthetics()

# PGF for LaTeX
generate_gpp_plots(
    filename_prefix=filename_prefix,
    categorized_runs=categorized_runs,
    new_labels=new_labels,
    format="pgf",
    legend_ncols=2,
)

# PNG for quick visual inspection
generate_gpp_plots(
    filename_prefix=filename_prefix,
    categorized_runs=categorized_runs,
    new_labels=new_labels,
    format="png",
    legend_ncols=2,
)
