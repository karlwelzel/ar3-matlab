import wandb
import json
import os
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots
from wandb_tools import set_plot_asthetics

groups = ["Exp_Benchmark_8"]
histories = cache_run_histories(groups)

# api = wandb.Api(timeout=10000)
api = wandb.Api()
wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={
        "group": "Exp_Benchmark_8",
    },
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

print(f"{categorized_runs.sorted_methods=}")
print(f"{len(categorized_runs.all_problem_setups)=}")

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
        acc_label = r"$\norm{\nabla \tilde{m}_k} \leq 10^{-10}$"
    elif stop_rule == "ARP_Theory":
        acc_label = r"$\norm{\nabla \tilde{m}_k} \leq 10^2 \norm{\tilde{\vek{s}}}^2$"
    else:
        acc_label = stop_rule

    # Example: \textsf{MCM, Exact}, \textsf{GLRT, Inexact}
    label = rf"\textsf{{AR3-Interp\textsuperscript{{+}} + AR2-Simple + {solver_label}, {acc_label}}}"
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
