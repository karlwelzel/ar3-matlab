import wandb
import json
import os
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import convergence_dot_plot
from wandb_tools import set_plot_asthetics


inner_stop_rule = "First_Order"

histories = cache_run_histories(["Exp_Benchmark_0"])

# Filter runs
api = wandb.Api()

wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={
        "group": "Exp_Benchmark_0",
        "config.update_type": "Simple",
        "config.inner_solver": "AR2",
        "config.p": 3,
        "config.update_use_prerejection": False,
        "config.inner_stop_rule": inner_stop_rule,
        "tags": "training",
        "$or": [
            {"config.problem": "5"},
            {"config.problem": "13"},
        ],
    },
)

# Categorize runs
method_parameters = [
    "update_sigma0",
]


def method_sort_key(method: str) -> tuple[float, str]:
    method_dict = json.loads(method)
    try:
        sigma0 = float(method_dict["update_sigma0"])
    except ValueError:
        sigma0 = float("inf")
    return (sigma0, method)


categorized_runs = categorize_runs(
    wandb_runs=wandb_runs,
    histories=histories,
    method_parameters=method_parameters,
    ignore=[
        "wandb_project",
        "wandb_group",
        "inner_solver",
        "inner_stop_theta",
        "inner_stop_tolerance_g",
        "inner_update_type",
        "inner_update_decrease_measure",
        "inner_inner_solver",
        "inner_inner_stop_rule",
        "stop_max_iterations",
        "inner_stop_max_iterations",
        "update_decrease_measure",
        "p",
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)

categorized_runs.sort(key=lambda run: (*method_sort_key(run.method), int(run.problem)))

print(f"{categorized_runs.sorted_methods=}")
print(f"{categorized_runs.all_problem_setups=}")

# custom titles
custom_titles = [
    r"$\sigma_0=10^{-8}$",
    r"$\sigma_0=1$",
    r"$\sigma_0=10^8$",
    r"$\sigma_0$: \texttt{Taylor}",
]

set_plot_asthetics()

grid = convergence_dot_plot(
    categorized_runs,
    row_order=["5", "13"],
    col_order=categorized_runs.sorted_methods,
    col_titles=custom_titles,
)

os.makedirs("sigma0", exist_ok=True)
filename = f"sigma0/convergence {inner_stop_rule}.pgf"
grid.figure.savefig(filename)
print(f"Saved {filename!r}")
