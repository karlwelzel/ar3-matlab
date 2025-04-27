import wandb
import json
import os
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import convergence_dot_plot
from wandb_tools import set_plot_asthetics

for inner_stop_rule in ["First_Order", "ARP_Theory"]:
    histories = cache_run_histories(["Exp_Benchmark_0"])

    # Filter runs
    api = wandb.Api()

    wandb_runs = api.runs(
        path="ar3-project/all_experiments",
        filters={
            "group": "Exp_Benchmark_0",
            "config.update_sigma0": "TAYLOR",
            "config.inner_solver": "AR2",
            "config.p": 3,
            "config.inner_stop_rule": inner_stop_rule,
            "config.inner_stop_tolerance_g": 1e-9,
            "tags": "training",
            "$and": [
                {
                    "$or": [
                        {"config.update_type": "Interpolation_m"},
                        {"config.update_type": "Simple"},
                        # {"config.update_type": "BGMS"},
                    ],
                },
                {
                    "$or": [
                        {"config.problem": "5"},
                        {"config.problem": "13"},
                    ],
                },
            ],
        },
    )

    # Categorize runs
    method_parameters = [
        "update_use_prerejection",
        "update_type",
    ]

    def method_sort_key(method: str) -> tuple[float, str]:
        method_dict = json.loads(method)
        update_type = method_dict["update_type"]
        # Ensure that "Simple" comes before "Interpolation_m"
        if update_type == "Simple":
            return (1, method)
        else:
            return (2, method)

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
        ],
        method_sort_key=method_sort_key,
        error_on_duplicate=True,
    )
    print(f"{categorized_runs.sorted_methods=}")
    print(f"{categorized_runs.all_problem_setups=}")

    # custom titles
    custom_titles = [
        r"\textsf{AR$3$-Simple}",
        r"\textsf{AR$3$-Simple\textsuperscript{+}}",
        r"\textsf{AR$3$-Interp}",
        r"\textsf{AR$3$-Interp\textsuperscript{+}}",
    ]

    set_plot_asthetics()

    grid = convergence_dot_plot(
        categorized_runs,
        row_order=["5", "13"],
        col_order=categorized_runs.sorted_methods,
        col_titles=custom_titles,
    )

    os.makedirs("prerejection", exist_ok=True)
    filename = f"prerejection/convergence {inner_stop_rule}.pgf"
    grid.figure.savefig(filename, dpi=100)
    print(f"Saved {filename!r}")
