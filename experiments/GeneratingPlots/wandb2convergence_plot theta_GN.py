import wandb
import json
import os
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import convergence_dot_plot
from wandb_tools import set_plot_asthetics

histories = cache_run_histories(["Exp_Benchmark_6"])

update_type = "Interpolation_m"

for p in [2, 3]:
    # Filter runs
    api = wandb.Api()
    wandb_runs = api.runs(
        path="ar3-project/all_experiments",
        filters={
            "group": "Exp_Benchmark_6",
            "config.update_type": "Interpolation_m",
            "config.update_sigma0": "TAYLOR",
            "config.inner_stop_rule": "General_Norm",
            "config.p": p,
            "tags": "training",
            "$or": [
                {"config.problem": "5"},
                {"config.problem": "13"},
            ],
        }
        # | ({"config.update_use_prerejection": True} if p == 3 else {}),
    )

    # Categorize runs
    method_parameters = [
        # "p",
        "update_use_prerejection",
        "inner_stop_theta",
    ]

    def method_sort_key(method: str) -> tuple[float, float, str]:
        method_dict = json.loads(method)
        # inner_stop_rule = method_dict["inner_stop_rule"]
        # Ensure that "First_Order" < "ARP_Theory"
        # if inner_stop_rule == "First_Order":
        #     return (1, 0, method)
        # else:
        #     # Ensure thetas are in order
        theta = method_dict.get("inner_stop_theta", 100)
        return (2, theta, method)

    categorized_runs = categorize_runs(
        wandb_runs=wandb_runs,
        histories=histories,
        method_parameters=method_parameters,
        ignore=[
            "wandb_project",
            "wandb_group",
            "inner_solver",
            "update_type",
            # "inner_stop_rule",
            "inner_stop_tolerance_g",
            "inner_inner_stop_tolerance_g",
            "inner_update_type",
            "inner_update_decrease_measure",
            "inner_inner_solver",
            "inner_inner_stop_rule",
        ],
        # method_sort_key=method_sort_key,
        error_on_duplicate=True,
    )
    print(f"{categorized_runs.sorted_methods=}")
    print(f"{categorized_runs.all_problem_setups=}")

    set_plot_asthetics()

    # custom titles
    if p == 2:
        legend_ncols = 2
        custom_titles = [
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^0 \sigma_k \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^2 \sigma_k \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^4 \sigma_k \|\mathbf{s}\|^2$",
        ]

        grid = convergence_dot_plot(
            categorized_runs,
            row_order=["5", "13"],
            col_order=categorized_runs.sorted_methods,
            col_titles=custom_titles,
            legend_ncols=legend_ncols,
            fig_scale = 1,
        )
    else:
        legend_ncols = None  # default
        custom_titles = [
            r"\textsf{AR$3$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^0 \sigma_k \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla t_k\| \leq 10^0 \sigma_k \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^2 \sigma_k \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla t_k\| \leq 10^2 \sigma_k \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^4 \sigma_k \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla t_k\| \leq 10^4 \sigma_k \|\mathbf{s}\|^3$",
        ]

        grid = convergence_dot_plot(
            categorized_runs,
            row_order=["5", "13"],
            col_order=categorized_runs.sorted_methods,
            col_titles=custom_titles,
            legend_ncols=legend_ncols,
            fig_scale = 0.85,
        )

    os.makedirs("theta_GN", exist_ok=True)
    filename = f"theta_GN/convergence {update_type} {p}.pgf"
    grid.figure.savefig(filename, dpi=100)
    print(f"Saved {filename!r}")
