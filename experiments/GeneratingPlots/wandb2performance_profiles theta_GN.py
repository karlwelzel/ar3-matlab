import wandb
import json
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots
from wandb_tools import set_plot_asthetics

histories = cache_run_histories(["Exp_Benchmark_6"])

update_type = "Interpolation_m"

for p in [2, 3]:
    # Filter runs
    # api = wandb.Api(timeout=10000)
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
        }
    )

    # Categorize runs
    method_parameters = [
        "update_use_prerejection",
        "inner_stop_theta",
    ]

    def method_sort_key(method: str) -> tuple[float, float, str]:
        method_dict = json.loads(method)
        # inner_stop_rule = method_dict["inner_stop_rule"]
        # # Ensure that "First_Order" < "ARP_Theory"
        # if inner_stop_rule == "First_Order":
        #     return (1, 0, method)
        # else:
            # Ensure thetas are in order
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
    print(f"{len(categorized_runs.all_problem_setups)=}")

    if p == 2:
        legend_ncols = 2
        custom_titles = [
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^0 \sigma_k \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^2 \sigma_k \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla t_k\| \leq 10^4 \sigma_k \|\mathbf{s}\|^2$",
        ]
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

    filename_prefix = f"theta_GN/performance_profile {p} {update_type}"

    set_plot_asthetics()

    generate_gpp_plots(
        filename_prefix=filename_prefix,
        categorized_runs=categorized_runs,
        new_labels=custom_titles,
        format="pgf",
    )
