import wandb
import json
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots
from wandb_tools import set_plot_asthetics

histories = cache_run_histories(["Exp_Benchmark_0"])

update_type = "Interpolation_m"

for p in [2, 3]:
    # Filter runs
    # api = wandb.Api(timeout=10000)
    api = wandb.Api()
    wandb_runs = api.runs(
        path="ar3-project/all_experiments",
        filters={
            "group": "Exp_Benchmark_0",
            "config.update_type": update_type,
            "config.update_sigma0": "TAYLOR",
            "config.p": p,
            "tags": "training",
        }
        | ({"config.update_use_prerejection": True} if p == 3 else {}),
    )

    # Categorize runs
    method_parameters = [
        # "p",
        "inner_stop_rule",
        "inner_stop_theta",
    ]

    def method_sort_key(method: str) -> tuple[float, float, str]:
        method_dict = json.loads(method)
        inner_stop_rule = method_dict["inner_stop_rule"]
        # Ensure that "First_Order" < "ARP_Theory"
        if inner_stop_rule == "First_Order":
            return (1, 0, method)
        else:
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
            "inner_stop_theta",
            "inner_stop_tolerance_g",
            "inner_inner_stop_tolerance_g",
            "inner_update_type",
            "inner_update_decrease_measure",
            "inner_inner_solver",
            "inner_inner_stop_rule",
        ],
        method_sort_key=method_sort_key,
        error_on_duplicate=True,
    )
    print(f"{categorized_runs.sorted_methods=}")
    print(f"{len(categorized_runs.all_problem_setups)=}")

    if p == 2:
        custom_titles = [
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla m_k\| \leq 10^{-9}$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla m_k\| \leq 10^{-2} \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla m_k\| \leq 10^0 \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla m_k\| \leq 10^2 \|\mathbf{s}\|^2$",
            r"\textsf{AR$2$-Interp}" + "\n" + r"$\|\nabla m_k\| \leq 10^4 \|\mathbf{s}\|^2$",
        ]
    else:
        custom_titles = [
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla m_k\| \leq 10^{-9}$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla m_k\| \leq 10^{-2} \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla m_k\| \leq 10^0 \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla m_k\| \leq 10^2 \|\mathbf{s}\|^3$",
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + r"$\|\nabla m_k\| \leq 10^4 \|\mathbf{s}\|^3$",
        ]

    filename_prefix = f"theta/performance_profile {update_type} {p}"

    set_plot_asthetics()

    generate_gpp_plots(
        filename_prefix=filename_prefix,
        categorized_runs=categorized_runs,
        new_labels=custom_titles,
        format="pgf",
    )
