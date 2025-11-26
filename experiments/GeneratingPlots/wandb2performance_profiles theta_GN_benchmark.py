import wandb
import json
import os
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots
from wandb_tools import set_plot_asthetics

histories = cache_run_histories(["Exp_Benchmark_6"])

generate_type = "benchmark"

for p in [2, 3]:
    # Filter runs
    # api = wandb.Api(timeout=10000)
    api = wandb.Api()
    wandb_runs = api.runs(
        path="ar3-project/all_experiments",
        filters={
            "group": "Exp_Benchmark_6",
            "config.update_sigma0": "TAYLOR",
            "tags": "training",
            "$or": [
                {
                    "config.update_type": "Interpolation_m",
                    "config.inner_stop_rule": "First_Order",
                    "config.p": p,
                },
                {
                    "config.update_type": "BGMS",
                    "config.inner_stop_rule": "First_Order",
                    "config.p": p,
                },
                {
                    "config.update_type": "Simple",
                    "config.inner_stop_rule": "First_Order",
                    "config.p": p,
                },
                {
                    "config.update_type": "Interpolation_m",
                    "config.inner_stop_rule": "ARP_Theory",
                    "config.p": p,
                },
                {
                    "config.update_type": "BGMS",
                    "config.inner_stop_rule": "ARP_Theory",
                    "config.p": p,
                },
                {
                    "config.update_type": "Simple",
                    "config.inner_stop_rule": "ARP_Theory",
                    "config.p": p,
                },
                {
                    "config.update_type": "Interpolation_m",
                    "config.inner_stop_rule": "General_Norm",
                    "config.inner_stop_theta": 1,
                    "config.update_use_prerejection": {"$ne": False},
                    "config.p": p,
                },
                {
                    "config.update_type": "BGMS",
                    "config.inner_stop_rule": "General_Norm",
                    "config.inner_stop_theta": 1,
                    "config.update_use_prerejection": {"$ne": False},
                    "config.p": p,
                },
                {
                    "config.update_type": "Simple",
                    "config.inner_stop_rule": "General_Norm",
                    "config.inner_stop_theta": 1,
                    "config.update_use_prerejection": {"$ne": False},
                    "config.p": p,
                },
            ],
        },
    )

    # Categorize runs
    method_parameters = [
        "p",
        "update_use_prerejection",
        "update_sigma0",
        "update_type",
        "inner_stop_rule",
        "inner_stop_theta",
    ]

    def method_sort_key(method: str) -> tuple[float, float, str]:
        method_dict = json.loads(method)
        inner_stop_rule = method_dict["inner_stop_rule"]
        # Ensure that "First_Order" < "ARP_Theory"
        if inner_stop_rule == "First_Order":
            return (1, 0, method)
        elif inner_stop_rule == "General_Norm":
            theta = method_dict.get("inner_stop_theta", 1)
            return (3, theta, method)
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
            "inner_stop_rule",
            "p",
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

    theta_tc = r"$\|\nabla m_k\| \leq 10^{{{exp}}} \|\mathbf{{s}}\|^3$"
    gn_tc = r"$\|\nabla t_k\| \leq \sigma_k \|\mathbf{s}\|^2$"
    first_order_tc = r"$\|\nabla m_k\| \leq 10^{{-9}}$"
    if p == 2:
        new_labels = [
            r"\textsf{AR$2$-BGMS}" + "\n" + first_order_tc,
            r"\textsf{AR$2$-Interp}" + "\n" + first_order_tc,
            r"\textsf{AR$2$-Simple}" + "\n" + first_order_tc,
            r"\textsf{AR$2$-BGMS}" + "\n" + theta_tc.format(exp=2),
            r"\textsf{AR$2$-Interp}" + "\n" + theta_tc.format(exp=2),
            r"\textsf{AR$2$-Simple}" + "\n" + theta_tc.format(exp=2),
            r"\textsf{AR$2$-BGMS}" + "\n" + gn_tc,
            r"\textsf{AR$2$-Interp}" + "\n" + gn_tc,
            r"\textsf{AR$2$-Simple}" + "\n" + gn_tc,
        ]
    else:
        new_labels = [
            r"\textsf{AR$3$-BGMS}" + "\n" + first_order_tc,
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + first_order_tc,
            r"\textsf{AR$3$-Simple\textsuperscript{+}}" + "\n" + first_order_tc,
            r"\textsf{AR$3$-BGMS}" + "\n" + theta_tc.format(exp=2),
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + theta_tc.format(exp=2),
            r"\textsf{AR$3$-Simple\textsuperscript{+}}" + "\n" + theta_tc.format(exp=2),
            r"\textsf{AR$3$-BGMS}" + "\n" + gn_tc,
            r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + gn_tc,
            r"\textsf{AR$3$-Simple\textsuperscript{+}}" + "\n" + gn_tc,
        ]

    os.makedirs("theta_GN", exist_ok=True)
    filename_prefix = f"theta_GN/performance_profile {generate_type} {p}"

    set_plot_asthetics()

    generate_gpp_plots(
        filename_prefix=filename_prefix,
        categorized_runs=categorized_runs,
        new_labels=new_labels,
        format="pgf",
    )
