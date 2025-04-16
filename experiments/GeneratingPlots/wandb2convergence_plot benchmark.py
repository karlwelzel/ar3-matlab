import wandb
import json
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import convergence_dot_plot
from wandb_tools import set_plot_asthetics


groups = ["Exp_Benchmark_0", "Exp_Benchmark_3"]
histories = cache_run_histories(groups)

# Filter runs
api = wandb.Api(timeout=10000)

wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={
        "group": {"$ne": "Exp_Benchmark_6"},
        "tags": "benchmark",
        "$and": [
            {
                "$or": [
                    {
                        "config.update_sigma0": "TAYLOR",
                        "config.update_type": "Interpolation_m",
                        "config.inner_stop_rule": "ARP_Theory",
                        "config.inner_stop_theta": 1e-2,
                        "config.p": 2,
                    },
                    {
                        "config.update_sigma0": "TAYLOR",
                        "config.update_type": "Interpolation_m",
                        "config.inner_stop_rule": "ARP_Theory",
                        "config.inner_stop_tolerance_g": 1e-9,  # Hack to get theta=100
                        "config.update_use_prerejection": True,
                        "config.p": 3,
                    },
                    # {
                    #     "config.update_sigma0": "TAYLOR",
                    #     "config.update_type": "BGMS",
                    #     "config.inner_solver": "MCMR",
                    #     "config.inner_stop_rule": "ARP_Theory",
                    #     "config.inner_stop_theta": 1e-2,
                    #     "config.p": 2,
                    # },
                    {
                        "config.update_sigma0": "TAYLOR",
                        "config.update_type": "BGMS",
                        "config.inner_solver": "AR2",
                        "config.inner_stop_rule": "ARP_Theory",
                        "config.inner_stop_tolerance_g": 1e-9,  # Hack to get theta=100
                        "config.update_use_prerejection": True,
                        "config.p": 3,
                    },
                    # {
                    #     "config.update_type": "BGMS",
                    #     "config.inner_solver": "GENCAN",
                    #     "config.inner_stop_rule": "ARP_Theory",
                    #     "config.p": 2,
                    # },
                    {
                        "config.update_type": "BGMS",
                        "config.inner_solver": "GENCAN",
                        "config.inner_stop_rule": "ARP_Theory",
                        "config.p": 3,
                    },
                    # {
                    #     "config.update_sigma0": "TAYLOR",
                    #     "config.update_type": "Interpolation_m",
                    #     "config.inner_stop_rule": "First_Order",
                    #     "config.p": 2,
                    # },
                    # {
                    #     "config.update_sigma0": "TAYLOR",
                    #     "config.update_type": "Interpolation_m",
                    #     "config.inner_stop_rule": "First_Order",
                    #     "config.update_use_prerejection": True,
                    #     "config.p": 3,
                    # },
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
    "p",
    "update_sigma0",
    "update_type",
    "inner_stop_rule",
    "inner_stop_theta",
    "inner_solver",
]


def method_sort_key(method: str) -> tuple[float, str, bool, str]:
    method_dict = json.loads(method)
    update_type = method_dict["update_type"]
    inner_solver = method_dict["inner_solver"]
    inner_stop_rule = method_dict["inner_stop_rule"]
    # Ensure that "Simple" < "Interpolation_m" < "BGMS"
    if update_type == "Simple":
        return (1, inner_stop_rule, inner_solver == "GENCAN", method)
    elif update_type == "Interpolation_m":
        return (2, inner_stop_rule, inner_solver == "GENCAN", method)
    else:
        return (3, inner_stop_rule, inner_solver == "GENCAN", method)


categorized_runs = categorize_runs(
    wandb_runs=wandb_runs,
    histories=histories,
    method_parameters=method_parameters,
    ignore=[
        "wandb_project",
        "wandb_group",
        "update_use_prerejection",
        "stop_tolerance_g",
        "stop_rule",
        "inner_solver",
        "inner_stop_theta",
        "inner_stop_tolerance_g",
        "inner_inner_stop_tolerance_g",
        "inner_update_type",
        "inner_update_decrease_measure",
        "inner_inner_solver",
        "inner_inner_stop_rule",
        "update_sigma0",
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)
print(f"{categorized_runs.sorted_methods=}")
print(f"{categorized_runs.all_problem_setups=}")

theta_tc = r"$\|\nabla m_k\| \leq 10^{{{exp}}} \|\mathbf{{s}}\|^{{{exp2}}}$"
first_order_tc = r"$\|\nabla m_k\| \leq 10^{{-9}}$"
custom_titles = [
    r"\textsf{AR$2$-Interp}" + "\n" + theta_tc.format(exp=-2, exp2=2),
    r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + theta_tc.format(exp=2, exp2=3),
    # r"\textsf{AR$2$-Interp}" + "\n" + first_order_tc,
    # r"\textsf{AR$3$-Interp\textsuperscript{+}}" + "\n" + first_order_tc,
    # r"\textsf{AR$2$-BGMS}" + "\n" + theta_tc.format(exp=-2, exp2=2),
    r"\textsf{AR$3$-BGMS}" + "\n" + theta_tc.format(exp=2, exp2=3),
    # r"\textsf{AR$2$-BGMS/Gencan}" + "\n" + theta_tc.format(exp=2, exp2=2),
    r"\textsf{AR$3$-BGMS/Gencan}" + "\n" + theta_tc.format(exp=2, exp2=3),
]

set_plot_asthetics()

grid = convergence_dot_plot(
    categorized_runs,
    row_order=["5", "13"],
    col_order=categorized_runs.sorted_methods,
    col_titles=custom_titles,
)

filename = "benchmark/convergence.pgf"
grid.figure.savefig(filename, dpi=100)
print(f"Saved {filename!r}")
