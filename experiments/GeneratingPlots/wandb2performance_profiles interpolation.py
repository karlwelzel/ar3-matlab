import wandb
import json
from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots
from wandb_tools import set_plot_asthetics

inner_stop_rule = "First_Order"

histories = cache_run_histories(["Exp_Benchmark_0"])

# Filter runs
# api = wandb.Api(timeout=10000)
api = wandb.Api()
wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={
        "group": "Exp_Benchmark_0",
        "config.update_sigma0": "TAYLOR",
        "config.inner_solver": "AR2",
        "config.p": 3,
        "config.inner_stop_rule": inner_stop_rule,
        "config.update_use_prerejection": False,
        "tags": "training",
        "$and": [
            {
                "$or": [
                    {"config.update_type": "Interpolation_m"},
                    {"config.update_type": "Simple"},
                    # {"config.update_type": "BGMS"},
                ]
            },
        ],
    },
)

# Categorize runs
method_parameters = [
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
    dump_type="gpp",
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
print(f"{len(categorized_runs.all_problem_setups)=}")

new_labels = [
    r"\textsf{AR$3$-Simple}",
    r"\textsf{AR$3$-Interp}",
]

filename_prefix = f"interpolation/performance_profile {inner_stop_rule}"

set_plot_asthetics()

generate_gpp_plots(
    filename_prefix=filename_prefix,
    categorized_runs=categorized_runs,
    new_labels=new_labels,
    format="pgf",
)
