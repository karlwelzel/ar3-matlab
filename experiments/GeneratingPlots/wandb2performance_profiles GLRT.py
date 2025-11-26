import wandb
import json
import os

from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots
from wandb_tools import set_plot_asthetics


# -------------------------------
# User choices
# -------------------------------
group = "Exp_Benchmark_8"

# p_values options:
#   [2]    -> only p = 2   (4 curves: AR2 only)
#   [3]    -> only p = 3   (4 curves: AR3 only)
#   [2, 3] -> both p = 2 and p = 3 (8 curves: AR2 + AR3)
p_values = [2, 3]  # change to [2], [3], or [2, 3] as needed


# -------------------------------
# 1. Cache histories
# -------------------------------
histories = cache_run_histories([group])


# -------------------------------
# 2. Fetch runs from W&B
# -------------------------------
api = wandb.Api()

wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={"group": group, "$or": [{"config.p": p} for p in p_values]},
)


# -------------------------------
# 3. Categorize runs
# -------------------------------
method_parameters = [
    "p",
    "inner_inner_solver",
    "inner_inner_stop_rule",
    "inner_inner_stop_theta",
    "inner_inner_stop_tolerance_g",
    "inner_solver",
    "inner_stop_rule",
]


def method_sort_key(method: str) -> tuple[int, int, str, str]:
    """
    Total ordering on (p, solver, stop_rule):

    For p = 2 (AR2):
        1. p=2, MCMR + ARP_Theory
        2. p=2, MCMR + First_Order
        3. p=2, GLRT + ARP_Theory
        4. p=2, GLRT + First_Order

    For p = 3 (AR3):
        5. p=3, MCMR + ARP_Theory
        6. p=3, MCMR + First_Order
        7. p=3, GLRT + ARP_Theory
        8. p=3, GLRT + First_Order

    Any other combination causes a ValueError
    """
    method_dict = json.loads(method)
    p_val = method_dict["p"]

    if p_val == 3:
        solver = method_dict["inner_inner_solver"]
        stop_rule = method_dict["inner_inner_stop_rule"]
    elif p_val == 2:
        solver = method_dict["inner_solver"]
        stop_rule = method_dict["inner_stop_rule"]
    else:
        raise ValueError(f"Unknown method: {method}")

    solver_priority = {"MCMR": 1, "GLRT": 2}

    return (p_val, solver_priority[solver], stop_rule, method)


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
        "inner_stop_theta",
        "inner_stop_tolerance_g",
        "inner_stop_max_iterations",
        "inner_update_type",
        "inner_update_decrease_measure",
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)

print(f"{categorized_runs.sorted_methods=}")
print(f"{len(categorized_runs.all_problem_setups)=}")


# -------------------------------
# 4. Build legend labels
# -------------------------------
new_labels: list[str] = []
for method in categorized_runs.sorted_methods:
    method_dict = json.loads(method)
    p_val = method_dict["p"]

    if p_val == 3:
        solver = method_dict["inner_inner_solver"]
        stop_rule = method_dict["inner_inner_stop_rule"]
    elif p_val == 2:
        solver = method_dict["inner_solver"]
        stop_rule = method_dict["inner_stop_rule"]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Rename solver label (MCMR -> MCM)
    if solver == "MCMR":
        solver_label = "MCM"
    else:
        solver_label = solver

    # Select accuracy labels (same meaning for p=2 and p=3)
    if stop_rule == "First_Order":
        if p_val == 3:
            acc_label = r"$\|\nabla \tilde{m}_k\| \leq 10^{-10}$"
        else:
            acc_label = r"$\|\nabla m_k\| \leq 10^{-9}$"
    elif stop_rule == "ARP_Theory":
        if p_val == 3:
            acc_label = r"$\|\nabla \tilde{m}_k\| \leq 10^2 \|\tilde{\mathbf{s}}_k\|^3$"
        else:
            acc_label = r"$\|\nabla m_k\| \leq 10^2 \|\mathbf{s}_k\|^2$"
    else:
        acc_label = stop_rule

    # Include p in the label so p=2 and p=3 are distinguishable
    if p_val == 3:
        label = (
            rf"\textsf{{AR$3$-Interp\textsuperscript{{+}} "
            rf"+ {solver_label}, {acc_label}}}"
        )
    else:
        label = rf"\textsf{{AR$2$-Interp " rf"+ {solver_label}, {acc_label}}}"
    new_labels.append(label)

print("Legend labels (in order):", new_labels)


# -------------------------------
# 5. Generate performance profile plots
# -------------------------------
os.makedirs("GLRT", exist_ok=True)

sorted_p = sorted(p_values)
p_suffix = "p" + "".join(str(p) for p in sorted_p)
filename_prefix = f"GLRT/performance_profile {p_suffix}"

set_plot_asthetics()

generate_gpp_plots(
    filename_prefix=filename_prefix,
    categorized_runs=categorized_runs,
    new_labels=new_labels,
    format="pgf",
    legend_ncols=2,
)
