import json
import os

import matplotlib.pyplot as plt
import wandb

from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import generate_gpp_plots


# -------------------------------
# User choices
# -------------------------------
group = "Exp_Benchmark_8"

# p_values options:
#   [2]    -> only p = 2   (4 curves: AR2 only)
#   [3]    -> only p = 3   (4 curves: AR3 only)
#   [2, 3] -> both p = 2 and p = 3 (8 curves: AR2 + AR3)
#   None   -> treat as [2, 3]
p_values = [2,3]  # change to [2], [3], or [2, 3] as needed


# -------------------------------
# Local plot aesthetics
# (no dependency on settings/math.tex)
# -------------------------------
def set_plot_asthetics():
    """
    Use the PGF backend with a minimal LaTeX preamble that does NOT
    import ../settings/math.tex, so both PGF and PNG work.
    """
    plt.switch_backend("pgf")
    plt.rc(
        "pgf",
        texsystem="pdflatex",
        rcfonts=False,
        preamble=r"""
            \usepackage{amsmath,amssymb}
            \usepackage{bm}
        """,
    )
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Computer Modern Roman")
    plt.rc("lines", markersize=2.0, linewidth=1.0)
    plt.rc("savefig", dpi=300)


# -------------------------------
# 1. Cache histories
# -------------------------------
groups = [group]
histories = cache_run_histories(groups)

# -------------------------------
# 2. Fetch runs from W&B
# -------------------------------
api = wandb.Api()

include_p2 = p_values is None or 2 in p_values
include_p3 = p_values is None or 3 in p_values

filters: dict = {"group": group}
or_clauses: list[dict] = []

# AR2 (p = 2): inner_* keys
if include_p2:
    for solver in ("MCMR", "GLRT"):
        for stop_rule in ("ARP_Theory", "First_Order"):
            or_clauses.append(
                {
                    "config.inner_stop_rule": stop_rule,
                    "config.inner_solver": solver,
                    "config.p": 2,
                }
            )

# AR3 (p = 3): inner_inner_* keys
if include_p3:
    for solver in ("MCMR", "GLRT"):
        for stop_rule in ("ARP_Theory", "First_Order"):
            or_clauses.append(
                {
                    "config.inner_inner_stop_rule": stop_rule,
                    "config.inner_inner_solver": solver,
                    "config.p": 3,
                }
            )

if or_clauses:
    filters["$or"] = or_clauses

wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters=filters,
)

p_seen = {run.config.get("p", None) for run in wandb_runs}
inner_solvers = {run.config.get("inner_solver", None) for run in wandb_runs}
inner_stop_rules = {run.config.get("inner_stop_rule", None) for run in wandb_runs}
inner_inner_solvers = {run.config.get("inner_inner_solver", None) for run in wandb_runs}
inner_inner_stop_rules = {
    run.config.get("inner_inner_stop_rule", None) for run in wandb_runs
}

print("Distinct p values:", p_seen)
print("Distinct inner_solver values (p=2):", inner_solvers)
print("Distinct inner_stop_rule values (p=2):", inner_stop_rules)
print("Distinct inner_inner_solver values (p=3):", inner_inner_solvers)
print("Distinct inner_inner_stop_rule values (p=3):", inner_inner_stop_rules)


# -------------------------------
# 3. Categorize runs
# -------------------------------
method_parameters = [
    "p",
    "inner_inner_solver",
    "inner_inner_stop_rule",
    "inner_solver",
    "inner_stop_rule",
]


def method_sort_key(method: str) -> tuple[int, str]:
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

    Any other combination is placed at the end.
    """
    method_dict = json.loads(method)
    p_val = method_dict.get("p", None)

    if p_val == 3:
        solver = method_dict.get("inner_inner_solver", "")
        stop_rule = method_dict.get("inner_inner_stop_rule", "")
    elif p_val == 2:
        solver = method_dict.get("inner_solver", "")
        stop_rule = method_dict.get("inner_stop_rule", "")
    else:
        # Fallback: pick whatever is present
        solver = method_dict.get("inner_inner_solver") or method_dict.get(
            "inner_solver", ""
        )
        stop_rule = method_dict.get("inner_inner_stop_rule") or method_dict.get(
            "inner_stop_rule", ""
        )

    idx = 99
    if p_val == 2:
        if solver == "MCMR" and stop_rule == "ARP_Theory":
            idx = 1
        elif solver == "MCMR" and stop_rule == "First_Order":
            idx = 2
        elif solver == "GLRT" and stop_rule == "ARP_Theory":
            idx = 3
        elif solver == "GLRT" and stop_rule == "First_Order":
            idx = 4
    elif p_val == 3:
        if solver == "MCMR" and stop_rule == "ARP_Theory":
            idx = 5
        elif solver == "MCMR" and stop_rule == "First_Order":
            idx = 6
        elif solver == "GLRT" and stop_rule == "ARP_Theory":
            idx = 7
        elif solver == "GLRT" and stop_rule == "First_Order":
            idx = 8

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
        "inner_stop_theta",
        "inner_stop_tolerance_g",
        "inner_stop_max_iterations",
        "inner_inner_stop_tolerance_g",
        "inner_update_type",
        "inner_update_decrease_measure",
        # NOTE: we deliberately do NOT ignore p, inner_* or inner_inner_* keys,
        # since they are part of method_parameters.
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)

print(f"{categorized_runs.sorted_methods=}")
print(f"{len(categorized_runs.all_problem_setups)=}")


# -------------------------------
# 3a. Redefine problem_setup by problem only
#      (so p=2 and p=3 share the same setup)
# -------------------------------
for run in categorized_runs:
    # Use only the base problem ID as the setup identifier
    run.problem_setup = json.dumps({"problem": run.problem}, sort_keys=True)

# Rebuild all_problem_setups and all_combinations
categorized_runs.all_problem_setups = set()
categorized_runs.all_combinations = set()
for run in categorized_runs:
    categorized_runs.all_problem_setups.add(run.problem_setup)
    categorized_runs.all_combinations.add((run.method, run.problem_setup))

print(
    "After redefining problem_setup by problem only, "
    f"len(categorized_runs.all_problem_setups)={len(categorized_runs.all_problem_setups)}"
)


# -------------------------------
# 3b. Drop problem setups that are not present for all methods
#     (avoid NaNs in gpp_costs)
# -------------------------------
methods = categorized_runs.sorted_methods
ps_by_method = {
    m: {run.problem_setup for run in categorized_runs if run.method == m}
    for m in methods
}

common_ps = set.intersection(*ps_by_method.values()) if methods else set()
if len(common_ps) < len(categorized_runs.all_problem_setups):
    print(
        "Dropping problem setups without all methods:",
        len(categorized_runs.all_problem_setups) - len(common_ps),
    )
    filtered = [run for run in categorized_runs if run.problem_setup in common_ps]
    categorized_runs.clear()
    categorized_runs.extend(filtered)
    # Rebuild the cached sets again
    categorized_runs.all_methods = {run.method for run in categorized_runs}
    categorized_runs.all_problem_setups = {run.problem_setup for run in categorized_runs}
    categorized_runs.all_combinations = {
        (run.method, run.problem_setup) for run in categorized_runs
    }

print(
    f"After filtering, len(categorized_runs.all_problem_setups)={len(categorized_runs.all_problem_setups)}"
)

if not categorized_runs.all_problem_setups:
    raise RuntimeError(
        "No common problem setups across the selected methods. "
        "Check that p=2 and p=3 runs really share the same underlying problems "
        "and that the W&B filters are correct."
    )


# -------------------------------
# 3c. Build legend labels
# -------------------------------
new_labels: list[str] = []
for method in categorized_runs.sorted_methods:
    method_dict = json.loads(method)
    p_val = method_dict.get("p", "?")

    if p_val == 3:
        solver = method_dict.get("inner_inner_solver", "UNKNOWN")
        stop_rule = method_dict.get("inner_inner_stop_rule", "UNKNOWN")
    elif p_val == 2:
        solver = method_dict.get("inner_solver", "UNKNOWN")
        stop_rule = method_dict.get("inner_stop_rule", "UNKNOWN")
    else:
        solver = method_dict.get("inner_inner_solver") or method_dict.get(
            "inner_solver", "UNKNOWN"
        )
        stop_rule = method_dict.get("inner_inner_stop_rule") or method_dict.get(
            "inner_stop_rule", "UNKNOWN"
        )

    # Renaming solver label (MCMR -> MCM)
    if solver == "MCMR":
        solver_label = "MCM"
    else:
        solver_label = solver

    # Accuracy labels (same meaning for p=2 and p=3)
    if stop_rule == "First_Order":
        if p_val == 3:
            acc_label = r"$\|\nabla \tilde{m}_k\| \leq 10^{-10}$"
        else:
            acc_label = r"$\|\nabla m_k\| \leq 10^{-9}$"
    elif stop_rule == "ARP_Theory":
        if p_val == 3:
            acc_label = r"$\|\nabla \tilde{m}_k\| \leq 10^2 \|\tilde{s}_k\|^2$"
        else:
            acc_label = r"$\|\nabla m_k\| \leq 10^2 \|s_k\|^2$"
    else:
        acc_label = stop_rule

    # Include p in the label so p=2 and p=3 are distinguishable
    if p_val == 3:
        label = (
            rf"\textsf{{AR3-Interp\textsuperscript{{+}} "
            rf"+ {solver_label}, {acc_label}}}"
        )
    else:
        label = (
            rf"\textsf{{AR2-Interp "
            rf"+ {solver_label}, {acc_label}}}"
        )
    new_labels.append(label)

print("Legend labels (in order):", new_labels)


# -------------------------------
# 4. Generate performance profile plots (PGF + PNG)
# -------------------------------
os.makedirs("GLRT", exist_ok=True)

if p_values is None:
    p_suffix = "p_all"
else:
    sorted_p = sorted(p_values)
    if sorted_p == [2, 3]:
        p_suffix = "p2-3"
    else:
        p_suffix = f"p{sorted_p[0]}"

filename_prefix = f"GLRT/performance_profile {p_suffix}"

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
