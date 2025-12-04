import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import wandb

from wandb_tools import cache_run_histories
from wandb_tools import categorize_runs
from wandb_tools import set_plot_asthetics


# ============================================================
# Configuration
# ============================================================
GROUPS = ["Exp_Benchmark_9"]
histories = cache_run_histories(GROUPS)

THRESH_NORMG = 1e-6  # threshold for marking "unsolved" points

# Per-method styles: consistent across all plots
# (p, solver) -> style
STYLE_BY_METHOD = {
    (2, "MCMR"): {"color": "C0", "marker": "s", "linestyle": "-"},
    (2, "GLRT"): {"color": "C1", "marker": "o", "linestyle": "--"},
    (3, "MCMR"): {"color": "C2", "marker": "s", "linestyle": "-"},
    (3, "GLRT"): {"color": "C3", "marker": "o", "linestyle": "--"},
}

# Desired ordering of methods
ORDER_INDEX = {
    (2, "MCMR"): 0,  # AR2 + MCMR
    (2, "GLRT"): 1,  # AR2 + GLRT
    (3, "MCMR"): 2,  # AR3 + MCMR
    (3, "GLRT"): 3,  # AR3 + GLRT
}


def get_final(history, col):
    """Return the last value of column `col` from the history, or None."""
    last = history.iloc[-1]
    if col in last.index:
        return float(last[col])
    return None


def get_dimension(problem_value):
    """Extract 'dim' from problem description (string or dict)."""
    if isinstance(problem_value, str):
        prob = json.loads(problem_value)
    else:
        prob = problem_value
    return int(prob["dim"])


def method_id_from_method_string(method_str):
    """
    Return (p, solver) for a method JSON string:
      - solver = inner_solver if p=2
      - solver = inner_inner_solver if p=3
    """
    m = json.loads(method_str)
    p_val = m["p"]
    if p_val == 2:
        solver = m["inner_solver"]
    elif p_val == 3:
        solver = m["inner_inner_solver"]
    else:
        raise ValueError(f"Unexpected p value: {p_val}")
    return p_val, solver


def method_label(p_val, solver):
    """LaTeX label for a method."""
    solver_label = "MCM" if solver == "MCMR" else solver
    if p_val == 2:
        return rf"\textsf{{AR2-Interp + {solver_label}}}"
    else:  # p_val == 3
        return rf"\textsf{{AR3-Interp\textsuperscript{{+}} + {solver_label}}}"


def method_sort_key(method_str):
    """Sort methods according to ORDER_INDEX."""
    p_val, solver = method_id_from_method_string(method_str)
    return ORDER_INDEX[(p_val, solver)]


def mark_unsolved(ax, dims, y_vals, metrics_for_dim, color, tol=THRESH_NORMG):
    """Overlay empty circles where final norm_g > tol."""
    bad_x, bad_y = [], []
    for d, y in zip(dims, y_vals):
        g = metrics_for_dim[d].get("norm_g")
        if g is not None and g > tol:
            bad_x.append(d)
            bad_y.append(y)

    if bad_x:
        ax.plot(
            bad_x,
            bad_y,
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=color,
        )


# ============================================================
# Fetch runs and categorize
# ============================================================
api = wandb.Api(timeout=10000)
wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={
        "group": {"$in": GROUPS},
        "$and": [
            {
                "$or": [
                    {"config.inner_solver": "GLRT", "config.p": 2},
                    {"config.inner_solver": "MCMR", "config.p": 2},
                    {"config.inner_inner_solver": "GLRT", "config.p": 3},
                    {"config.inner_inner_solver": "MCMR", "config.p": 3},
                ]
            }
        ],
    },
)

print(f"Number of runs fetched: {len(wandb_runs)}")

method_parameters = ["p", "inner_solver", "inner_inner_solver"]

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
        "update_type",
        "inner_stop_rule",
        "inner_stop_theta",
        "inner_stop_tolerance_g",
        "inner_inner_stop_tolerance_g",
        "inner_update_type",
        "inner_update_decrease_measure",
        "inner_inner_stop_rule",
        "update_sigma0",
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)

print(f"sorted_methods = {categorized_runs.sorted_methods}")
print(f"all_problem_setups = {categorized_runs.all_problem_setups}")


# ============================================================
# Aggregate final metrics per (p, solver, dim)
# ============================================================
# method_to_dim_metrics[(p, solver)][dim] = {time, total_hvp, total_chol, norm_g}
method_to_dim_metrics = defaultdict(dict)

for run in categorized_runs:
    p_val, solver = method_id_from_method_string(run.method)
    dim = get_dimension(run.problem)
    hist = run.history

    time_val = get_final(hist, "time")
    total_hvp = get_final(hist, "total_hvp")
    total_chol = get_final(hist, "total_chol")
    norm_g_val = get_final(hist, "norm_g")

    key = (p_val, solver)
    current = method_to_dim_metrics[key].get(dim)

    # If multiple runs exist for the same (p, solver, dim), keep the one with smaller final time
    if current is None or time_val < current["time"]:
        method_to_dim_metrics[key][dim] = {
            "time": time_val,
            "total_hvp": total_hvp,
            "total_chol": total_chol,
            "norm_g": norm_g_val,
        }

print("Methods and available dimensions:")
all_dims = set()
for key, dim_metrics in method_to_dim_metrics.items():
    dims_here = sorted(dim_metrics.keys())
    all_dims.update(dims_here)
    print(f"  {key}: dims = {dims_here}")
all_dims = sorted(all_dims)


# ============================================================
# Plot setup
# ============================================================
set_plot_asthetics()
os.makedirs("rosenbrock_scaling", exist_ok=True)


# ============================================================
# Plot 1: Time vs dimension (log2 x, log y)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(4, 3))

for (p_val, solver), dim_metrics in sorted(
    method_to_dim_metrics.items(), key=lambda kv: ORDER_INDEX[kv[0]]
):
    dims = sorted(dim_metrics.keys())
    times = [dim_metrics[d]["time"] for d in dims]

    label = method_label(p_val, solver)
    style = STYLE_BY_METHOD[(p_val, solver)]

    (line,) = ax1.plot(dims, times, label=label, **style)
    color = line.get_color()

    mark_unsolved(ax1, dims, times, dim_metrics, color)

ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks(all_dims)
ax1.set_xticklabels([str(d) for d in all_dims])
ax1.set_xlabel("Dimension")
ax1.set_ylabel("Time (s)")
ax1.set_title("Time vs Dimension")
ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
ax1.legend()
fig1.tight_layout()

pgf_time = "rosenbrock_scaling/rosenbrock_time_vs_dim_log2.pgf"
png_time = "rosenbrock_scaling/rosenbrock_time_vs_dim_log2.png"
fig1.savefig(pgf_time)
fig1.savefig(png_time, dpi=300)
print(f"Saved {pgf_time!r} and {png_time!r}")


# ============================================================
# Plot 2: Total HVP vs dimension (GLRT only, log2 x, log y)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(4, 3))

for (p_val, solver), dim_metrics in sorted(
    method_to_dim_metrics.items(), key=lambda kv: ORDER_INDEX[kv[0]]
):
    if solver != "GLRT":
        continue

    dims = sorted(dim_metrics.keys())
    dims_hvp, hvps = [], []
    for d in dims:
        v = dim_metrics[d]["total_hvp"]
        if v is None or v <= 0:
            continue
        dims_hvp.append(d)
        hvps.append(v)

    if not dims_hvp:
        continue

    label = method_label(p_val, solver)
    style = STYLE_BY_METHOD[(p_val, solver)]
    (line,) = ax2.plot(dims_hvp, hvps, label=label, **style)
    color = line.get_color()

    sub_metrics = {d: dim_metrics[d] for d in dims_hvp}
    mark_unsolved(ax2, dims_hvp, hvps, sub_metrics, color)

ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.set_xticks(all_dims)
ax2.set_xticklabels([str(d) for d in all_dims])
ax2.set_xlabel("Dimension")
ax2.set_ylabel("Total HVPs")
ax2.set_title("HVPs vs Dimension")
ax2.grid(True, which="both", linestyle=":", linewidth=0.5)
ax2.legend()
fig2.tight_layout()

pgf_hvp = "rosenbrock_scaling/rosenbrock_hvp_vs_dim_log2.pgf"
png_hvp = "rosenbrock_scaling/rosenbrock_hvp_vs_dim_log2.png"
fig2.savefig(pgf_hvp)
fig2.savefig(png_hvp, dpi=300)
print(f"Saved {pgf_hvp!r} and {png_hvp!r}")


# ============================================================
# Plot 3: Total Chol vs dimension (MCMR only, log2 x, log y)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(4, 3))

for (p_val, solver), dim_metrics in sorted(
    method_to_dim_metrics.items(), key=lambda kv: ORDER_INDEX[kv[0]]
):
    if solver != "MCMR":
        continue

    dims = sorted(dim_metrics.keys())
    dims_chol, chols = [], []
    for d in dims:
        v = dim_metrics[d]["total_chol"]
        if v is None or v <= 0:
            continue
        dims_chol.append(d)
        chols.append(v)

    if not dims_chol:
        continue

    label = method_label(p_val, solver)
    style = STYLE_BY_METHOD[(p_val, solver)]
    (line,) = ax3.plot(dims_chol, chols, label=label, **style)
    color = line.get_color()

    sub_metrics = {d: dim_metrics[d] for d in dims_chol}
    mark_unsolved(ax3, dims_chol, chols, sub_metrics, color)

ax3.set_xscale("log", base=2)
ax3.set_yscale("log")
ax3.set_xticks(all_dims)
ax3.set_xticklabels([str(d) for d in all_dims])
ax3.set_xlabel("Dimension")
ax3.set_ylabel("Total Cholesky factorizations")
ax3.set_title("Cholesky vs Dimension")
ax3.grid(True, which="both", linestyle=":", linewidth=0.5)
ax3.legend()
fig3.tight_layout()

pgf_chol = "rosenbrock_scaling/rosenbrock_chol_vs_dim_log2.pgf"
png_chol = "rosenbrock_scaling/rosenbrock_chol_vs_dim_log2.png"
fig3.savefig(pgf_chol)
fig3.savefig(png_chol, dpi=300)
print(f"Saved {pgf_chol!r} and {png_chol!r}")
