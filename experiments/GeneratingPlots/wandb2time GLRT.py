import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

from wandb_tools import cache_run_summaries
from wandb_tools import categorize_runs
from wandb_tools import set_plot_asthetics


# ============================================================
# Configuration
# ============================================================
GROUPS = ["Exp_Benchmark_9"]
histories = cache_run_summaries(GROUPS)

THRESH_NORMG = 1e-6  # threshold for empty markers

# Consistent across all plots
# (p, solver) -> style
STYLE_BY_METHOD = {
    (1, "BASE"): {"color": "C0", "marker": "*", "linestyle": "-"},
    (2, "MCMR"): {"color": "C1", "marker": "s", "linestyle": "-."},
    (2, "GLRT"): {"color": "C2", "marker": "o", "linestyle": "--"},
    (3, "MCMR"): {"color": "C3", "marker": "s", "linestyle": "-."},
    (3, "GLRT"): {"color": "C4", "marker": "o", "linestyle": "--"},
}

# Desired ordering of methods in plotting
ORDER_INDEX = {
    (1, "BASE"): 0,   # AR1
    (2, "MCMR"): 1,   # AR2 + MCMR
    (2, "GLRT"): 2,   # AR2 + GLRT
    (3, "MCMR"): 3,   # AR3 + MCMR
    (3, "GLRT"): 4,   # AR3 + GLRT
}


# ============================================================
# Helpers
# ============================================================
def get_final(history, col):
    """Return the last value of column `col` from the history, or None."""
    last = history.iloc[-1]
    if col in last.index:
        return float(last[col])
    return None


def get_dimension(problem_value):
    """Extract 'dim' from problem description."""
    if isinstance(problem_value, str):
        prob = json.loads(problem_value)
    else:
        prob = problem_value
    return int(prob["dim"])


def method_id_from_method_string(method_str):
    m = json.loads(method_str)
    p_val = m["p"]
    if p_val == 1:
        solver = "BASE"
    elif p_val == 2:
        solver = m["inner_solver"]
    elif p_val == 3:
        solver = m["inner_inner_solver"]
    else:
        raise ValueError(f"Unexpected p value: {p_val}")
    return p_val, solver


def method_label(p_val, solver):
    """LaTeX label for a method."""
    if p_val == 1:
        return r"\textsf{AR1}"
    solver_label = "MCM" if solver == "MCMR" else solver
    if p_val == 2:
        return rf"\textsf{{AR2-Interp + {solver_label}}}"
    else:  # p_val == 3
        return rf"\textsf{{AR3-Interp\textsuperscript{{+}} + {solver_label}}}"


def method_sort_key(method_str):
    p_val, solver = method_id_from_method_string(method_str)
    return ORDER_INDEX[(p_val, solver)]


def mark_unsolved(ax, dims, y_vals, metrics_for_dim, color, marker_shape):
    """
    Overlay empty markers where norm_g > THRESH_NORMG.
    Uses the same shape as the method but with empty face.
    """
    bad_x, bad_y = [], []
    for d, y in zip(dims, y_vals):
        g = metrics_for_dim[d].get("norm_g")
        if g is not None and g > THRESH_NORMG:
            bad_x.append(d)
            bad_y.append(y)

    if bad_x:
        ax.plot(
            bad_x,
            bad_y,
            linestyle="None",
            marker=marker_shape, # Use specific marker shape
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=color,
            zorder=10 # Ensure it sits on top
        )


# ============================================================
# Fetch runs and categorize
# ============================================================
api = wandb.Api(timeout=10000)
wandb_runs = api.runs(
    path="ar3-project/all_experiments",
    filters={"group": {"$in": GROUPS}},
)

print(f"Number of runs fetched: {len(wandb_runs)}")

method_parameters = ["p", "inner_solver", "inner_inner_solver"]

categorized_runs = categorize_runs(
    wandb_runs=wandb_runs,
    histories=histories,
    method_parameters=method_parameters,
    ignore=[
        "wandb_project", "wandb_group", "update_use_prerejection", "stop_tolerance_g",
        "stop_rule", "update_type", "inner_stop_rule", "inner_stop_theta",
        "inner_stop_tolerance_g", "inner_inner_stop_tolerance_g",
        "inner_update_type", "inner_update_decrease_measure",
        "inner_inner_stop_rule", "update_sigma0",
    ],
    method_sort_key=method_sort_key,
    error_on_duplicate=True,
)


# ============================================================
# Aggregate final metrics
# ============================================================
method_to_dim_metrics = defaultdict(dict)

for run in categorized_runs:
    p_val, solver = method_id_from_method_string(run.method)
    if (p_val, solver) not in STYLE_BY_METHOD:
        continue

    dim = get_dimension(run.problem)
    hist = run.history

    time_val = get_final(hist, "time")
    total_hvp = get_final(hist, "total_hvp")
    total_chol = get_final(hist, "total_chol")
    norm_g_val = get_final(hist, "norm_g")

    key = (p_val, solver)
    current = method_to_dim_metrics[key].get(dim)

    if current is None or (time_val is not None and time_val < current["time"]):
        method_to_dim_metrics[key][dim] = {
            "time": time_val,
            "total_hvp": total_hvp,
            "total_chol": total_chol,
            "norm_g": norm_g_val,
        }


# ============================================================
# Plot setup
# ============================================================
set_plot_asthetics()
os.makedirs("rosenbrock_scaling", exist_ok=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 3.2))

# Store handles for legend
legend_handles_map = {}
legend_labels_map = {}

# Define exponents 2, 4, ..., 14 for x-axis ticks
exponents = range(2, 16, 2) 
custom_ticks = [2**e for e in exponents]
custom_labels = [rf"$2^{{{e}}}$" for e in exponents]


# ============================================================
# Plot 1: Time vs dimension
# ============================================================
for (p_val, solver), dim_metrics in sorted(
    method_to_dim_metrics.items(), key=lambda kv: ORDER_INDEX[kv[0]]
):
    dims = sorted(dim_metrics.keys())
    times = [dim_metrics[d]["time"] for d in dims]

    label = method_label(p_val, solver)
    style = STYLE_BY_METHOD[(p_val, solver)]

    (line,) = ax1.plot(dims, times, label=label, **style)
    color = line.get_color()
    
    # Save handle for legend
    legend_handles_map[(p_val, solver)] = line
    legend_labels_map[(p_val, solver)] = label

    # Unsolved markers use same shape (d, s, or o) but empty face
    mark_unsolved(ax1, dims, times, dim_metrics, color, style["marker"])

ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks(custom_ticks)
ax1.set_xticklabels(custom_labels)
ax1.set_xlabel("Dimension")
ax1.set_ylabel("Time (s)")
ax1.set_title("Time vs dimension")
ax1.grid(True, which="both", linestyle=":", linewidth=0.5)


# ============================================================
# Plot 2: HVP vs dimension
# ============================================================
for (p_val, solver), dim_metrics in sorted(
    method_to_dim_metrics.items(), key=lambda kv: ORDER_INDEX[kv[0]]
):
    if solver != "GLRT": continue

    dims = sorted(dim_metrics.keys())
    dims_hvp, hvps = [], []
    for d in dims:
        v = dim_metrics[d]["total_hvp"]
        if v is not None and v > 0:
            dims_hvp.append(d)
            hvps.append(v)
    if not dims_hvp: continue

    label = method_label(p_val, solver)
    style = STYLE_BY_METHOD[(p_val, solver)]
    (line,) = ax2.plot(dims_hvp, hvps, label=label, **style)
    color = line.get_color()

    sub_metrics = {d: dim_metrics[d] for d in dims_hvp}
    mark_unsolved(ax2, dims_hvp, hvps, sub_metrics, color, style["marker"])

ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.set_xticks(custom_ticks)
ax2.set_xticklabels(custom_labels)
ax2.set_xlabel("Dimension")
ax2.set_ylabel("Total HVPs")
ax2.set_title("HVPs (GLRT) vs dimension")
ax2.grid(True, which="both", linestyle=":", linewidth=0.5)


# ============================================================
# Plot 3: Cholesky vs dimension
# ============================================================
for (p_val, solver), dim_metrics in sorted(
    method_to_dim_metrics.items(), key=lambda kv: ORDER_INDEX[kv[0]]
):
    if solver != "MCMR": continue

    dims = sorted(dim_metrics.keys())
    dims_chol, chols = [], []
    for d in dims:
        v = dim_metrics[d]["total_chol"]
        if v is not None and v > 0:
            dims_chol.append(d)
            chols.append(v)
    if not dims_chol: continue

    label = method_label(p_val, solver)
    style = STYLE_BY_METHOD[(p_val, solver)]
    (line,) = ax3.plot(dims_chol, chols, label=label, **style)
    color = line.get_color()

    sub_metrics = {d: dim_metrics[d] for d in dims_chol}
    mark_unsolved(ax3, dims_chol, chols, sub_metrics, color, style["marker"])

ax3.set_xscale("log", base=2)
ax3.set_yscale("log")
ax3.set_xticks(custom_ticks)
ax3.set_xticklabels(custom_labels)
ax3.set_xlabel("Dimension")
ax3.set_ylabel("Total Cholesky factorizations")
ax3.set_title("Cholesky (MCM) vs dimension")
ax3.grid(True, which="both", linestyle=":", linewidth=0.5)


# ============================================================
# Consolidated legend
# ============================================================

# Create a transparent proxy for the empty slot
proxy_handle = Line2D([], [], color="none", label="")

def get_h_l(key):
    """Retrieve handle/label if exists, else return proxy."""
    if key in legend_handles_map:
        return legend_handles_map[key], legend_labels_map[key]
    return proxy_handle, ""

# Get handles for all 5 methods
h1, l1          = get_h_l((1, "BASE"))
h2_mcm, l2_mcm   = get_h_l((2, "MCMR"))
h3_mcm, l3_mcm   = get_h_l((3, "MCMR"))
h2_glrt, l2_glrt = get_h_l((2, "GLRT"))
h3_glrt, l3_glrt = get_h_l((3, "GLRT"))

# ------------------------------------------------------------------
# CONSTRUCT LIST FOR ROW-MAJOR FILL (ncol=3)
# Desired Visual Layout:
# Col 1: AR1,     Empty
# Col 2: AR2-MCM, AR2-GLRT
# Col 3: AR3-MCM, AR3-GLRT
#
# Matplotlib fills Row 1 first, then Row 2.
# Row 1 must contain: [AR1,   AR2-MCM,  AR3-MCM]
# Row 2 must contain: [Empty, AR2-GLRT, AR3-GLRT]
# ------------------------------------------------------------------

final_handles = [h1, proxy_handle, h2_mcm, h2_glrt, h3_mcm, h3_glrt]
final_labels  = [l1, "",           l2_mcm, l2_glrt, l3_mcm, l3_glrt]

fig.legend(
    final_handles,
    final_labels,
    loc="lower center",
    ncol=3, 
    bbox_to_anchor=(0.5, 0.8),
    frameon=True,
    columnspacing=1.5 # Optional: Adds space between columns for clarity
)

fig.tight_layout(rect=(0, 0, 1, 0.8))

out_pgf = "rosenbrock_scaling/rosenbrock_time_hvp_chol_vs_dim_log2.pgf"
out_png = "rosenbrock_scaling/rosenbrock_time_hvp_chol_vs_dim_log2.png"
fig.savefig(out_pgf)
fig.savefig(out_png, dpi=300)
print(f"Saved {out_pgf!r} and {out_png!r}")