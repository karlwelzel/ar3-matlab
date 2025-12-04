from collections import defaultdict
import dataclasses
import enum
import json
import matplotlib.axes
import matplotlib.collections
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pathlib
import pickle
import seaborn as sns
from typing import Any, Callable, Literal
import wandb


class Evals(enum.Enum):
    NONE = "No evaluation"
    FUN = "$f$ evaluated"
    FUN_AND_DER = r"$f, \dots, \nabla^3 f$ evaluated"
    FINAL = "Final iterate"


class ToleranceMeasure(enum.Enum):
    FUNCTION_VALUE = "f"
    GRADIENT_NORM = "g"


class PlotType(enum.Enum):
    TAU_PLOT = enum.auto()
    EPS_PLOT = enum.auto()


@dataclasses.dataclass
class CategorizedRun:
    history: pandas.DataFrame
    method: str
    problem: str
    problem_setup: str


class CategorizedRuns(list[CategorizedRun]):
    def __init__(self, method_sort_key: Callable[[str], Any] | None = None):
        super().__init__()
        self.all_methods = set()
        self.all_problem_setups = set()
        self.all_combinations = set()
        self.method_sort_key = method_sort_key

    def append(self, item: CategorizedRun):
        super().append(item)
        self.all_methods.add(item.method)
        self.all_problem_setups.add(item.problem_setup)
        self.all_combinations.add((item.method, item.problem_setup))

    @property
    def sorted_methods(self):
        return sorted(self.all_methods, key=self.method_sort_key)


def split_config(
    config: dict[str, Any],
    method_parameters: list[str],
    ignore: list[str] = [],
) -> tuple[dict[str, Any], dict[str, Any]]:
    method_config = dict()
    problem_setup_config = dict()
    for key, value in config.items():
        if key in method_parameters:
            method_config[key] = value
        elif key in ignore:
            pass
        else:
            problem_setup_config[key] = value

    return method_config, problem_setup_config


def categorize_runs(
    wandb_runs,
    histories: dict[str, pandas.DataFrame],
    method_parameters: list[str],
    dump_type: str = "gpp",
    ignore: list[str] = [],
    method_sort_key: Callable[[str], Any] | None = None,
    error_on_duplicate=False,
) -> CategorizedRuns:
    categorized_runs = CategorizedRuns(method_sort_key=method_sort_key)
    for run in wandb_runs:
        method_config, problem_setup_config = split_config(
            config=run.config,
            method_parameters=method_parameters,
            ignore=ignore,
        )
        if dump_type == "gpp":
            method, problem_setup = gpp_dumps(method_config, problem_setup_config)
        else:
            method, problem_setup = gcp_dumps(method_config, problem_setup_config)
        categorized_runs.append(
            CategorizedRun(
                history=histories[run.name],
                method=method,
                problem=run.config["problem"],
                problem_setup=problem_setup,
            )
        )

    if error_on_duplicate and len(wandb_runs) != len(categorized_runs.all_combinations):
        raise ValueError("The combination of method and problem setup is not unique")

    categorized_runs.sort(key=lambda run: (str(run.method), str(run.problem)))
    return categorized_runs


def cache_run_histories(groups: list[str]) -> dict[str, pandas.DataFrame]:
    # Prepare cache file for all relevant runs
    cache_file = pathlib.Path(" ".join(groups) + "_cache.bin")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            histories = pickle.load(f)
    else:
        # api = wandb.Api(timeout=10000) # avoiding timeout for the first download
        api = wandb.Api()
        wandb_runs = api.runs(
            path="ar3-project/all_experiments",
            filters={"$or": [{"group": group} for group in groups]},
        )

        histories = dict()
        for j, wandb_run in enumerate(wandb_runs):
            histories[wandb_run.name] = wandb_run.history(samples=10000, pandas=True)
            print(f"{j+1}/{len(wandb_runs)}")

        with open(cache_file, "wb") as f:
            pickle.dump(histories, f)

    return histories


def merge_histories_from_files(groups: list[str]) -> dict[str, Any]:
    histories = dict()
    for group in groups:
        cache_file = pathlib.Path(f"{group}_cache.bin")
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cache file for group '{group}' not found: {cache_file}"
            )
        with open(cache_file, "rb") as f:
            group_histories = pickle.load(f)
            histories.update(group_histories)
    return histories


# functions for performance profiles
def gpp_dumps(method_config, problem_setup_config):
    method, problem_setup = json.dumps(method_config), json.dumps(
        problem_setup_config,
        sort_keys=True,
    )
    return method, problem_setup


def gpp_costs(
    categorized_runs: CategorizedRuns,
    cost_measure: str,
    tolerance_measure: ToleranceMeasure,
    tolerance: float,
) -> pandas.DataFrame:
    if tolerance_measure == ToleranceMeasure.FUNCTION_VALUE:
        f_best = defaultdict(lambda: np.inf)
        f_ini = dict()
        norm_g_ini = dict()
        for run in categorized_runs:
            run_f_best = np.min(run.history["f"])
            f_best[run.problem] = min(run_f_best, f_best[run.problem])
            f_ini[run.problem] = run.history["f"][0]
            norm_g_ini[run.problem] = run.history["norm_g"][0]

    costs = pandas.DataFrame(
        index=categorized_runs.sorted_methods,
        columns=sorted(categorized_runs.all_problem_setups),
        dtype=np.float64,
    )

    for run in categorized_runs:
        if tolerance_measure == ToleranceMeasure.FUNCTION_VALUE:
            problem_f_best = f_best[run.problem]
            # problem_f_ini = f_ini[run.problem]
            (solution_indices,) = np.where(
                run.history["f"]
                <= problem_f_best + tolerance * max(1, abs(problem_f_best))
                # <= problem_f_best + tolerance * (problem_f_ini - problem_f_best)
            )
        elif tolerance_measure == ToleranceMeasure.GRADIENT_NORM:
            (solution_indices,) = np.where(
                run.history["norm_g"] <= tolerance  # * norm_g_ini
            )

        if len(solution_indices) != 0:
            first_index = solution_indices[0]
            costs.loc[run.method, run.problem_setup] = run.history.loc[
                first_index, cost_measure
            ]
        else:
            costs.loc[run.method, run.problem_setup] = np.inf

    if np.isnan(costs).any(axis=None):
        print(costs)
        conflicting_keys = set(json.loads(costs.columns[0]).keys()) ^ set(
            json.loads(costs.columns[-1]).keys()
        )
        raise RuntimeError(
            "A combination of method and problem setup is missing. "
            + "Consider ignoring the following keys: "
            + str(conflicting_keys)
        )

    return costs


def set_plot_asthetics():
    # Construct the path to 'math.tex'
    path_to_settings = pathlib.Path.cwd()

    plt.switch_backend("pgf")
    plt.rc(
        "pgf",
        texsystem="pdflatex",
        rcfonts=False,
        preamble=rf"""
            \usepackage{{import}}
            \import{{\detokenize{{{path_to_settings.as_posix()}}}}}{{math.tex}}
            """,
    )
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Computer Modern Roman")
    plt.rc("lines", markersize=1.5, linewidth=1)
    plt.rc("savefig", dpi=300)

    # Change default sizes, "large" -> "medium" and "medium" -> "small"
    plt.rc("axes", titlesize="medium", labelsize="small")
    plt.rc("xtick", labelsize="small")
    plt.rc("ytick", labelsize="small")
    plt.rc("legend", fontsize="small")
    plt.rc("figure", titlesize="medium")


def gpp_plot_title(
    cost_measure: str,
    tolerance_measure: ToleranceMeasure,
    tolerance: float,
    plot_type: PlotType,
):
    measure_label = {
        "_step": "Subproblem solves",
        "total_solves": "Subproblem solves",
        "total_fun": "Function evaluations",
        "total_der": "Derivative evaluations",
    }[cost_measure]

    if plot_type == PlotType.TAU_PLOT:
        tolerance_exponent = int(round(np.log10(tolerance)))
        return rf"{measure_label} ($\varepsilon_{tolerance_measure.value} = 10^{{{tolerance_exponent}}}$)"
    elif plot_type == PlotType.EPS_PLOT:
        if tolerance == float("inf"):
            tau_str = r"\infty"
        elif tolerance == int(tolerance):
            # Use integer when possible
            tau_str = str(int(tolerance))
        else:
            tau_str = str(tolerance)
        return rf"{measure_label} ($\tau = {tau_str}$)"


def gpp_tau_plot(
    ax: matplotlib.axes.Axes,
    categorized_runs: CategorizedRuns,
    cost_measure: str,
    tolerance_measure: ToleranceMeasure,
    tolerance: float,
    taus,
    new_labels: list[str] | None = None,
) -> matplotlib.axes.Axes:
    pp_curves = pandas.DataFrame(
        index=taus,
        columns=categorized_runs.sorted_methods,
        dtype=np.float64,
    )

    costs = gpp_costs(
        categorized_runs=categorized_runs,
        tolerance_measure=tolerance_measure,
        tolerance=tolerance,
        cost_measure=cost_measure,
    )

    num_problem_setups = len(categorized_runs.all_problem_setups)

    for method in categorized_runs.all_methods:
        for tau in taus:
            count = np.count_nonzero(
                (costs.loc[method, :] <= tau * costs.min(axis=0))
                & (costs.loc[method, :] < np.inf)
            )
            pp_curves.loc[tau, method] = count / num_problem_setups

    graph = sns.lineplot(data=pp_curves, ax=ax)
    graph.set_xlabel(r"$\tau$")
    graph.set_title(
        gpp_plot_title(
            cost_measure=cost_measure,
            tolerance_measure=tolerance_measure,
            tolerance=tolerance,
            plot_type=PlotType.TAU_PLOT,
        )
    )

    graph.set_ylim(0, 1.1)
    graph.set_xlim(min(taus), max(taus))
    graph.legend()

    if new_labels is not None:
        # Customize the legend
        handles, labels = graph.get_legend_handles_labels()
        graph.legend(handles, new_labels)

    return graph


def gpp_eps_plot(
    ax: matplotlib.axes.Axes,
    categorized_runs: CategorizedRuns,
    cost_measure: str,
    tolerance_measure: ToleranceMeasure,
    tolerances,
    tau: float,
    new_labels: list[str] | None = None,
) -> matplotlib.axes.Axes:
    pp_curves = pandas.DataFrame(
        index=tolerances,
        columns=categorized_runs.sorted_methods,
        dtype=np.float64,
    )

    for tolerance in tolerances:
        costs = gpp_costs(
            categorized_runs=categorized_runs,
            tolerance_measure=tolerance_measure,
            tolerance=tolerance,
            cost_measure=cost_measure,
        )

        num_problem_setups = len(categorized_runs.all_problem_setups)

        for method in categorized_runs.all_methods:
            if tau == np.inf:
                count = np.count_nonzero(costs.loc[method, :] < np.inf)
            else:
                count = np.count_nonzero(
                    (costs.loc[method, :] <= tau * costs.min(axis=0))
                    & (costs.loc[method, :] < np.inf)
                )
            pp_curves.loc[tolerance, method] = count / num_problem_setups

    graph = sns.lineplot(data=pp_curves, ax=ax)
    graph.set_xlabel(rf"$\varepsilon_{tolerance_measure.value}$")
    graph.set_title(
        gpp_plot_title(
            cost_measure=cost_measure,
            tolerance_measure=tolerance_measure,
            tolerance=tau,
            plot_type=PlotType.EPS_PLOT,
        )
    )

    graph.set_xscale("log")
    graph.set_xlim(min(tolerances), max(tolerances))
    graph.set_ylim(0, 1.1)
    graph.legend()

    if new_labels is not None:
        # Customize the legend
        handles, labels = graph.get_legend_handles_labels()
        graph.legend(handles, new_labels)

    return graph


def consolidate_legends(
    figure: matplotlib.figure.Figure,
    axs,
    legend_height: float = 0.15,
    ncol: int = 3,  # Default number of columns set to 3
):
    # Move legends from individual subplots to the top of the figure
    height_ratio = legend_height / figure.get_figheight()
    legends = [ax.legend_ for row in axs for ax in row if ax.legend_ is not None]
    if not legends:
        return  # No legends to consolidate

    # Collect handles and labels from the first legend
    handles = legends[0].legend_handles
    labels = [text.get_text() for text in legends[0].get_texts()]

    # Create a consolidated legend with the specified number of columns
    figure.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1 - height_ratio),
        ncol=ncol,  # Use 'ncol' to specify the number of columns
        title=None,
        frameon=True,
    )

    # Remove individual legends from subplots
    for legend in legends:
        legend.remove()

    # Adjust layout to accommodate the new legend
    figure.tight_layout(pad=1, rect=(0, 0, 1, 1 - height_ratio))


def generate_gpp_plots(
    filename_prefix: str,
    categorized_runs: CategorizedRuns,
    new_labels: list[str] | None = None,
    format: str = "png",
    legend_ncols: int = 3,
):
    cost_measures = ["total_fun", "total_der", "total_solves"]
    tolerances = {
        ToleranceMeasure.FUNCTION_VALUE: [1e-3, 1e-8, 1e-13],
        ToleranceMeasure.GRADIENT_NORM: [1e0, 1e-4, 1e-8],
    }
    num_tolerances = 3
    taus = np.linspace(1, 3.1, 100)

    legend_rows = np.ceil(len(new_labels) / legend_ncols)
    legend_height = 0.5 * legend_rows  # inches

    # One line for main part of the paper
    figure, axs = plt.subplots(
        nrows=1,
        ncols=len(cost_measures),
        figsize=(2.5 * len(cost_measures), 2.5 + legend_height),
        squeeze=False,
    )

    tolerance_measure = ToleranceMeasure.FUNCTION_VALUE
    for i, cost_measure in enumerate(cost_measures):
        for j, tolerance in enumerate(tolerances[tolerance_measure][1:2]):
            gpp_tau_plot(
                ax=axs[j][i],
                categorized_runs=categorized_runs,
                cost_measure=cost_measure,
                tolerance_measure=tolerance_measure,
                tolerance=tolerance,
                taus=taus,
                new_labels=new_labels,
            )
    consolidate_legends(
        figure=figure, axs=axs, legend_height=legend_height, ncol=legend_ncols
    )

    filename = f"{filename_prefix} tau {tolerance_measure.value} one line.{format}"
    figure.savefig(filename)
    print(f"Saved {filename!r}")

    with plt.rc_context(
        {
            "font.size": 6,
            "lines.linewidth": 0.75,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    ):
        legend_rows = np.ceil(len(new_labels) / legend_ncols)
        legend_height = 0.3 * legend_rows  # inches

        # Full tau plot for appendix
        for tolerance_measure in ToleranceMeasure:
            figure, axs = plt.subplots(
                nrows=num_tolerances,
                ncols=len(cost_measures),
                figsize=(1.5 * len(cost_measures), 1.3 * num_tolerances + legend_height),
                squeeze=False,
            )

            for i, cost_measure in enumerate(cost_measures):
                for j, tolerance in enumerate(tolerances[tolerance_measure]):
                    axs[j][i].tick_params(axis="y", pad=-1)
                    gpp_tau_plot(
                        ax=axs[j][i],
                        categorized_runs=categorized_runs,
                        cost_measure=cost_measure,
                        tolerance_measure=tolerance_measure,
                        tolerance=tolerance,
                        taus=taus,
                        new_labels=new_labels,
                    )
            consolidate_legends(
                figure=figure, axs=axs, legend_height=legend_height, ncol=legend_ncols
            )

            filename = f"{filename_prefix} tau {tolerance_measure.value}.{format}"
            figure.savefig(filename)
            print(f"Saved {filename!r}")

        tolerances = {
            ToleranceMeasure.FUNCTION_VALUE: np.logspace(-16, 0, 100),
            ToleranceMeasure.GRADIENT_NORM: np.logspace(-8, 0, 100),
        }
        taus = [1, 1.5, np.inf]

        # Full eps plot for appendix
        for tolerance_measure in ToleranceMeasure:
            figure, axs = plt.subplots(
                nrows=len(taus),
                ncols=len(cost_measures),
                figsize=(1.5 * len(cost_measures), 1.3 * len(taus) + legend_height),
                squeeze=False,
            )

            for i, cost_measure in enumerate(cost_measures):
                for j, tau in enumerate(taus):
                    axs[j][i].tick_params(axis="y", pad=-1)
                    gpp_eps_plot(
                        ax=axs[j][i],
                        categorized_runs=categorized_runs,
                        cost_measure=cost_measure,
                        tolerance_measure=tolerance_measure,
                        tolerances=tolerances[tolerance_measure],
                        tau=tau,
                        new_labels=new_labels,
                    )
            consolidate_legends(
                figure=figure, axs=axs, legend_height=legend_height, ncol=legend_ncols
            )

            filename = f"{filename_prefix} eps {tolerance_measure.value}.{format}"
            figure.savefig(filename)
            print(f"Saved {filename!r}")


# functions for convergence dot plots
def gcp_dumps(method_config, problem_setup_config):
    method, problem_setup = json.dumps(list(method_config.values())), json.dumps(
        problem_setup_config,
        sort_keys=True,
    )
    return method, problem_setup


def objective_evaluations(history, i):
    if i + 1 == len(history):
        return Evals.FINAL.value
    elif history.loc[i + 1, "total_fun"] == history.loc[i, "total_fun"]:
        return Evals.NONE.value
    elif history.loc[i + 1, "total_der"] == history.loc[i, "total_der"]:
        return Evals.FUN.value
    else:
        return Evals.FUN_AND_DER.value


def convergence_dot_plot(
    categorized_runs,
    col_titles: list[str] | None = None,
    legend_ncols: int | None = None,
    fig_scale: float = 1,
    legend_loc: Literal["top"] | Literal["left"] = "top",
    **kwargs,
):
    f_best = defaultdict(lambda: np.inf)
    for run in categorized_runs:
        run_f_best = np.min(run.history["f"])
        f_best[run.problem] = min(run_f_best, f_best[run.problem])

    eps = np.finfo(np.float64).eps
    f_best = {problem: value - eps * abs(value) for problem, value in f_best.items()}

    # f_star for 5 (Beale) and 13 (Powell singular) are zero
    f_best["5"] = np.float64(0)
    f_best["13"] = np.float64(0)

    data = pandas.DataFrame(
        data=[
            {
                "sigma": max(float(run.history.loc[i, "sigma"]), 1e-10),
                "norm_g": max(run.history.loc[i, "norm_g"], 1e-25),
                "f": max(run.history.loc[i, "f"] - f_best[run.problem], 1e-25),
                "evals": objective_evaluations(run.history, i),
                "method": run.method,
                "problem": run.problem,
                "type": "data",
            }
            for run in categorized_runs
            for i in range(len(run.history))
        ]
    )

    markers = {
        Evals.NONE.value: "o",
        Evals.FUN.value: "o",
        Evals.FUN_AND_DER.value: "D",
        Evals.FINAL.value: "v",
    }

    default_color = sns.color_palette()[0]
    dark_color = sns.color_palette("dark")[0]

    palette = {
        Evals.NONE.value: (0, 0, 0, 0),
        Evals.FUN.value: default_color,
        Evals.FUN_AND_DER.value: dark_color,
        Evals.FINAL.value: dark_color,
    }

    grid = sns.relplot(
        data=data,
        x="f",
        y="sigma",
        style="evals",
        hue="evals",
        col="method",
        row="problem",
        hue_order=list(palette.keys()),
        markers=markers,
        palette=palette,
        facet_kws={"sharey": "row", "sharex": "row", "margin_titles": True},
        # legend=False,
        **kwargs,
    )
    grid.set(xscale="log", yscale="log")
    grid.set(ylim=(1e-11, 2e8))
    grid.axes[0, 0].set_xlim(1e4, 1e-26)
    grid.axes[1, 0].set_xlim(1e4, 1e-14)
    grid.set_axis_labels(x_var=r"$f - f^*$", y_var=r"$\sigma$")

    # Hack to turn some dots into circles
    for (i, j, _), _ in grid.facet_data():
        for child in grid.axes[i, j]._children:
            if isinstance(child, matplotlib.collections.PathCollection):
                path_collection = child
                break
        else:
            continue
        facecolors = path_collection.get_facecolor()
        edgecolors = np.ones_like(facecolors)
        for i, color in enumerate(facecolors):
            if np.all(color == [0, 0, 0, 0]):
                edgecolors[i] = (*default_color, 1)
            else:
                edgecolors[i] = color
        path_collection.set_edgecolor(edgecolors)  # type: ignore

    # Adapt the legend
    assert grid.legend is not None
    if legend_loc == "top":
        legend_height = 0.4 * (len(Evals) / (legend_ncols or len(Evals)))  # inches
        grid.figure.set_size_inches(
            len(grid.col_names) * 1.5 * fig_scale,
            len(grid.row_names) * 1.7 * fig_scale + legend_height,
        )
        height_ratio = legend_height / grid.figure.get_figheight()
        extra_ratio = 0.05 if col_titles is not None and "\n" in col_titles[0] else 0
        grid.figure.tight_layout(
            pad=0.5, rect=(0, 0, 1, 1 - height_ratio - extra_ratio)
        )
        sns.move_legend(
            grid,
            loc="lower center",
            bbox_to_anchor=(0.5, 1 - height_ratio),
            ncols=legend_ncols or len(Evals),
            title=None,
            frameon=True,
            markerscale=2,
        )
    elif legend_loc == "left":
        legend_width = 2  # inches
        grid.figure.set_size_inches(
            len(grid.col_names) * 1.7 + legend_width,
            len(grid.row_names) * 1.7,
        )
        width_ratio = legend_width / grid.figure.get_figwidth()
        grid.figure.tight_layout(pad=0.5, rect=(width_ratio, 0, 1, 1))
        sns.move_legend(
            grid,
            loc="center right",
            bbox_to_anchor=(width_ratio, 0.5),
            ncols=legend_ncols or len(Evals),
            title=None,
            frameon=True,
            markerscale=2,
        )
    for line in grid.legend.get_lines():
        line.set_markeredgewidth(None)  # Set default edge width
        if line.get_markerfacecolor() == (0, 0, 0, 0):
            line.set_markeredgecolor(default_color)
        else:
            line.set_markeredgecolor(line.get_markerfacecolor())

    grid.set_titles(row_template=r"MGH{row_name}")
    if col_titles is not None:
        for j, col_title in enumerate(col_titles):
            grid.axes[0, j].set_title(col_title)

    return grid
