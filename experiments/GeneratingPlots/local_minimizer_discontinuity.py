import matplotlib.pyplot
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Iterable
from itertools import pairwise
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D, TransformedPath

from wandb_tools import set_plot_asthetics


class SegmentBboxTransform(Affine2D):
    def __init__(self, data_point1, data_point2, data_transform, width=200):
        super().__init__()
        self.data_point1 = data_point1
        self.data_point2 = data_point2
        self.data_transform = data_transform
        self.width = width
        self._invalid = 1

    def get_matrix(self):
        if self._invalid:
            point1 = self.data_transform.transform(self.data_point1)
            point2 = self.data_transform.transform(self.data_point2)
            inner_transform = (
                Affine2D()
                .translate(-0.5, -0.5)
                .scale(np.linalg.norm(point2 - point1), 200)
                .rotate(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
                .translate((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
            )
            self._mtx = inner_transform.get_matrix()
        return self._mtx


x = np.poly1d([1.0, 0])

objective = np.poly1d([3, 2, 0, 1, 0])
expansion_point = -1.0


def compute_rho(step, model):
    actual_decrease = objective(expansion_point) - objective(expansion_point + step)
    predicted_decrease = model(0) - model(step)
    if predicted_decrease < 0:
        print("Negative predicted decrease")
    return actual_decrease / predicted_decrease


def ar1_minimizers(taylor1: np.poly1d, sigmas: Iterable) -> list[dict[str, np.number]]:
    result = []
    for sigma in sigmas:
        model = taylor1 + x**2 * (sigma / 2)
        sp = model.deriv().roots[0]
        result.append(
            {
                "sigma": sigma,
                "step": sp.real,
                "rho": compute_rho(sp.real, taylor1),
                "global": True,
                "branch": 0,
            }
        )
    return result


def ar2_minimizers(taylor2: np.poly1d, sigmas: Iterable) -> list[dict[str, np.number]]:
    result = []
    for sigma in sigmas:
        for direction in [-1, 1]:
            model = np.poly1d(taylor2.coeffs * [1, direction, 1]) + x**3 * (sigma / 3)
            sp = max(model.deriv().roots, key=lambda num: num.real)
            if sp.imag == 0 and sp.real >= 0:
                result.append(
                    {
                        "sigma": sigma,
                        "step": direction * sp.real,
                        "rho": compute_rho(
                            sp.real, np.poly1d(taylor2.coeffs * [1, direction, 1])
                        ),
                        "global": True if direction == 1 else False,
                        "branch": {1: 0, -1: 1}[direction],
                    }
                )
    return result


def ar3_minimizers(taylor3: np.poly1d, sigmas: Iterable) -> list[dict[str, np.number]]:
    result = []
    for sigma in sigmas:
        model = taylor3 + x**4 * (sigma / 4)
        stationary_points = sorted(model.deriv().roots, key=lambda num: abs(num))
        global_min = min(
            [model(sp.real) for sp in stationary_points[::2] if sp.imag == 0]
        )
        for i, sp in enumerate(stationary_points[::2]):
            if sp.imag == 0:
                result.append(
                    {
                        "sigma": sigma,
                        "step": sp.real,
                        "rho": compute_rho(sp.real, taylor3),
                        "global": model(sp) == global_min,
                        "branch": i,
                    }
                )
    return result


taylor1 = x * objective.deriv(1)(expansion_point) + objective(expansion_point)
taylor2 = taylor1 + x**2 * (1 / 2) * objective.deriv(2)(expansion_point)
taylor3 = taylor2 + x**3 * (1 / 6) * objective.deriv(3)(expansion_point)

sigmas = np.logspace(-2, 4, 1000)

ar1_data = ar1_minimizers(taylor1=taylor1, sigmas=sigmas)
ar2_data = ar2_minimizers(taylor2=taylor2, sigmas=sigmas)
ar3_data = ar3_minimizers(taylor3=taylor3, sigmas=sigmas)

all_data = pd.DataFrame(
    [point | {"p": 1} for point in ar1_data]
    + [point | {"p": 2} for point in ar2_data]
    + [point | {"p": 3} for point in ar3_data]
)
all_data.loc[:, "step_norm"] = all_data.loc[:, "step"].abs()
all_data.loc[:, "clipped_rho"] = all_data.loc[:, "rho"].clip(-1, 1)
# print(all_data.to_string())

print("calculations done")

set_plot_asthetics()

# There is a bug in the way PGF does clipping of filled paths
matplotlib.pyplot.switch_backend("pdf")

palette = sns.color_palette("coolwarm", as_cmap=True)

grid = sns.relplot(
    data=all_data,
    x="sigma",
    y="step_norm",
    hue="branch",
    col="p",
    kind="line",
    legend=False,
)

grid.set_xlabels(r"$\sigma$")
grid.set_ylabels(r"$\|s\|$")

for p, ax in grid.axes_dict.items():
    ax.set_title(f"$p = {p}$")

print("seaborn done")

rho_normalize = Normalize()
rho_normalize.autoscale(all_data.loc[:, "clipped_rho"])

# Hack to draw a colored line
for (i, j, _), facet_data in grid.facet_data():
    ax: Axes = grid.axes[i, j]
    ax._children.clear()
    for branch in [0, 1]:
        branch_data = facet_data.query("branch == @branch")
        xs = branch_data.loc[:, "sigma"].to_numpy().reshape(-1, 1)
        ys = branch_data.loc[:, "step_norm"].to_numpy().reshape(-1, 1)
        for (_, row), (_, next_row) in pairwise(branch_data.iterrows()):
            # For every two consecutive data points draw a line with a
            # different color. Then use clip_path to only show the relevant
            # segment of that line. This way dashed linestyles can be used.
            data_point1 = np.array([row["sigma"], row["step_norm"]])
            data_point2 = np.array([next_row["sigma"], next_row["step_norm"]])
            clip_path = TransformedPath(
                Path.unit_rectangle(),
                SegmentBboxTransform(data_point1, data_point2, ax.transData),
            )
            line = Line2D(
                xdata=xs,
                ydata=ys,
                color=palette(rho_normalize(row["clipped_rho"])),
                linewidth=3 if row["global"] else 1,
                linestyle=(0, (2, 0.5)) if row["clipped_rho"] < 0 else "-",
            )
            ax.add_line(line)
            line.set_clip_path(clip_path)
    ax.set_xlim(left=10 ** (-1.2), right=10**3.2)

print("hack done")

colorbar = grid.figure.colorbar(
    ScalarMappable(norm=rho_normalize, cmap=palette),
    label=r"$\rho$",
    ax=grid.axes.ravel().tolist(),
)

grid.set(xscale="log", yscale="log")
grid.figure.set_figwidth(6)
grid.figure.set_figheight(1.8)
# grid.savefig("local_minimizer_discontinuity.png", dpi=300)
# grid.savefig("local_minimizer_discontinuity.pgf")
grid.savefig("local_minimizer_discontinuity.pdf")

print("saving done")
