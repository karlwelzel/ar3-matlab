import os
from itertools import groupby
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from wandb_tools import set_plot_asthetics

set_plot_asthetics()

functions = [
    np.poly1d([3, -10, 12, -5, 0]),
    np.poly1d([3, 0, -1 / 2, -10 / 9, -25 / 144]),  # same function expanded at x=0.8
]

limits_selection = {  # (p, i): [xleft, xright, ybottom, ytop]
    (2, 0): [-0.25, 1.5, -1, 4],
    (2, 1): [-2.8, 1.2, -1.85, 1],
    (3, 0): [-0.6, 3.1, -7, 4],
    (3, 1): [-1.8, 2.2, -2, 3],
}

sigmas_selection = {
    (2, 0): [1, 10, 100, 1000],
    (2, 1): [1, 5, 25, 125],
    (3, 0): [9, 10.25, 11.5, 12.75],
    (3, 1): [1, 10, 100, 1000],
}

xis_selection = {
    (2, 0): [0, 1, 5, 10, 20],
    (2, 1): [0, 1, 10, 100, 1000],
    (3, 0): [0, 0.5, 0.9, 1.3, 1.4],
    (3, 1): [0, 1, 2, 4, 6],
}

for p in [2, 3]:
    for i, fun in enumerate(functions):
        for relaxed in [False, True]:
            taylor = np.poly1d(fun.coefficients[-p - 1 :])
            def model(alpha, sigma):
                return taylor(alpha) + (sigma / (p + 1)) * abs(
                            alpha
                        ) ** (p + 1)

            fig, ax = plt.subplots(figsize=(3.8, 1.8), layout="tight")
            ax.spines["left"].set_position("zero")
            ax.spines["bottom"].set_position("zero")
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.set_xlabel(r"$\alpha$", loc="right")

            limits = limits_selection[(p, i)]
            sigmas = sigmas_selection[(p, i)]
            xis = xis_selection[(p, i)] if relaxed else [0]
            alphas = np.linspace(limits[0], limits[1], 500)

            # f(x + \alpha)
            ax.plot(
                alphas,
                fun(alphas),
                color="k",
                label=r"$f(\alpha)$",
            )

            # t(\alpha)
            ax.plot(
                alphas,
                taylor(alphas),
                color="r",
                linestyle="dashdot",
                label=r"$t(\alpha)$",
            )

            # m_{\sigma}(\alpha)

            for j, sigma in enumerate(sigmas):
                kwargs = {"color": "blue", "linestyle": "dashed"}
                if j == 0:
                    kwargs["label"] = r"$m_{{\sigma}}(\alpha)$"
                ax.plot(alphas, model(alphas, sigma), **kwargs)

            # persistent interval
            for j, xi in enumerate(xis):
                def is_minimizer(alpha):
                    return ((xi - taylor.deriv(1)(alpha)) * np.sign(alpha) >= 0
                                    and (
                                        taylor.deriv(2)(alpha) * alpha
                                        + p * (xi - taylor.deriv(1)(alpha))
                                    )
                                    * np.sign(alpha)
                                    >= 0)
                kwargs_span: dict[str, Any] = {"alpha": 0.16}
                kwargs_line = {"color": "black", "linestyle": "dotted"}
                if j == 0:
                    kwargs_span["label"] = "persistent mins"
                    kwargs_line["label"] = r"$\bar{\alpha}$"

                first_transient = True
                for is_minimizer_group, group in groupby(alphas, key=is_minimizer):
                    group = list(group)
                    if is_minimizer_group and abs(group[0]) < 1e-1:
                        if "label" in kwargs_span:
                            kwargs_span["label"] = "persistent mins"
                        ax.axvspan(group[0], group[-1], color="blue", **kwargs_span)
                        alpha_bar = group[-1]
                    if is_minimizer_group and not abs(group[0]) < 1e-1 and not relaxed:
                        if "label" in kwargs_span:
                            if first_transient:
                                kwargs_span["label"] = "transient mins"
                            else:
                                del kwargs_span["label"]
                        ax.axvspan(group[0], group[-1], color="red", **kwargs_span)
                        first_transient = False

                if p == 2 and i == 0 and not relaxed:
                    # Add legend entry, not visible
                    ax.axvspan(0, 0, color="red", alpha=0.16, label="transient mins")

                if alpha_bar != limits[1]:
                    ax.axvline(alpha_bar, **kwargs_line)

            if p == 2 and i == 0:
                ax.legend()
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])

            os.makedirs("illustration", exist_ok=True)
            filename = (
                f"illustration/AR{p} func{i}" + (" relaxed" if relaxed else "") + ".pgf"
            )
            fig.savefig(filename)
            print(f"Saved {filename}")
