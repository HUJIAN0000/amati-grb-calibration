"""Regenerate the robustness table and the joint intercept-slope posterior plot."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots
import numpy as np

from run_robustness import OUT, read_json

COLUMNS = (
    r"\shortstack{Offset\\constant}", r"\shortstack{Offset\\linear}",
    r"\shortstack{Offset\\logarithmic}", r"\shortstack{Intercept\\drift}",
    r"\shortstack{Joint slope\\and intercept}",
)
ROWS = {
    "b": r"b", "b0": r"b_0", "b1": r"b_1", "a1": r"a_1",
    "A_SN": r"A_{\rm SN}", "A_HII": r"A_{\rm H}", "beta_HII": r"\beta_{\rm H}",
    "sigma_GRB": r"\sigma_{\rm G}\;({\rm mag})", "sigma_HII": r"\sigma_{\rm H}\;({\rm mag})",
}
ROW_END = r" \\"


def interval(value):
    return "$" + f"{value['median']:.3f}^{{+{value['plus']:.3f}}}_{{-{value['minus']:.3f}}}" + "$"


def limits(value):
    lo, hi = value["ci95"]
    return f"$[{lo:.3f},\\,{hi:.3f}]$"


def table(report):
    fits = report["fits"].values()
    data = [fit["parameters"] for fit in fits]
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             "Parameter & " + " & ".join(COLUMNS) + ROW_END, r"\midrule"]
    for name, label in ROWS.items():
        values = [interval(fit[name]) if name in fit else "--" for fit in data]
        lines.append("$" + label + "$ & " + " & ".join(values) + ROW_END)
    lines.append(r"\midrule")
    for name in ("b1", "a1"):
        values = [limits(fit[name]) if name in fit else "--" for fit in data]
        lines.append("$" + ROWS[name] + "$ (95\\%) & " + " & ".join(values) + ROW_END)
    deltas = [fit["delta_aic"] for fit in fits]
    aic = ["$" + f"{delta:+.2f}" + "$" if delta else "$0$" for delta in deltas]
    lines.append(r"$\Delta{\rm AIC}$ & " + " & ".join(aic) + ROW_END)
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def contour(path):
    with np.load(OUT / "joint_linear" / "posterior_plot_samples.npz") as data:
        columns = [data["names"].tolist().index(name) for name in ("b1", "a1")]
        values = data["samples"][:, columns]
    sample = MCSamples(samples=values, names=["b1", "a1"], labels=["b_1", "a_1"],
                       settings={"contours": [.68, .95]})
    plotter = plots.get_single_plotter(width_inch=3.5, ratio=.72)
    plotter.plot_2d(sample, "b1", "a1", filled=True, colors=["#175883"])
    plotter.subplots[0, 0].axvline(0, color="0.5", lw=.6, ls=":")
    plotter.subplots[0, 0].axhline(0, color="0.5", lw=.6, ls=":")
    plotter.export(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    args.table.write_text(table(read_json(OUT / "validation.json")), encoding="utf-8")
    contour(args.figure)
