"""Build submission tables and figures from the bundled posterior draws."""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cosmo_tools as ct
from amati_pipeline.model import MODES

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "latex"
FIG = PAPER / "figures"
SAMPLE = "reference"
MODELS = MODES
COLOR = "#175883"
PARAM = {
    "A_SN": r"A_{\rm SN}", "A_HII": r"A_{\rm H}", "b": "b", "b0": "b_0",
    "b1": "b_1", "beta_HII": r"\beta_{\rm H}",
    "sigma_GRB": r"\sigma_{\rm G}", "sigma_HII": r"\sigma_{\rm H}",
}
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def catalog(sample, table):
    return pd.read_csv(ROOT / "inputs" / sample / f"{table}.csv", dtype={"grb_id": str})


def selected_grbs(sample):
    used = set(catalog(sample, "sn").grb_id) | set(catalog(sample, "hii").grb_id)
    grbs = catalog(sample, "grb")
    selected = grbs[grbs.grb_id.isin(used)]
    assert len(selected) == len(used)
    return selected


def draws(sample, model):
    path = ROOT / "runs" / sample / model / "posterior_plot_samples.npz"
    with np.load(path) as payload:
        return dict(zip(payload["names"], payload["samples"].T))


def show(p):
    return f"{p['median']:.3f}^{{+{p['plus']:.3f}}}_{{-{p['minus']:.3f}}}"


def sample_plot(parent, grbs):
    fig, axs = plt.subplots(1, 3, figsize=(8.8, 2.7), layout="constrained",
                            gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for frame, color, label in zip((parent, grbs), ("#75767b", COLOR),
                                   ("Full A118", "83 matched GRBs")):
        axs[0].hist(frame.z, bins=np.arange(0, 9.1, .6), histtype="step",
                    density=True, lw=1.55, color=color, label=label)
        axs[1].hist(np.log10(frame.ep_keV), bins=np.arange(.8, 4.8, .23),
                    histtype="step", density=True, lw=1.55, color=color)
    axs[0].set(xlabel=r"GRB redshift $z$", ylabel="Probability density")
    axs[1].set(xlabel=r"$\log_{10}(E_p/{\rm keV})$")
    axs[0].legend(frameon=False, fontsize=8)
    for table, marker, color, label in (("sn", "o", COLOR, "SN Ia"),
                                        ("hii", "^", "#ad481e", "H II")):
        pairs = catalog(SAMPLE, table).merge(grbs[["grb_id", "z"]], on="grb_id")
        axs[2].scatter(pairs.z, pairs.calibrator_z - pairs.z, marker=marker,
                       s=14, color=color, alpha=.76, linewidths=0, label=label)
    axs[2].axhline(0, color="0.65", lw=.6, zorder=0)
    axs[2].set(xlabel="GRB redshift", ylabel=r"$z_{\rm cal}-z_{\rm GRB}$")
    axs[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(FIG / "sample_selection.pdf")
    plt.close(fig)


def evolution_plot(grbs):
    fig, axs = plt.subplots(1, 2, figsize=(7.9, 2.8), layout="constrained", sharex=True, sharey=True)
    z = np.linspace(grbs.z.min(), grbs.z.max(), 130)
    baseline = np.median(draws(SAMPLE, "constant")["b"])
    for ax, model, title in zip(axs, ("linear", "log"), (r"$b_0+b_1z$", r"$b_0+b_1\ln(1+z)$")):
        chain = draws(SAMPLE, model)
        basis = z if model == "linear" else np.log1p(z)
        slope = chain["b0"][:, None] + chain["b1"][:, None] * basis
        lo, med, hi = np.quantile(slope, [.16, .5, .84], axis=0)
        ax.fill_between(z, lo, hi, color=COLOR, alpha=.22, lw=0)
        ax.plot(z, med, color=COLOR, lw=1.45)
        ax.axhline(baseline, color="0.30", ls=":", lw=1, label="Constant slope")
        ax.set(title=title, xlabel="Redshift")
    axs[0].set(ylabel=r"Amati slope $b(z)$")
    axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "matching_comparison.pdf")
    plt.close(fig)


def triangle(model, columns, output):
    folder = ROOT / "runs" / SAMPLE / model
    config = read(folder / "config.json")
    bounds = dict(zip(config["parameter_names"], config["bounds"]))
    ranges = {f"p{i}": bounds[key] for i, key in enumerate(columns)}
    fit = draws(SAMPLE, model)
    values = np.column_stack([fit[key] for key in columns])
    original_samples = ct.MCSamples

    def bounded_samples(*args, settings=None, **kwargs):
        return original_samples(*args, ranges=ranges,
                                settings={**(settings or {}), "contours": [.68, .95]}, **kwargs)

    with patch.object(ct, "MCSamples", bounded_samples):
        ct.plot_getdist_comparison([values], [PARAM[key] for key in columns],
                                   ["83 matched GRBs"], [COLOR], filename=str(FIG / output))
    plt.close("all")


def fitted_table(sample):
    columns = ("b", "b0", "b1", "A_SN", "A_HII", "beta_HII", "sigma_GRB", "sigma_HII")
    summaries = [read(ROOT / "runs" / sample / model / "summary.json")["parameters"]
                 for model in MODELS]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Parameter & Constant & Linear in redshift & Logarithmic \\",
             r"\midrule"]
    for key in columns:
        label = "$" + PARAM[key] + "$"
        if key.startswith("sigma"):
            label += " [mag]"
        values = ["$" + show(s[key]) + "$" if key in s else "--" for s in summaries]
        lines.append(" & ".join([label, *values]) + r" \\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_and_checks():
    rows = []
    report = {}
    metrics = {}
    for model, label in zip(MODELS, ("Constant", "Linear", "Logarithmic")):
        root = ROOT / "runs" / SAMPLE / model
        summary = read(root / "summary.json")
        best = read(root / "bestfit.json")
        assert summary["convergence_50tau"] and summary["tau_stable_5pct"]
        assert best["optimizer_success"]
        params = summary["parameters"]
        aic = 2 * len(params) - 2 * best["lnL"]
        if model == "constant":
            aic0 = aic
        delta = aic - aic0
        vals = ["$" + show(params[key]) + "$" if key in params else "--"
                for key in ("b", "b0", "b1")]
        b1 = params.get("b1")
        ci = "$[" + ",\\,".join(f"{x:.4f}" for x in b1["ci95"]) + "]$" if b1 else "--"
        rows.append(" & ".join([label,
                                *vals, ci, "$" + f"{delta:+.2f}" + "$" if b1 else "0"]) + r" \\")
        metrics[model] = {"parameters": params, "delta_aic": delta, "lnL": best["lnL"],
                          "min_steps_over_tau": min(summary["postburn_steps_div_tau"].values()),
                          "prior_boundary_fraction": summary["prior_boundary_fraction"]}
    report[SAMPLE] = metrics
    (PAPER / f"full_{SAMPLE}.tex").write_text("\n".join(fitted_table(SAMPLE)) + "\n",
                                              encoding="ascii")
    (ROOT / "submission_metrics.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="ascii")
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Model & \(b\) & \(b_0\) & \(b_1\) & \(b_1\) (95\%) & \(\Delta{\rm AIC}\)\\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (PAPER / "relative_table.tex").write_text("\n".join(lines) + "\n", encoding="ascii")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    table_and_checks()
    if not args.tables_only:
        FIG.mkdir(parents=True, exist_ok=True)
        grbs = selected_grbs(SAMPLE)
        parent = pd.read_csv(ROOT / "inputs" / "a118_full.csv")
        sample_plot(parent, grbs)
        evolution_plot(grbs)
        for model in MODELS:
            columns = ["b"] if model == "constant" else ["b0", "b1"]
            columns += ["A_SN", "A_HII", "beta_HII", "sigma_GRB", "sigma_HII"]
            triangle(model, columns, f"triangle_{model}.pdf")

