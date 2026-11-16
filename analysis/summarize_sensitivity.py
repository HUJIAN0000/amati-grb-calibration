"""Verify and tabulate the robustness fits."""
import argparse
from pathlib import Path
import json
import numpy as np

from amati_pipeline.data import json_write
from amati_pipeline.model import MODES
from make_literature_comparison import interval
from run_sensitivity import CASES, ROOT, make_model
from verify_fits import audit

LABELS = (
    "Baseline MILP",
    r"Fixed \(\beta_{\rm H}=5.022\)",
    "SN Ia links only",
    r"\(3\%\) distance window",
    r"\(\Omega_m=0.25\) in matching",
    r"\(\Omega_m=0.35\) in matching",
)


def links(data):
    return {"sn": dict(zip(data.sn.grb_id, data.sn.parent_row)),
            "hii": dict(zip(data.hii.grb_id, data.hii.calibrator_id))}


def reduce_posteriors(full_chains):
    report = {}
    ref = make_model("reference", "constant").data
    ref_links = links(ref)
    for case, label in zip(("reference", *CASES), LABELS):
        data = make_model(case, "constant").data
        pair_links = links(data)
        grbs = set(pair_links["sn"]) | set(pair_links["hii"])
        selected = data.grb[data.grb.grb_id.isin(grbs)]
        report[case] = {
            "label": label, "counts": [len(data.sn), len(data.hii), len(grbs)],
            "z_range": [selected.z.min(), selected.z.max()],
            "common_links_with_baseline": {
                tracer: len(ref_links[tracer].items() & pair_links[tracer].items())
                for tracer in pair_links
            },
        }
        fits = {}
        for mode in MODES:
            if case == "reference":
                folder = ROOT / "runs" / "reference" / mode
                model = make_model(case, mode)
                model.config.pop("sensitivity_case")
                model.config.pop("use_hii_links")
            else:
                folder = ROOT / "sensitivity_runs" / case / mode
                model = make_model(case, mode)
            original = getattr(model, "model", model)
            expected = original.initial if original is model else model.expand(model.initial)
            np.testing.assert_allclose(model(model.initial), original(expected), rtol=0, atol=1e-12)
            diagnostics = audit(model, folder, full_chains)
            saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            fits[mode] = {"parameters": saved["parameters"], "lnL": diagnostics["lnL_max"],
                          "AIC": 2 * (len(model.names) - diagnostics["lnL_max"]),
                          "min_steps_over_tau": diagnostics["min_steps_over_tau"],
                          "normalized_covariance_check_error": diagnostics["dense_log_likelihood_error"]}
        for fit in fits.values():
            fit["delta_aic"] = fit["AIC"] - fits["constant"]["AIC"]
        report[case]["fits"] = fits
    return report


def latex(report):
    rows = [
        r"\begin{tabular}{@{}lrrrccc@{}}",
        r"\toprule",
        r"Check & SN & \hii & GRBs & Constant \(b\) & Linear \(b_1\) & Logarithmic \(b_1\) \\",
        r"\midrule",
    ]
    for entry in report.values():
        fits = entry["fits"]
        shown = [interval(fits[mode]["parameters"][key])
                 for mode, key in zip(MODES, ("b", "b1", "b1"))]
        rows.append(" & ".join([entry["label"], *map(str, entry["counts"]),
                                *("$" + value + "$" for value in shown)]) + r" \\")
    return "\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-chains", action="store_true")
    args = parser.parse_args()
    report = reduce_posteriors(args.full_chains)
    json_write(ROOT / "sensitivity_validation.json",
               {"full_chain_quantiles_checked": args.full_chains, "checks": report})
    (ROOT.parent / "latex/sensitivity_table.tex").write_text(latex(report), encoding="utf-8")
    for key, entry in report.items():
        print(key, entry["counts"], entry["common_links_with_baseline"], entry["z_range"])
        for mode, fit in entry["fits"].items():
            value = fit["parameters"]["b" if mode == "constant" else "b1"]
            print(mode, interval(value), value["ci95"], round(fit["delta_aic"], 2), fit["lnL"])
