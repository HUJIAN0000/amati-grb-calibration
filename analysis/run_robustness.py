"""Fit and verify the finite-offset and Amati-intercept sensitivity models."""
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from amati_pipeline.data import Dataset, json_write
from amati_pipeline.inference import optimize, run
from amati_pipeline.model import RelativeCalibration
from verify_fits import audit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "robustness"
CASES = {
    "offset_constant": ("constant", True, False),
    "offset_linear": ("linear", True, False),
    "offset_log": ("log", True, False),
    "intercept_linear": ("constant", False, True),
    "joint_linear": ("linear", False, True),
}
OMEGA_M = .30


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def luminosity_distance(z):
    integral = quad(lambda u: 1 / np.sqrt(OMEGA_M * (1 + u)**3 + 1 - OMEGA_M), 0, z)[0]
    return (1 + z) * integral  # In units of c/H0, which cancels in the ratio.


def pair_offsets(model):
    calibrators = np.r_[model.data.sn.calibrator_z, model.data.hii.calibrator_z]
    distance_grb = np.array([luminosity_distance(z) for z in model.z])
    distance_cal = np.array([luminosity_distance(z) for z in calibrators])
    return 5 * np.log10(distance_grb / distance_cal)


class RobustnessCalibration(RelativeCalibration):
    def __init__(self, data, mode, apply_offsets, intercept_drift):
        super().__init__(data, mode)
        self.offsets = pair_offsets(self) if apply_offsets else np.zeros_like(self.z)
        self.intercept_drift = intercept_drift
        if apply_offsets:
            self.config["residual_distance_correction"] = {"model": "flat_lcdm", "omega_m": OMEGA_M}
        if intercept_drift:
            self.config["amati_intercept_drift"] = "shared a1*z at 1 keV"
            self.config["zero_points"] = "A_SN(z) = A_SN + a1*z; A_HII(z) = A_HII + a1*z"
            self.names.append("a1")
            self.initial = np.r_[self.initial, 0.]
            self.bounds = np.vstack([self.bounds, [-2., 2.]])

    def components(self, theta):
        rs, rh, vg, vh = super().components(theta)
        shift = -self.offsets
        if self.intercept_drift:
            a1 = np.atleast_2d(theta)[:, -1:]
            shift = shift + 2.5 * a1 * self.z
        return rs + shift[..., :self.ns], rh + shift[..., self.ns:], vg, vh


def fit_subset(data, ids, starts, mode):
    sn = np.flatnonzero(data.sn.grb_id.isin(ids))
    hii = np.flatnonzero(data.hii.grb_id.isin(ids))
    subset = replace(data, sn=data.sn.iloc[sn], hii=data.hii.iloc[hii],
                     sn_cov=data.sn_cov[np.ix_(sn, sn)])
    model = RelativeCalibration(subset, mode)
    result = optimize(model, [starts[mode][key] for key in model.names])
    if not result.success:
        raise RuntimeError(result.message)
    return dict(zip(model.names, result.x)), 2 * (len(model.names) + result.fun)


def influence(data):
    starts = {mode: read_json(ROOT / f"runs/reference/{mode}/bestfit.json")["parameters"]
              for mode in ("constant", "linear")}
    ids = set(data.sn.grb_id) | set(data.hii.grb_id)
    estimates = {}
    for gid in sorted(ids):
        parameters, _ = fit_subset(data, ids - {gid}, starts, "linear")
        estimates[gid] = parameters["b1"]
    cuts = {}
    for limit in (3., 2.5):
        keep = ids & set(data.grb.loc[data.grb.z <= limit, "grb_id"])
        _, constant_aic = fit_subset(data, keep, starts, "constant")
        parameters, aic = fit_subset(data, keep, starts, "linear")
        cuts[str(limit)] = {"n_grb": len(keep), "b1": parameters["b1"], "delta_aic": aic - constant_aic}
    return {"leave_one_grb_out_b1": estimates, "redshift_cuts": cuts}


def verify(data, full_chains):
    base = RelativeCalibration(data, "constant")
    delta = pair_offsets(base)
    stats = {}
    for family, section in (("sn", slice(None, base.ns)), ("hii", slice(base.ns, None))):
        z, d = base.z[section], delta[section]
        stats[family] = {"mean": d.mean(), "min": d.min(), "max": d.max(),
                         "pearson_r": np.corrcoef(z, d)[0, 1],
                         "slope_mag_per_redshift": np.polyfit(z, d, 1)[0]}
    baseline = read_json(ROOT / "runs/reference/constant/bestfit.json")
    uncorrected_aic = 2 * (len(baseline["parameters"]) - baseline["lnL"])
    checks = {}
    for name, args in CASES.items():
        model = RobustnessCalibration(data, *args)
        folder = OUT / name
        result = audit(model, folder, full_chains)
        cfg = read_json(folder / "config.json")
        np.testing.assert_array_equal(cfg["bounds"], model.bounds)
        summary = read_json(folder / "summary.json")
        if not args[1]:
            nested = RelativeCalibration(data, args[0])
            np.testing.assert_allclose(model(np.r_[nested.initial, 0.]), nested(nested.initial),
                                       rtol=0, atol=1e-12)
        aic = 2 * (len(model.names) - result["lnL_max"])
        with np.load(folder / "posterior_plot_samples.npz") as payload:
            if name == "joint_linear":
                samples = dict(zip(payload["names"], payload["samples"].T))
                result["a1_b1_posterior_pearson_r"] = np.corrcoef(samples["a1"], samples["b1"])[0, 1]
        checks[name] = {**result, "AIC": aic, "parameters": summary["parameters"],
                        "steps": summary["steps"], "tau_stable_5pct": summary["tau_stable_5pct"],
                        "prior_boundary_fraction": summary["prior_boundary_fraction"]}
    for name, result in checks.items():
        aic0 = checks["offset_constant"]["AIC"] if CASES[name][1] else uncorrected_aic
        result["delta_aic"] = result["AIC"] - aic0
    return {"pair_offsets_mag": stats, "full_chain_quantiles_checked": full_chains,
            "fits": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=(*CASES, "verify", "influence"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-chains", action="store_true")
    args = parser.parse_args()
    data = Dataset.load(ROOT / "inputs/reference")
    if args.case == "verify":
        json_write(OUT / "validation.json", verify(data, args.full_chains))
    elif args.case == "influence":
        OUT.mkdir(exist_ok=True)
        json_write(OUT / "influence.json", influence(data))
    else:
        model = RobustnessCalibration(data, *CASES[args.case])
        seed = 2026111500 + list(CASES).index(args.case) + 100
        summary = run(model, OUT / args.case,
                      dict(walkers=72, steps=14000, burn=3000, max_steps=50000,
                           seed=seed, chunk=2000, auto_extend=True), resume=args.resume)
        assert summary["convergence_50tau"] and summary["tau_stable_5pct"]
