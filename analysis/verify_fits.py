"""Verify saved fits, nested models, and the shared-burst covariance algebra."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.stats import multivariate_normal

from amati_pipeline.data import Dataset, sha256
from amati_pipeline.model import MODES, RelativeCalibration
from run_sample import ROOT, SAMPLE


def check_likelihood(model, theta):
    rs, rh, vg, vh = (a[0] for a in model.components(theta))
    cov = np.diag(vg)
    ns = model.ns
    cov[:ns, :ns] += model.data.sn_cov
    i = ns + np.arange(model.nh)
    cov[i, i] += vh
    i, j = model.shared_s, ns + model.shared_h
    cov[i, j] = cov[j, i] = vg[i]
    check = multivariate_normal.logpdf(np.r_[rs, rh], cov=cov)
    return abs(check - model(theta))


def audit(model, folder, full):
    config = json.loads((folder / "config.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    best = json.loads((folder / "bestfit.json").read_text())
    assert config["parameter_names"] == model.names and config["model"] == model.config
    assert config["input_manifest_sha256"] == sha256(model.data.source / "manifest.json")
    assert summary["convergence_50tau"] and summary["tau_stable_5pct"] and best["optimizer_success"]
    theta = np.array([best["parameters"][name] for name in model.names])
    np.testing.assert_allclose(model(theta), best["lnL"], atol=1.e-9, rtol=0)
    with np.load(folder / "posterior_plot_samples.npz") as samples:
        assert list(samples["names"]) == model.names
        assert np.isfinite(samples["samples"]).all()
    if full:
        with h5py.File(folder / "chain.h5", "r") as file:
            chain = file["mcmc/chain"]
            assert chain.shape == (summary["steps"], summary["walkers"], len(model.names))
            kept = chain[summary["burn"]:].reshape(-1, len(model.names))
            quantiles = np.quantile(kept, [.025, .16, .5, .84, .975], axis=0).T
        for key, q in zip(model.names, quantiles):
            p = summary["parameters"][key]
            np.testing.assert_allclose(q, [p["ci95"][0], p["ci68"][0], p["median"],
                                           p["ci68"][1], p["ci95"][1]], atol=1.e-12, rtol=0)
    error = max(check_likelihood(model, x) for x in (theta, model.initial))
    assert error < 1.e-9
    return {"dense_log_likelihood_error": error, "lnL_max": best["lnL"],
            "min_steps_over_tau": min(summary["postburn_steps_div_tau"].values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-chains", action="store_true")
    args = parser.parse_args()
    result = {}
    data = Dataset.load(ROOT / "inputs" / SAMPLE)
    constant = RelativeCalibration(data, "constant")
    for name in MODES:
        model = RelativeCalibration(data, name)
        result[f"{SAMPLE}/{name}"] = audit(model, ROOT / "runs" / SAMPLE / name, args.full_chains)
        if name != "constant":
            np.testing.assert_allclose(model(np.r_[constant.initial, 0.]), constant(constant.initial),
                                       atol=1.e-12, rtol=0)
    repeat = RelativeCalibration(Dataset.load(ROOT / "inputs/reference"), "linear")
    result["reference/linear_repeat"] = audit(repeat, ROOT / "checks/reference_linear_repeat", args.full_chains)
    print(json.dumps(result, indent=2))

