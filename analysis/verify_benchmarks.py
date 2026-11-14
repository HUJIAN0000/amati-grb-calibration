"""Check fixed-background likelihoods, GRB membership, and saved posterior summaries."""
import argparse
import json

import h5py
import numpy as np
from scipy.stats import norm

from amati_pipeline.data import Dataset, sha256
from make_literature_comparison import SAMPLES, read
from run_benchmark import ROOT, FixedBackground, bursts


def independent_likelihood(grb, theta):
    # Independent 48-point Gauss-Legendre evaluation of the distance integral.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    z = grb.z.to_numpy()
    u = z[:, None] * (nodes + 1.) * .5
    integral = .5 * z * (1. / np.sqrt(.3 * (1 + u)**3 + .7) @ weights)
    dl_cm = (1 + z) * integral * (299792.458 / 70.) * 3.0856775814913673e24
    a, b, scatter = theta
    x = np.log10(grb.ep_keV)
    y = np.log10(4. * np.pi * dl_cm**2 * grb.fluence / (1. + z))
    sx = grb.ep_err_keV / grb.ep_keV / np.log(10.)
    sy = grb.fluence_err / grb.fluence / np.log(10.)
    sigma_dex = np.sqrt((scatter / 2.5)**2 + sy**2 + (b * sx)**2)
    # Convert the density on log10 E_iso to the density on distance modulus.
    return np.sum(norm.logpdf(y, loc=a + b * x, scale=sigma_dex) - np.log(2.5))


def verify_chain(folder, model, summary):
    with h5py.File(folder / "chain.h5", "r") as file:
        chain = file["mcmc/chain"]
        assert chain.shape == (summary["steps"], summary["walkers"], len(model.names))
        kept = chain[summary["burn"]:].reshape(-1, len(model.names))
    assert len(kept) == summary["n_samples"]
    quantiles = np.quantile(kept, [.025, .16, .5, .84, .975], axis=0).T
    for key, q in zip(model.names, quantiles):
        p = summary["parameters"][key]
        np.testing.assert_allclose(q, [p["ci95"][0], p["ci68"][0], p["median"],
                                       p["ci68"][1], p["ci95"][1]], rtol=0, atol=1.e-12)


def verify(full_chains):
    parent_file = ROOT / "inputs/a118_full.csv"
    report = {"a118_csv_sha256": sha256(parent_file), "full_chain_quantiles_checked": full_chains}
    for sample, _, size in SAMPLES:
        grb = bursts(sample)
        assert len(grb) == size and grb.index.is_unique and (grb > 0).all().all()
        if sample != "a118":
            data = Dataset.load(ROOT / "inputs" / sample)
            members = sorted(set(data.sn.grb_id) | set(data.hii.grb_id))
            paired = data.grb.set_index("grb_id").loc[members]
            assert grb.index.equals(paired.index)
            np.testing.assert_allclose(grb.values, paired[grb.columns].values, rtol=1.e-12, atol=0)

        model = FixedBackground(sample)
        folder = ROOT / "benchmarks" / sample
        cfg, best, post = (read(folder / f"{name}.json") for name in ("config", "bestfit", "summary"))
        assert cfg["model"] == model.config and cfg["parameter_names"] == model.names
        assert post["convergence_50tau"] and post["tau_stable_5pct"] and best["optimizer_success"]
        theta = np.array([best["parameters"][key] for key in model.names])
        np.testing.assert_allclose(model(theta), best["lnL"], rtol=0, atol=1.e-9)
        error = max(abs(model(t) - independent_likelihood(grb, t)) for t in (theta, model.initial))
        assert error < 1.e-9
        with np.load(folder / "posterior_plot_samples.npz") as plot:
            assert list(plot["names"]) == model.names and np.isfinite(plot["samples"]).all()
        if full_chains:
            verify_chain(folder, model, post)
        report[sample] = {"n_grb": size, "log_likelihood_error": error,
                          "min_steps_over_tau": min(post["postburn_steps_div_tau"].values()),
                          "slope": post["parameters"]["b"]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-chains", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.full_chains), indent=2))
