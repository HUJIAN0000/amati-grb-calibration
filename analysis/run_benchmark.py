"""Constant Amati slope benchmarks with the A118 equation (9) convention."""
import argparse
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.integrate import quad
from types import SimpleNamespace

from amati_pipeline.data import Dataset, LN10
from amati_pipeline.inference import run
from run_sample import ROOT

SUBSETS = ("a118", "reference")


def bursts(sample):
    parent = pd.read_csv(ROOT / "inputs/a118_full.csv", dtype={"grb_id": str}).set_index("grb_id")
    if sample == "a118":
        return parent.sort_index()
    data = Dataset.load(ROOT / "inputs" / sample)
    matched = set(data.sn.grb_id) | set(data.hii.grb_id)
    return parent.loc[sorted(matched)]


class FixedBackground:
    def __init__(self, sample):
        grb = bursts(sample)
        self.data = SimpleNamespace(source=ROOT / "inputs" / sample)
        self.config = {"sample": sample, "n_grb": len(grb), "H0": 70., "Omega_m": .3,
                       "equation": "Khadka et al. 2021, Eq. (9)"}
        self.names = ["a_GRB", "b", "sigma_GRB"]
        self.bounds = np.array([(0., 300.), (-3., 5.), (1e-5, 5.)])
        self.initial = np.array([50., 1.2, 1.])
        z = grb.z.to_numpy()

        def inverse_expansion(u):
            return 1. / np.sqrt(.3 * (1 + u)**3 + .7)

        distance = (1 + z) * (299792.458 / 70.) * 3.0856775814913673e24
        distance *= np.array([quad(inverse_expansion, 0., zi)[0] for zi in z])
        self.x = np.log10(grb.ep_keV.to_numpy())
        self.y = np.log10(4 * np.pi * distance**2 * grb.fluence.to_numpy() / (1 + z))
        self.sx = (grb.ep_err_keV / grb.ep_keV).to_numpy() / LN10
        self.sy = (grb.fluence_err / grb.fluence).to_numpy() / LN10

    def log_likelihood(self, parameters, return_parts=False):
        a, b, scatter = np.atleast_2d(parameters).T[:, :, None]
        residual = 2.5 * (a + b * self.x - self.y)
        variance = 6.25 * (self.sy**2 + b**2 * self.sx**2) + scatter**2
        ll = -.5 * np.sum(residual**2 / variance + np.log(2 * np.pi * variance), axis=1)
        return ll[:, None] if return_parts else ll

    def __call__(self, parameters):
        t = np.atleast_2d(parameters)
        mask = np.isfinite(t).all(axis=1)
        mask &= ((t > self.bounds[:, 0]) & (t < self.bounds[:, 1])).all(axis=1)
        value = np.full(len(t), -np.inf)
        value[mask] = self.log_likelihood(t[mask])
        return float(value[0]) if np.ndim(parameters) == 1 else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", choices=SUBSETS)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "work" / "benchmarks")
    args = parser.parse_args()
    model = FixedBackground(args.sample)
    run(model, args.output / args.sample,
        dict(walkers=96, steps=6000, burn=1500, max_steps=20000, chunk=1000,
             auto_extend=True, seed=2026111400 + 60 + SUBSETS.index(args.sample)))

