"""Sensitivity of the Amati slope to the H II calibration and the MILP assignment."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

from amati_pipeline.data import Dataset, json_write, sha256
from amati_pipeline.inference import run
from amati_pipeline.model import MODES, RelativeCalibration
from run_sample import ROOT

CASES = ("fixed_hii", "sn_only", "window03", "om025", "om035")
MATCH_SETTINGS = {"window03": (.03, .30), "om025": (.05, .25), "om035": (.05, .35)}
FIXED_BETA = 5.022


class RestrictedCalibration:
    """Evaluate the existing likelihood on its free coordinates."""
    def __init__(self, model, fixed):
        self.model = model
        self.data = model.data
        self.config = {**model.config, "fixed_parameters": fixed}
        free = np.array([name not in fixed for name in model.names])
        self.names = [name for name in model.names if name not in fixed]
        self.initial = model.initial[free]
        self.bounds = model.bounds[free]
        self.free = free
        self.template = np.array([fixed.get(name, start)
                                  for name, start in zip(model.names, model.initial)])

    def expand(self, theta):
        values = np.asarray(theta)
        expanded = np.broadcast_to(self.template, values.shape[:-1] + self.template.shape).copy()
        expanded[..., self.free] = values
        return expanded

    def __call__(self, theta):
        return self.model(self.expand(theta))

    def components(self, theta):
        return self.model.components(self.expand(theta))

    @property
    def ns(self):
        return self.model.ns

    @property
    def nh(self):
        return self.model.nh

    @property
    def shared_s(self):
        return self.model.shared_s

    @property
    def shared_h(self):
        return self.model.shared_h

    def log_likelihood(self, theta, return_parts=False):
        return self.model.log_likelihood(self.expand(theta), return_parts)


def match(case, pantheon_table, pantheon_covariance):
    import CosmoMatcher as matcher
    tolerance, omega_m = MATCH_SETTINGS[case]
    grb_path = ROOT / "inputs/a118_full.csv"
    hii_path = ROOT / "inputs/hii_full_190.csv"
    grb = pd.read_csv(grb_path, dtype={"grb_id": str})
    hii = pd.read_csv(hii_path, dtype={"calibrator_id": str})
    sn = pd.read_csv(pantheon_table, sep=r"\s+", dtype={"CID": str})
    covariance = matcher.load_matrix(pantheon_covariance, expected_n=len(sn))

    def solve(target, tracer, ids):
        problem = matcher.build_problem(target, tracer, groups=ids, tol=tolerance,
                                        tol_type="dd", omega_m=omega_m)
        links, cert = matcher.solve_problem(problem, log_func=lambda _: None)
        if not cert["optimal"]:
            raise RuntimeError(f"Uncertified assignment for {case}")
        return links[np.argsort(links[:, 0], kind="stable"), :2].T

    g, s = solve(grb.z, sn.zHD, sn.CID)
    h, j = solve(hii.z, grb.z, grb.grb_id)
    sp = sn.iloc[s][["CID", "zHD", "m_b_corr"]].rename(
        columns={"CID": "calibrator_id", "zHD": "calibrator_z"}).copy()
    sp["parent_row"] = s
    sp["grb_id"] = grb.grb_id.iloc[g].to_numpy()
    hp = hii.iloc[h].rename(columns={"z": "calibrator_z"}).copy()
    hp["grb_id"] = grb.grb_id.iloc[j].to_numpy()
    anchors = Dataset.load(ROOT / "inputs/reference").anchors
    folder = ROOT / "sensitivity_inputs" / case
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in (("grb", grb), ("sn", sp), ("hii", hp), ("anchors", anchors)):
        frame.to_csv(folder / f"{name}.csv", index=False)
    np.save(folder / "sn_cov.npy", covariance[np.ix_(s, s)])
    json_write(folder / "manifest.json", {
        "algorithm": "milp", "version": matcher.__version__, "tolerance": tolerance,
        "omega_m": omega_m, "criterion": "dd", "hii_direction": "hii_target",
        "parent_hashes": {path.name: sha256(path)
                          for path in (grb_path, hii_path, pantheon_table, pantheon_covariance)},
        "files": {name: sha256(folder / name)
                  for name in ("grb.csv", "sn.csv", "hii.csv", "anchors.csv", "sn_cov.npy")},
    })


def make_model(case, mode):
    source = ROOT / ("sensitivity_inputs/" + case if case in MATCH_SETTINGS else "inputs/reference")
    data = Dataset.load(source)
    fixed = {}
    if case == "sn_only":
        data = replace(data, hii=data.hii.iloc[:0].copy())
        fixed = {"A_HII": 11., "beta_HII": FIXED_BETA, "sigma_HII": .6}
    elif case == "fixed_hii":
        fixed = {"beta_HII": FIXED_BETA}
    base = RelativeCalibration(data, mode)
    base.config.update({"sensitivity_case": case, "use_hii_links": case != "sn_only"})
    return RestrictedCalibration(base, fixed) if fixed else base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rebuild-pairs", type=Path, nargs=2, metavar=("TABLE", "COVARIANCE"))
    args = parser.parse_args()
    if args.rebuild_pairs:
        match(args.case, *args.rebuild_pairs)
    for mode in args.modes:
        model = make_model(args.case, mode)
        seed = 2026111500 + 10 * CASES.index(args.case) + MODES.index(mode)
        run(model, ROOT / "sensitivity_runs" / args.case / mode,
            dict(walkers=56, steps=9000, burn=2000, chunk=1500,
                 max_steps=30000, auto_extend=True, seed=seed))
