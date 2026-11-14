"""Verify the MILP tracer assignments against the frozen fit inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import CosmoMatcher as matcher

ROOT = Path(__file__).resolve().parent


def csv(path):
    return pd.read_csv(path, dtype={"grb_id": str, "calibrator_id": str})


def solve(target, tracer, ids):
    problem = matcher.build_problem(target, tracer, groups=ids, tol=.05,
                                    tol_type="dd", omega_m=.3)
    links, cert = matcher.solve_problem(problem, log_func=lambda _: None)
    assert cert["optimal"]
    return links[np.argsort(links[:, 0], kind="stable"), :2].T


def verify(grb, sn, hii, covariance):
    g, s = solve(grb.z, sn.zHD, sn.CID)
    h, j = solve(hii.z, grb.z, grb.grb_id)
    folder = ROOT / "inputs" / "reference"
    sn_used, hii_used = (csv(folder / f"{name}.csv") for name in ("sn", "hii"))
    np.testing.assert_array_equal(grb.grb_id.iloc[g], sn_used.grb_id)
    np.testing.assert_array_equal(sn.CID.iloc[s], sn_used.calibrator_id)
    np.testing.assert_array_equal(s, sn_used.parent_row)
    np.testing.assert_allclose(sn.zHD.iloc[s], sn_used.calibrator_z, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sn.m_b_corr.iloc[s], sn_used.m_b_corr, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(grb.grb_id.iloc[j], hii_used.grb_id)
    np.testing.assert_array_equal(hii.calibrator_id.iloc[h], hii_used.calibrator_id)
    np.testing.assert_allclose(hii.z.iloc[h], hii_used.calibrator_z, rtol=0, atol=1e-12)
    cols = ["log_sigma", "log_flux", "err_log_sigma", "err_log_flux"]
    np.testing.assert_allclose(hii[cols].iloc[h], hii_used[cols], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(covariance[np.ix_(s, s)], np.load(folder / "sn_cov.npy"))
    ids = set(sn_used.grb_id) | set(hii_used.grb_id)
    shared = len(set(sn_used.grb_id) & set(hii_used.grb_id))
    assert (len(g), len(h), shared, len(ids)) == (33, 69, 19, 83)
    return {"sn": len(g), "hii": len(h), "shared_grbs": shared,
            "unique_grbs": len(ids), "covariance_max_error": 0.}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pantheon_table", type=Path)
    parser.add_argument("pantheon_covariance", type=Path)
    args = parser.parse_args()
    grb = csv(ROOT / "inputs" / "a118_full.csv")
    hii = csv(ROOT / "inputs" / "hii_full_190.csv")
    sn = pd.read_csv(args.pantheon_table, sep=r"\s+", dtype={"CID": str})
    covariance = matcher.load_matrix(args.pantheon_covariance, expected_n=len(sn))
    report = verify(grb, sn, hii, covariance)
    print(json.dumps(report, indent=2))


