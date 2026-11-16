"""Relative Amati likelihood with constant zero points."""
import numpy as np

from .data import LN10, MU_CONST

MODES = ("constant", "linear", "log")
MAG_OFFSET = MU_CONST - 2.5 * np.log10(4 * np.pi)


class RelativeCalibration:
    def __init__(self, data, evolution):
        if evolution not in MODES:
            raise ValueError(f"Unknown evolution model: {evolution}")
        self.data = data
        self.evolution = evolution
        self.config = {
            "evolution": evolution,
            "log_base": 10,
            "absolute_anchors_in_likelihood": False,
            "zero_points": "A_SN and A_HII are constant",
        }
        if evolution == "log":
            self.config["evolution_log_base"] = "e"
        self.ns, self.nh = len(data.sn), len(data.hii)
        edges = np.r_[data.sn.grb_id, data.hii.grb_id]
        grb = data.grb.set_index("grb_id").loc[edges]
        self.z = grb.z.to_numpy()
        self.x = np.log10(grb.ep_keV.to_numpy())
        self.sx = (grb.ep_err_keV / grb.ep_keV).to_numpy() / LN10
        self.ss = (grb.fluence_err / grb.fluence).to_numpy() / LN10
        fluence = np.log10(grb.fluence.to_numpy())
        self.grb_base = np.log10(1 + self.z) - fluence
        self.evolution_term = np.log1p(self.z) if evolution == "log" else self.z
        h_index = {gid: j for j, gid in enumerate(data.hii.grb_id)}
        shared = [(i, h_index[gid]) for i, gid in enumerate(data.sn.grb_id) if gid in h_index]
        self.shared_s, self.shared_h = np.asarray(shared, dtype=int).reshape(-1, 2).T
        self.mag = data.sn.m_b_corr.to_numpy()
        self.h, self.f, self.sh, self.sf = data.hii[
            ["log_sigma", "log_flux", "err_log_sigma", "err_log_flux"]].to_numpy().T

        parameters = {
            "A_SN": ((25, 60), 42.3),
            "b" if evolution == "constant" else "b0": ((-3, 5), 1.25),
            "A_HII": ((-10, 40), 11.),
            "beta_HII": ((0.1, 10), 3.),
            "sigma_GRB": ((1e-5, 5), 1.),
            "sigma_HII": ((1e-5, 5), .6),
        }
        if evolution != "constant":
            parameters["b1"] = ((-2, 2), 0.)
        self.names = list(parameters)
        self.bounds = np.array([bounds for bounds, _ in parameters.values()], dtype=float)
        self.initial = np.array([start for _, start in parameters.values()], dtype=float)

    def components(self, theta):
        p = dict(zip(self.names, np.atleast_2d(theta).T[:, :, None]))
        b = p["b"] if self.evolution == "constant" else p["b0"] + p["b1"] * self.evolution_term
        g = b * self.x + self.grb_base
        rs = 2.5 * (g[:, :self.ns] + p["A_SN"]) + MAG_OFFSET - self.mag
        rh = 2.5 * (g[:, self.ns:] + p["A_HII"] - p["beta_HII"] * self.h + self.f)
        vg = 6.25 * ((b * self.sx)**2 + self.ss**2) + p["sigma_GRB"]**2
        vh = 6.25 * ((p["beta_HII"] * self.sh)**2 + self.sf**2) + p["sigma_HII"]**2
        return rs, rh, vg, vh

    def log_likelihood(self, theta, return_parts=False):
        rs, rh, vg, vh = self.components(theta)
        ds = vg[:, :self.ns].copy()
        dh = vg[:, self.ns:] + vh
        # Exact Schur complement for the shared GRB variance between pair families.
        shared = ds[:, self.shared_s].copy()
        fraction = shared / dh[:, self.shared_h]
        ds[:, self.shared_s] -= fraction * shared
        rs[:, self.shared_s] -= fraction * rh[:, self.shared_h]
        cov = np.broadcast_to(self.data.sn_cov, (len(rs), self.ns, self.ns)).copy()
        indices = np.arange(self.ns)
        cov[:, indices, indices] += ds
        chol = np.linalg.cholesky(cov)
        white = np.linalg.solve(chol, rs[..., None])[..., 0]
        sn = -.5 * ((white**2).sum(axis=1)
                    + 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
                    + self.ns * np.log(2 * np.pi))
        hii = -.5 * np.sum(rh**2 / dh + np.log(2 * np.pi * dh), axis=1)
        parts = np.column_stack([sn, hii])
        return parts if return_parts else parts.sum(axis=1)

    def __call__(self, theta):
        t = np.atleast_2d(np.asarray(theta, dtype=float))
        valid = np.isfinite(t).all(axis=1)
        valid &= ((t > self.bounds[:, 0]) & (t < self.bounds[:, 1])).all(axis=1)
        values = np.full(len(t), -np.inf)
        values[valid] = self.log_likelihood(t[valid])
        return float(values[0]) if np.ndim(theta) == 1 else values

