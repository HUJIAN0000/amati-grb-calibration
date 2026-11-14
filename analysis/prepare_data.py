"""Download pinned source data and restore the fit inputs from published pair indexes."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from amati_pipeline.data import ROOT, sha256

PANTHEON = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
            "c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
HIIGS = ("https://raw.githubusercontent.com/blackdragonae/hiigs/"
         "037ff35d2478098a611423a4892464546c658bc0/dat/indat/")
MATCHER = ("https://raw.githubusercontent.com/HUJIAN0000/CosmoMatcher/"
           "38f092ad7f12d45941e1ca438c605b4cb158dffc/CosmoMatcher.py")
SOURCES = {
    "a118.html": ("https://arxiv.org/html/2105.12692v3", None),
    "Pantheon+SH0ES.dat": (PANTHEON + "Pantheon%2BSH0ES.dat",
                          "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8"),
    "Pantheon+SH0ES_STAT+SYS.cov": (PANTHEON + "Pantheon%2BSH0ES_STAT%2BSYS.cov",
                                   "abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc"),
    "Union2020v16.dat": (HIIGS + "Union2020v16.dat",
                         "d037d162468ffaa57507468c7957ebc48ebd5932e42ab084be4c45eb967776c9"),
    "hzLlerena.csv": (HIIGS + "hzLlerena.csv",
                      "133637bdd96baf14f36974ab95281e6eb828ad8586f4e099c4c578dd060b6670"),
    "Llerena2023FluxesClean.csv": (HIIGS + "Llerena2023FluxesClean.csv",
                                   "d6596b44dfc3ffc142a3a0b485c4163a797f05cdcf97b100d4d97129ad479e5f"),
    "Llerena2023Redshifts.tex": (HIIGS + "Llerena2023Redshifts.tex",
                                 "417074553fbb0651dc39268d30e19c236bde7b61f29952bdffcacbe0cdda9340"),
}
MATCHER_SHA256 = "9f7f3d233f8b7f17571c1ec7e2389e51407a4d5e60728952bb84bb756d41c956"
A118_SHA256 = "aa33e8a22d03344b47ea65e204be61b59fb642646b4419ae870dba9b40003e0d"
FOLDERS = ("inputs/reference", "sensitivity_inputs/window03", "sensitivity_inputs/om025",
           "sensitivity_inputs/om035")


def fetch(path, url, expected):
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".download")
        request = Request(url, headers={"User-Agent": "Amati GRB calibration reproducibility"})
        with urlopen(request, timeout=90) as response:
            tmp.write_bytes(response.read())
        if expected and sha256(tmp) != expected:
            raise ValueError(f"Checksum mismatch for the download from {url}")
        tmp.replace(path)
    if expected and sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


class TableParser(HTMLParser):
    """Capture the text of direct table rows, as published in arXiv HTML."""
    def __init__(self):
        super().__init__()
        self.tables = []
        self.rows = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            self.tables.append(self.rows)
            self.rows = None


def a118(html):
    parsed = TableParser()
    parsed.feed(html)
    table = next(rows for rows in parsed.tables
                 if any(row and row[0] == "080916C" and "6953.87" in row[2] for row in rows))
    records = []
    for row in table:
        if len(row) < 4 or not re.fullmatch(r"\d{6}[A-Z]?", row[0]):
            continue
        energy = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", row[2])]
        fluence = [float(x) * 1e-5 for x in re.findall(r"\d+(?:\.\d+)?", row[3])]
        if len(energy) != 2 or len(fluence) != 2:
            raise ValueError(f"Unexpected A118 measurement format for {row[0]}")
        records.append([row[0], float(row[1]), *energy, *fluence])
    frame = pd.DataFrame(records, columns=("grb_id", "z", "ep_keV", "ep_err_keV", "fluence", "fluence_err"))
    if len(frame) != 118 or not frame.grb_id.is_unique:
        raise ValueError("Table 3 must contain 118 distinct GRBs")
    return frame


def hii_parent(cache):
    cols = ("source_index", "log_sigma", "log_flux", "z", "err_log_sigma", "err_log_flux", "err_z", "flag")
    union = pd.DataFrame(np.loadtxt(cache / "Union2020v16.dat"), columns=cols)
    keep = (union.flag != 0) & (union.log_sigma - union.err_log_sigma <= 1.83)
    union = union[keep].astype({"source_index": int, "flag": int}).copy()
    union["calibrator_id"] = "UNION:" + union.source_index.astype(str)

    width = pd.read_csv(cache / "hzLlerena.csv", dtype={"ID": str},
                        usecols=["ID", "z", "w_O3_single", "w_O3_single_err"])
    flux = pd.read_csv(cache / "Llerena2023FluxesClean.csv", dtype={"ID": str},
                       usecols=["ID", "Hb", "HbErr", "Ha", "HaErr", "EWHb"])
    redshifts = []
    for line in (cache / "Llerena2023Redshifts.tex").read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split("&")]
        if len(fields) == 5 and re.fullmatch(r"(?:[A-Z]{3,4})?\d+", fields[0]):
            redshifts.append(fields[0])
    # The release computes with the redshift in the width table. The coordinate
    # table is used for sample membership and contains rounded redshifts.
    galaxies = width.merge(flux, on="ID", validate="one_to_one")
    galaxies = galaxies.loc[galaxies.ID.isin(redshifts)]
    galaxies = galaxies.loc[galaxies.HbErr > 0].copy()
    sigma = galaxies.w_O3_single.to_numpy() + 2.91
    log_sigma = np.log10(sigma)
    err_sigma = np.sqrt(galaxies.w_O3_single_err.to_numpy()**2 + .31**2) / (np.log(10) * sigma)
    hb, ehb, ha, eha = galaxies[["Hb", "HbErr", "Ha", "HaErr"]].to_numpy().T
    detected = eha > 0
    missing = eha < 0
    kha, khb = 2.22, 3.33249
    ebv = np.zeros(len(hb))
    err_ebv = np.zeros(len(hb))
    ebv[detected] = (np.log10(2.86) - np.log10(ha[detected] / hb[detected])) / (.4 * (kha - khb))
    err_ebv[detected] = np.sqrt((eha[detected] / ha[detected])**2 + (ehb[detected] / hb[detected])**2)
    err_ebv[detected] /= abs(.4 * (kha - khb) * np.log(10))
    ebv[missing] = ebv[detected].mean()
    err_ebv[missing] = err_ebv[detected].std()
    llerena = pd.DataFrame({
        "calibrator_id": "LLERENA:" + galaxies.ID, "z": galaxies.z, "log_sigma": log_sigma,
        "err_log_sigma": err_sigma, "log_flux": np.log10(hb * 1e-18) + .4 * khb * ebv,
        "err_log_flux": np.sqrt((ehb / hb / np.log(10))**2 + (.4 * khb * err_ebv)**2),
        "flag": 7, "err_z": galaxies.z * .0001,
    })
    llerena = llerena.loc[(galaxies.EWHb >= 49) & (log_sigma - err_sigma <= 1.85)].copy()
    llerena["source_index"] = np.arange(227, 227 + len(llerena))
    if (len(union), len(llerena)) != (181, 9):
        raise ValueError("Unexpected H II catalog counts")
    frame = pd.concat([union, llerena], ignore_index=True)
    frame["parent_row"] = np.arange(len(frame))
    return frame


def restore(folder, grb, sn, hii, cov):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    for name, expected in manifest["indexes"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Index checksum mismatch: {folder / name}")
    for catalog, index_name, column in ((sn, "sn", "CID"), (hii, "hii", "calibrator_id")):
        index = pd.read_csv(folder / f"{index_name}_index.csv", dtype={"grb_id": str, "calibrator_id": str})
        rows = index.parent_row.to_numpy()
        matched = catalog.iloc[rows].reset_index(drop=True)
        np.testing.assert_array_equal(index.calibrator_id, matched[column])
        if index_name == "sn":
            index["calibrator_z"] = matched.zHD
            index["m_b_corr"] = matched.m_b_corr
            np.save(folder / "sn_cov.npy", cov[np.ix_(rows, rows)])
        else:
            index["calibrator_z"] = matched.z
            for c in ("log_sigma", "log_flux", "err_log_sigma", "err_log_flux"):
                index[c] = matched[c]
        index.to_csv(folder / f"{index_name}.csv", index=False, lineterminator="\n")
    grb.to_csv(folder / "grb.csv", index=False, lineterminator="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use only existing files in upstream/")
    args = parser.parse_args()
    cache = ROOT.parent / "upstream"
    cache.mkdir(exist_ok=True)
    if not args.offline:
        for name, (url, expected) in SOURCES.items():
            fetch(cache / name, url, expected)
        fetch(ROOT / "CosmoMatcher.py", MATCHER, MATCHER_SHA256)
    for name, (_, expected) in SOURCES.items():
        path = cache / name
        if not path.is_file() or (expected and sha256(path) != expected):
            raise ValueError(f"Missing or unexpected source file: {path}")
    if sha256(ROOT / "CosmoMatcher.py") != MATCHER_SHA256:
        raise ValueError("Expected the pinned CosmoMatcher release")
    from CosmoMatcher import load_matrix

    grb = a118((cache / "a118.html").read_text(encoding="utf-8"))
    hii = hii_parent(cache)
    sn = pd.read_csv(cache / "Pantheon+SH0ES.dat", sep=r"\s+", dtype={"CID": str})
    cov = load_matrix(cache / "Pantheon+SH0ES_STAT+SYS.cov", expected_n=len(sn))
    grb_path = ROOT / "inputs/a118_full.csv"
    grb.to_csv(grb_path, index=False, lineterminator="\n")
    if sha256(grb_path) != A118_SHA256:
        raise ValueError("A118 table differs from the frozen numerical catalog")
    hii.to_csv(ROOT / "inputs/hii_full_190.csv", index=False, lineterminator="\n")
    for name in FOLDERS:
        restore(ROOT / name, grb, sn, hii, cov)
    print("Rebuilt inputs for the 83-GRB baseline and three MILP sensitivity samples.")


if __name__ == "__main__":
    main()
