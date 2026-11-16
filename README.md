# Constraints on the Amati slope and its redshift evolution with a cosmology-independent matched-pair likelihood

This repository accompanies the study by Jian Hu, Yi Liu, Chen Wang, Liangqing Zhu,
Jia Zhao and Jian-Ping Hu. The primary calibration links 83 A118 GRBs to
33 Pantheon+ supernovae and 69 H II galaxies. The logarithmic evolution model
uses `b(z) = b0 + b1 ln(1 + z)`.

The `v0.1.0` release archives the original base-10 baseline. The revised
`v1.0.0` analysis adds sensitivity fits for finite pair separations and
Amati intercept drift.

## Reproduce the revised analysis

Download `Amati_reproducibility.zip` from the
[GitHub Releases page](https://github.com/HUJIAN0000/amati-grb-calibration/releases).
Extract it into a fresh folder. The Release asset includes the frozen numerical
inputs and archived posterior samples required by the verification commands.
Run the following from the extracted folder with Python 3.12.

```sh
python -m pip install -r analysis/requirements.txt
python analysis/verify_fits.py
python analysis/run_robustness.py verify
python analysis/make_robustness_assets.py robustness_table.tex intercept_slope.pdf
```

See [analysis/README.md](analysis/README.md) for fit and matching commands,
[DATA.md](DATA.md) for data provenance, and [CITATION.cff](CITATION.cff)
for citation metadata. The [MIT License](LICENSE) applies to this project's
original analysis code and documentation. The manuscript tables are under
[results/](results).
