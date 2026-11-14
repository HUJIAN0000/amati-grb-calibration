# Cosmology-independent constraints on the Amati slope

This repository accompanies the study by Jian Hu, Yi Liu, Chen Wang, Liangqing Zhu, Jia Zhao and Jian-Ping Hu. The primary calibration uses 83 A118 gamma-ray bursts linked to 33 Pantheon+ supernovae and 69 H II galaxies, with 19 GRBs shared between the two sets of links. We fit constant, linear-in-redshift and logarithmic-in-redshift Amati slopes with constant relative zero points.

The paired constant slope is **1.130 ± 0.127** (central 68% credible interval). The study also tests five sensitivity configurations and compares constant slopes with fixed flat ΛCDM fits to all 118 and the 83 matched bursts.

## Reproduce the tables and figures

Use Python 3.12. The first data preparation downloads about 35 MB. Run from the repository root.

~~~sh
python -m pip install -r analysis/requirements.txt
python analysis/prepare_data.py
python analysis/verify_fits.py
python analysis/verify_benchmarks.py
python analysis/summarize_sensitivity.py
python -X utf8 analysis/make_assets.py
python analysis/make_literature_comparison.py
~~~

The resulting tables and vector figures go into **generated/**. Verification checks the likelihood on the reconstructed observations and the archived fit results. The table and figure scripts use archived summaries and compact posterior draws.

| Path | Contents |
| --- | --- |
| analysis/amati_pipeline/ | Relative likelihood, shared-GRB and Pantheon+ covariance, MCMC and input validation |
| analysis/inputs/reference/ | Ordered pairing indexes for the primary sample |
| analysis/sensitivity_inputs/ | Pairing indexes for the three rematching sensitivity checks |
| analysis/runs/, analysis/checks/, analysis/benchmarks/, analysis/sensitivity_runs/ | Archived configurations, fit points, parameter summaries and compact posterior draws |
| results/ | Manuscript tables as LaTeX fragments |
| DATA.md | Source links, input schema, matching conventions and rights information |
| CITATION.cff | Citation metadata |

The data preparation script restores the numerical observations and covariance from pinned upstream files and downloads the pinned CosmoMatcher file. Use **python analysis/prepare_data.py --offline** with an existing cache. See [DATA.md](DATA.md) for dataset citations.

## Verify the MILP assignments

These commands rerun the matching and verify the ordered identities and Pantheon+ covariance.

~~~sh
python analysis/verify_pairings.py "upstream/Pantheon+SH0ES.dat" "upstream/Pantheon+SH0ES_STAT+SYS.cov"
python analysis/run_sensitivity.py window03 --verify-pairs "upstream/Pantheon+SH0ES.dat" "upstream/Pantheon+SH0ES_STAT+SYS.cov"
python analysis/run_sensitivity.py om025 --verify-pairs "upstream/Pantheon+SH0ES.dat" "upstream/Pantheon+SH0ES_STAT+SYS.cov"
python analysis/run_sensitivity.py om035 --verify-pairs "upstream/Pantheon+SH0ES.dat" "upstream/Pantheon+SH0ES_STAT+SYS.cov"
~~~

## Resample the posteriors

New results go into **work/** by default. The scripts use the archived settings and seeds. MCMC estimates can vary at the level of Monte Carlo noise across software or platform builds.

~~~sh
python analysis/run_sample.py reference constant
python analysis/run_sample.py reference linear
python analysis/run_sample.py reference log
python analysis/run_sample.py reference linear --repeat
python analysis/run_benchmark.py a118
python analysis/run_benchmark.py reference
python analysis/run_sensitivity.py fixed_hii
python analysis/run_sensitivity.py sn_only
python analysis/run_sensitivity.py window03
python analysis/run_sensitivity.py om025
python analysis/run_sensitivity.py om035
~~~

The compact NPZ arrays give draws under the key **samples** and parameter order under **names**. The archived key **A_HII** represents the relative zero point \(A_{\rm H}\) in the manuscript. Optional **--full-chains** verification requires the corresponding original HDF5 chains alongside the archived summaries.

## Citation

Please cite the accompanying study and the source datasets and software. [CITATION.cff](CITATION.cff) gives the authors and title. [DATA.md](DATA.md) lists the source publications.

## License

The original analysis code and its associated software documentation are available under the [MIT License](LICENSE). See [DATA.md](DATA.md) for the scope of the license and the terms applicable to third-party sources.
