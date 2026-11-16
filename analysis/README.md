# Reproducibility

The paired calibration uses the 83 A118 GRBs from the 5% fractional comoving-distance window and MILP assignment. The two assignments contain 33 Pantheon+ and 69 H II galaxy links, with 19 shared bursts. We fit the constant, linear-in-redshift and linear-in-ln(1+z) Amati slopes on these same pairs with constant relative zero points. The frozen input folder is inputs/reference. The likelihood carries the Pantheon+ covariance submatrix and the covariance from shared bursts.

## Replay and verify

From the analysis folder, use Python 3.12 with the dependencies in requirements.txt. Unpack the LaTeX files under latex at the same level as analysis to rebuild the manuscript assets.

    python verify_fits.py
    python verify_benchmarks.py
    python -X utf8 make_assets.py
    python make_literature_comparison.py
    python run_sample.py reference linear

The other paired models are constant and log. Each fit uses independent uniform priors and its recorded seed. New fits write to runs. The archive contains fit configurations, optima, summaries and compact posterior samples. The archived baseline full HDF5 chains remain in the parent analysis version. With those chains present, pass --full-chains to either fit verifier. The independent linear repeat in checks used seed 2026111491 and 72 walkers, 16000 initial steps, 3000 burn-in, 2000-step chunks and automatic extension up to 50000 steps. It checks the numerical sensitivity of the upper interval endpoint near zero.

## Pairing replay

Use the [CosmoMatcher v1.4.3 release](https://github.com/HUJIAN0000/CosmoMatcher/releases/tag/v1.4.3), commit 38f092ad7f12d45941e1ca438c605b4cb158dffc. Download its [CosmoMatcher.py](https://raw.githubusercontent.com/HUJIAN0000/CosmoMatcher/38f092ad7f12d45941e1ca438c605b4cb158dffc/CosmoMatcher.py) into the analysis folder. The SHA-256 of the release file used here is 9f7f3d233f8b7f17571c1ec7e2389e51407a4d5e60728952bb84bb756d41c956. Obtain Pantheon+SH0ES.dat and Pantheon+SH0ES_STAT+SYS.cov from the [Pantheon+ data release](https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR) and run:

    python verify_pairings.py PATH_TO_DATA PATH_TO_COVARIANCE

The script checks exact pair identities, Pantheon+ parent row positions and covariance, and tracer measurements against the frozen inputs. The H II galaxies are the targets for GRB--H II matching, and the GRBs are the targets for GRB--SN matching. The release code is only needed for this pairing replay.

## Constant-slope literature comparison

The fixed flat LambdaCDM benchmark uses H0 = 70 km/s/Mpc and Omega_m = 0.30. Its two samples are the full 118 parent bursts and the same 83 bursts used in the paired calibration. The GRB-only likelihood uses redshifts, energies, fluences and propagated measurement errors, and infers the intercept, constant slope and intrinsic scatter. The verifier compares the matched GRB observables with the parent catalog and checks the likelihood with independent quadrature.

    python run_benchmark.py a118
    python run_benchmark.py reference
    python verify_benchmarks.py --full-chains

Each benchmark used 96 walkers, 6000 initial steps and a burn-in of 1500 steps. Both reached 8000 steps and passed the 50-autocorrelation-times and 5% stability checks. The LaTeX source package can be built independently with latexmk -pdf main.tex.

The published table entries are marginal slope constraints for log10(E_iso) on log10(E_p) using A118 or a subset of it. Source locators:

- [Khadka et al. (2021)](https://arxiv.org/html/2105.12692) uses equation (9). See the marginalized-parameter table in section 5 for the two flat LambdaCDM rows, A118 and A118 + BAO + H(z). The GRB-only fit fixes H0 to 70 and varies the matter density. The multi-probe fit also varies H0.
- [Cao et al. (2022)](https://arxiv.org/html/2110.14840) reports the slope as beta. See the marginalized-parameter table in section 5, flat LambdaCDM, A118 alone. The fit varies the matter density with H0 = 70 and Omega_b = 0.05 fixed.
- [Liang et al. (2022)](https://arxiv.org/html/2211.02473), table 1, gives the Pantheon GP result for 20 A118 bursts at z < 1.4.
- [Li et al. (2023)](https://arxiv.org/pdf/2212.14291), table 2, fits the same 20 bursts with two likelihoods using distances from a cosmic-chronometer GP reconstruction.
- [Han et al. (2024)](https://link.springer.com/article/10.1140/epjc/s10052-024-13304-5), table 1, gives the two GP results in the 48-burst standard Amati rows for H(z) and Pantheon+, respectively.

## Sensitivity to H II calibration and MILP matching

The appendix tests five configurations against the 83-burst baseline: a fixed H II L--sigma slope of 5.022 from González-Morán et al. (2019, equation 18); the 33 original SN Ia links alone; a 3% distance window; and fiducial matching densities of 0.25 and 0.35 at 5%. For all matching checks only one setting changes, with 0.30 used at 3%. Each configuration has constant, linear and logarithmic Amati fits with the same priors on free coordinates. The full chains for the 15 fits remain under sensitivity_runs in the working folder. The archive carries fit configurations, optima, complete parameter summaries and compact posterior draws.

The three alternative pair catalogs are bundled. To replay the fits from a fresh extract and regenerate the table:

    python run_sensitivity.py fixed_hii
    python run_sensitivity.py sn_only
    python run_sensitivity.py window03
    python run_sensitivity.py om025
    python run_sensitivity.py om035
    python summarize_sensitivity.py

After downloading CosmoMatcher and Pantheon+ as described above, add --rebuild-pairs PATH_TO_DATA PATH_TO_COVARIANCE to the rematched cases to regenerate their catalogs. With the full chains in place, add --full-chains to summarize_sensitivity.py to verify their reported quantiles. The table generator also checks the normalization and covariance against the dense multivariate-normal calculation. The machine-readable sensitivity_validation.json includes the 95% intervals and within-configuration AIC differences.

## Finite pair separations and Amati intercept drift

The three offset fits use the same pairs and covariance as the main analysis and
subtract the pair distance-modulus differences evaluated in flat LambdaCDM with
Omega_m = 0.30. The intercept fits use the original residuals and add a single
Amati drift coefficient a1 multiplying z at the A118 reference energy of 1 keV.
It is shared by both tracer families. Free coefficients use the main analysis
priors with an additional uniform a1 prior on (-2, 2). All logarithmic slope
fits use natural logarithms and retain the existing b1 prior (-2, 2).

    python run_robustness.py offset_constant
    python run_robustness.py offset_linear
    python run_robustness.py offset_log
    python run_robustness.py intercept_linear
    python run_robustness.py joint_linear
    python run_robustness.py influence
    python run_robustness.py verify
    python make_robustness_assets.py robustness_table.tex intercept_slope.pdf

The new fit outputs are in robustness/. The local revision directory holds the
full HDF5 chains. Use verify --full-chains there or after rerunning the five fits
to verify the full-chain quantiles. The distributed validation.json records the
verification performed on the full chains. Use an isolated directory when
regenerating table fragments. The submission source includes the tables inline.
