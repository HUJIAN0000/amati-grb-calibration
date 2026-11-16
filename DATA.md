# Data and software provenance

The v1.0.0 reproducibility archive contains the matched inputs and manifests used for the manuscript. Each input manifest records hashes for the frozen inputs. The repository retains the ordered pairing indexes and source references.

| Input | Source and processing |
| --- | --- |
| A118 gamma-ray bursts | [Khadka et al. (2021), JCAP 09, 042](https://doi.org/10.1088/1475-7516/2021/09/042), [arXiv v3, Table 3](https://arxiv.org/html/2105.12692v3). Reads the burst ID, redshift, rest-frame peak energy and its error in keV, and converts the fluence and its error from the table's 10⁻⁵ units to erg cm⁻². |
| Pantheon+ | [Scolnic et al. (2022)](https://doi.org/10.3847/1538-4357/ac8b7a), [Brout et al. (2022)](https://doi.org/10.3847/1538-4357/ac8e04) and [Carr et al. (2022)](https://doi.org/10.1017/pasa.2022.41). Uses zHD, m_b_corr and the statistical-plus-systematic covariance from [DataRelease, commit c447f0f](https://github.com/PantheonPlusSH0ES/DataRelease/tree/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR). |
| H II galaxies | [Chávez et al. (2025)](https://doi.org/10.1093/mnras/staf386), [Llerena et al. (2023)](https://doi.org/10.1051/0004-6361/202346232) and [hiigs release files, commit 037ff35](https://github.com/blackdragonae/hiigs/tree/037ff35d2478098a611423a4892464546c658bc0/dat/indat). Reconstructs 181 Union galaxies with the width cut and nine Llerena additions using the release width and flux conversions, extinction correction, EW(Hβ) and width cuts. Redshifts for these additions come from hzLlerena.csv and the coordinate table supplies membership. |
| H II slope sensitivity | [González-Morán et al. (2019), equation 18](https://doi.org/10.1093/mnras/stz1577) supplies the fixed slope 5.022. |
| Matching | [Hu et al., CosmoMatcher v1.4.3](https://github.com/HUJIAN0000/CosmoMatcher/releases/tag/v1.4.3), commit 38f092ad7f12d45941e1ca438c605b4cb158dffc. |

## Pair indexes and generated files

- sn_index.csv and hii_index.csv contain grb_id, calibrator_id and zero-based parent_row. Read IDs as strings to preserve leading zeros. H II identifiers include the survey namespace.
- The generated grb.csv has GRB peak energies and fluences with their errors. The generated sn.csv has calibrator_z and m_b_corr. The generated hii.csv has calibrator_z, log_sigma, log_flux, err_log_sigma and err_log_flux. H II width and flux logs are base ten, relative to km s⁻¹ and erg s⁻¹ cm⁻².
- The generated sn_cov.npy contains the Pantheon+ submatrix in squared magnitudes, ordered as in sn.csv. The primary matrix is 33 × 33 and the 3% check is 28 × 28.
- The primary MILP uses fractional comoving-distance tolerance 0.05 with fiducial flat Ωm = 0.30. The GRB is the target for SN Ia pairing and the H II galaxy is the target for H II pairing. The likelihood carries the covariance from GRBs shared between the two families.
- Fit configurations record the SHA-256 of their input manifest. Use the matching Release archive for the files consumed by the verification scripts.

## Rights

The upstream catalogs and CosmoMatcher source are retrieved from their respective sources. Cite those sources and consult their terms for reuse. CosmoMatcher v1.4.3 describes its [licensing status](https://github.com/HUJIAN0000/CosmoMatcher/blob/v1.4.3/docs/LICENSING_STATUS.md). The [hiigs license file](https://github.com/blackdragonae/hiigs/blob/037ff35d2478098a611423a4892464546c658bc0/LICENSE) describes its software grant. The [MIT License](LICENSE) covers the original analysis code and its associated software documentation. It does not grant rights to the upstream catalogs, externally retrieved software, or the scientific data and results.
