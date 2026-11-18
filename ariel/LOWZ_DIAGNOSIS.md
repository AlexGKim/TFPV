# Why the DR2 Tully-Fisher catalogues disagree at low redshift

*Diagnosis of the low-z `LOGDIST` offset in `catalogs/DESI-DR2_TF_pv_*.fits` (Hier: v5b, v5_3, v10b; Hyper: v5a, v7a, v9). Produced by `diagnose_lowz.py` (shared code in `lowz_common.py`); figures in `output/lowz_diagnosis/figs/`, numbers in `output/lowz_diagnosis/results/*.json`, full tables in `output/lowz_diagnosis/tables.md`. 2026-10-13.*

Notation: η = `LOGDIST` = 0.2 (`MU_ZCMB` − `MU_TF`) = log10(d_z / d_TF). A galaxy receding faster than the Hubble flow has η > 0, so η ≈ +K(z)·v_r with K = dη/dv_r ≈ 1/(c z ln10). Negative η at low z means either net infall toward us or galaxies that are fainter than the TF prediction. All velocity-unit numbers below are ⟨η⟩/⟨K⟩ for the bin.

## Summary

Every catalogue gives negative mean η below z ≈ 0.02 (−0.02 to −0.08 dex), and the catalogues differ from each other by up to 0.07 dex there. Three separate things produce this, in decreasing order of size:

1. **The observed Tully-Fisher relation itself drifts with redshift** (model-independent, not peculiar velocity, not in the mock). An inverse-TF regression (logV on M_obs, colour, axis ratio), fitted at 0.03 < z < 0.07 and immune to magnitude selection, shows ⟨η_inv⟩ running from −0.05 dex at z < 0.01 to +0.007 at z ≈ 0.06 (0.29 mag at fixed logV). The same regression on the AbacusSummit mock is flat (−0.14 ± 0.64 dex per unit z versus +0.9 in the data). The drift is identical in the North and South photometric systems. Its amplitude at z = 0.015–0.02 is equivalent to −440 km/s, four to five times what any plausible flow could do. Each catalogue's fitted TF model absorbs part of this drift inside its training redshift range (k-correction-like terms, redshift-dependent slope) and extrapolates it differently below the training cut. That extrapolation is what makes the catalogues disagree: the two Hyper training cuts differ by 0.07 dex at z < 0.015, Hier versus Hyper at the same cut by 0.04, and the three Hier versions by less than 0.01.

2. **A real coherent peculiar velocity** of the DESI low-z volume toward us. The 2M++ density reconstruction (Carrick et al. 2015 and Said et al. 2020 variants, via `pvhub`) predicts a mean radial velocity of −30 to −90 km/s for exactly these galaxies at z < 0.03, i.e. η_pred ≈ −0.003 to −0.008 dex. Independently, a monopole+dipole GLS fit finds a 1/z component in the data, and a dipole of ≈ 90–160 km/s (2–3σ, p ≈ 0.007 against a sky-shuffle null) pointing into the Virgo/north-galactic-pole quadrant. For the catalogue whose configuration matches the mock (v5b: Hier, z > 0.01 training, fixed slope) the predicted flow plus the pipeline's own bias accounts for the offset at 0.01 < z < 0.025 to within ±10 km/s, leaving only the innermost bin (z < 0.01, n = 128) at −64 ± 43 km/s. Peculiar velocity is therefore real and sufficient for the best-configured catalogue, but it is the smaller term for the others.

3. **Low-mass galaxies that do not follow the bright-end TF relation, in the Hyper sample only.** At z < 0.025 the Hyper MAIN catalogue keeps 1188 galaxies that fail the Hier phase-space cut (median logV = 1.75, M_r = −17.3, 44% irregular, σ_logV = 0.09). Their mean η is −0.056 against −0.017 for the galaxies both samples share, and their inverse-TF residual is −0.11 to −0.15 dex (equivalent to −600 to −1500 km/s, impossible for peculiar velocity). They triple the Hyper low-z offset relative to Hier. Within the Hier sample this effect is minor: irregulars are more negative than spirals by about 150 ± 65 km/s (v5b) but are 17% of the low-z sample, contributing ≈ −25 km/s to the mean.

What is **not** the cause: the pipeline's selection modelling or extrapolation on its own (the mock reproduces neither the offset nor the drift), a heliocentric/CMB frame error (the catalogue z_CMB is reproduced to 0.6 km/s rms), or the shift of the galaxy population toward lower logV at low z acting through the prediction's residual slope (the mock has the same population shift and the same slope and stays flat).

## Evidence, test by test

### T6 — Mock control (`t6_mock`)
The Hier pipeline applied to an AbacusSummit mock (true TFR without curvature, realistic peculiar velocities, the same MAIN selection) gives mean η = −0.006 ± 0.011 (z < 0.01), +0.001 (0.01–0.015), −0.004 (0.015–0.02), rising to +0.009 at z ≈ 0.06. The pipeline's own bias, ⟨η − η_true⟩, is −0.001 to −0.004 at z < 0.02 and +0.008 at z ≈ 0.06; it is used below as the template b_model(z). The mock population matches the data in median logV and M_r per bin to within 0.06 dex and 0.3 mag. So selection plus extrapolation, with random peculiar velocities, produces at most a 0.004 dex deficit at low z; the data show 0.02–0.08.

### T1 — Shape of the offset (`t1_shape`)
Mean η per catalogue (MAIN rows), dex / km/s:

| z range | v5b | v5_3 | v10b | v5a | v7a | v9 |
|---|---|---|---|---|---|---|
| 0.002–0.010 | −0.023 / −101 | −0.030 / −136 | −0.032 / −142 | −0.016 / −67 | −0.083 / −347 | −0.069 / −308 |
| 0.010–0.015 | −0.007 / −63 | −0.015 / −130 | −0.013 / −115 | +0.032 / +274 | −0.038 / −327 | −0.034 / −296 |
| 0.015–0.020 | −0.010 / −117 | −0.017 / −199 | −0.018 / −205 | +0.013 / +150 | −0.030 / −345 | −0.026 / −303 |
| 0.020–0.025 | 0.000 / −1 | −0.007 / −99 | −0.007 / −105 | +0.014 / +203 | −0.012 / −184 | −0.010 / −144 |
| 0.030–0.040 | +0.007 / +153 | +0.002 / +37 | +0.001 / +31 | +0.005 / +105 | +0.009 / +199 | +0.010 / +219 |
| 0.050–0.070 | +0.007 / +234 | +0.005 / +162 | +0.014 / +514 | +0.003 / +117 | +0.007 / +266 | +0.007 / +241 |

Nested shape fits on group-averaged data (constant / +1/z term / +linear-in-z term) prefer a 1/z term for v5b (b = −68 ± 27 km/s, ΔAIC −4.5) and v5_3 (−119 ± 26, ΔAIC −18.6), a linear term for v10b (the catalogue with a fitted redshift-dependent slope), and very large 1/z amplitudes for the Hyper z > 0.03 catalogues (v7a: −549 ± 21 km/s), which are not physical velocities. A joint fit across the six catalogues with one shared 1/z amplitude and per-catalogue zero-point and linear drift gives U = −144 ± 17 km/s (formal error; the catalogues share galaxies so the realistic error is ~2–3× larger), consistent with the earlier full-covariance bulk-flow result on grouped v5b (U = −161 ± 59 km/s).

Note that the positive high-z values (+0.007 to +0.014 dex at z > 0.04, i.e. +200 to +500 km/s) cannot be peculiar velocities either; they are the same drift seen from the other side, and the pipeline bias in the mock has the same sign and roughly the same size there (+0.008).

### T3 — Galaxy properties (`t3_lever`)
*Inverse TF (pipeline-independent).* logV regressed on (M_obs + 20, g−r, BA) at 0.03 < z < 0.07, uniform weights, slope b = −0.120 dex/mag, scatter 0.059 dex; η_inv = residual/(5b). Hier sample:

| z range | ⟨η_inv⟩ data | km/s | ⟨η_inv⟩ mock |
|---|---|---|---|
| 0.002–0.010 | −0.051 ± 0.012 | −226 | +0.006 ± 0.013 |
| 0.010–0.015 | −0.036 ± 0.007 | −310 | +0.011 ± 0.007 |
| 0.015–0.020 | −0.038 ± 0.004 | −443 | +0.002 ± 0.004 |
| 0.020–0.025 | −0.021 ± 0.003 | −323 | +0.006 ± 0.004 |
| 0.030–0.040 | −0.009 ± 0.002 | −196 | +0.002 ± 0.002 |
| 0.050–0.070 | +0.007 ± 0.001 | +252 | −0.001 ± 0.002 |

PHOTSYS = N and S give the same trend (+0.86 and +0.88 dex/z). Adding surface brightness or angular D26 as a further covariate (one at a time) reduces the drift by about 20% each (0.88 → 0.72 and 0.67 dex/z); adding both together removes it, but that combination reconstructs the distance modulus from (m_r, D26, BA) and is degenerate with z, so it is not evidence. The partial reductions are consistent with the drift being tied to isophotal/aperture quantities, but the data alone cannot separate that from an angular-size-dependent selection.

*Residual slopes.* Within every z bin η decreases with logV (−0.2 to −0.45 dex/dex) and the slope steepens with z. The mock's pipeline bias η − η_true shows the same behaviour (−0.25 to −0.70), so this is a property of the posterior-predictive step, not of the galaxies. It means property splits must be read against the mock: splitting the z < 0.025 sample at the median logV gives +159 / −285 km/s (v5b, below / above) and +311 / −340 for the mock bias. The offset sits in the *high*-logV, normal spirals (−237 ± 33 km/s for logV > 2, BA < 0.7, 0.35 < g−r < 0.65, spiral; +39 ± 27 for the rest in v5b), the opposite of what a dwarf-contamination explanation predicts for the Hier sample.

*Transfer test* (fit η on properties plus linear z at z > 0.03, extrapolate to low z) is biased by −0.02 to −0.03 dex on the mock, where the truth is ≈ 0, because the pipeline bias is not linear in z. It is reported but not used.

### T5 — External velocity field (`t5_external`)
pvhub 2M++ predicted radial velocities for each galaxy (both 2M++ variants agree to ~10 km/s in the means), converted with the exact η(z, v) map, compared as bin means (regression slopes would be diluted by the ~150 km/s per-galaxy reconstruction noise). Remainder = η − η_pred − b_model(z):

| z range | ⟨v_pred⟩ km/s | v5b remainder km/s | v5_3 | v10b | v7a | v9 |
|---|---|---|---|---|---|---|
| 0.002–0.010 | −31 | −64 ± 43 | −98 | −105 | −319 | −255 |
| 0.010–0.015 | −39 | +8 ± 46 | −58 | −44 | −258 | −225 |
| 0.015–0.020 | −81 | −7 ± 42 | −89 | −95 | −244 | −202 |
| 0.020–0.025 | −40 | +10 ± 38 | −88 | −95 | −179 | −139 |

For v5b the predicted flow plus pipeline bias closes the budget at z > 0.01 (shape fit of the remainder: b = +47 ± 27 km/s, no residual 1/z term). For v5_3 and v10b about −60 to −100 km/s remains; for the Hyper catalogues −140 to −320 km/s remains. These remainders are catalogue-specific and therefore model, not sky.

### T4 — Spatial structure (`t4_spatial`)
GLS fit of zero-point + radial monopole U + bulk flow V + shell offsets + N/S term, σ_v and error scale profiled, one row per group, z < 0.04. All three Hier catalogues: |V| = 85–92 ± 25 km/s toward (l, b) ≈ (280°, +72°) (RA 187°, Dec +10°), sky-shuffle p = 0.007; U is unconstrained once shells are free (−75 ± 150 km/s); shell offsets consistent with zero. The full-covariance fit on the grouped v5b catalogue (`b1_fit.fit`, 4421 groups) gives |V| = 162 ± 81 km/s toward RA 191°, Dec +63°, σ_v = 300 km/s, error scale 0.82, and the same shell offsets (innermost −0.029 ± 0.033), reproducing the earlier `b2_shells` result; the dipole amplitude is robust at the 2–3σ level but its direction is only constrained to the Virgo/north-galactic-pole quadrant. The same design fitted to the 2M++ prediction for the same galaxies gives |V| = 220 ± 74 km/s toward (316°, −1°), the Shapley/Great-Attractor direction, so the data dipole is not the reconstruction's dipole. Literature bulk flows projected on the footprint give ⟨n̂·V⟩ = −20 to −50 km/s. η at z < 0.015 is ≈ 0 within 20° of Virgo and −90 to −200 km/s beyond 30°. The Hyper catalogues return U = −500 to −760 km/s with Z = +0.025 to +0.035 dex: that is the drift being fitted as a flow, not a velocity. A within-group test was not possible: the Hier catalogues contain essentially one galaxy per group (8342 groups from 8350 galaxies at z < 0.04). Full-covariance reproduction of the earlier bulk-flow fit on grouped v5b is recorded in `tables.md`.

### T2, T8 — consistency checks
Scatter growth toward low z gives σ_v = 250 (225–350) km/s for v5b but 350–650 for the other catalogues, inflated by their drifts; it is not a discriminator. The D26 size test gives κ_int = −0.10 dex/mag at z > 0.03, a low-z slope of −0.12 ± 0.013 (between κ_int and the pure-distance value −0.2, as expected for a ~1/3 PV share of the variance), and a low-z mean ⟨Δlog D26⟩ = −0.0085 ± 0.0064 against −0.012 (PV) or −0.006 (intrinsic): inconclusive at 1σ.

### T7 — Catalogue pairs (`t7_pairs`)
Δη at z < 0.015: v7a − v5a = −0.070 (Hyper training cut), v5b − v5a = −0.042 (framework at the same cut; v5a goes *positive* at 0.01 < z < 0.03 because its calibration absorbed the low-z galaxies), v10b − v5b = −0.009 (redshift-dependent slope; linear coefficient +0.40 dex/z), v5_3 − v5b = −0.007 (Hier training cut; a near-constant zero-point shift), v9 − v7a = +0.007. Regressions of Δη on (logV, g−r, z) explain 36–61% of the variance within a framework and 8–15% across frameworks.

## Conclusions and what to do next

- The disagreement between catalogues at low z is dominated by how each model handles a **redshift drift of the observed TF relation** (≈ 0.9 dex per unit z in η, 0.29 mag over 0.005 < z < 0.06). It is not peculiar velocity (wrong amplitude by a factor 4–5, present in both photometric systems, absent in the mock) and not the modelled selection. Its origin is not settled here; the partial dependence on surface brightness and angular size points at isophotal/aperture systematics (SB26 magnitudes and the 0.4 R26 velocity radius shrink physically with z through cosmological dimming; fibre-aperture averaging of rotation curves in small galaxies) or at angular-size-dependent selection. The decisive next test is to inject (1+z)⁴ dimming of the isophote and an angular-size selection into the mock and see whether the inverse-TF drift appears.
- There is also a **genuine coherent infall of the DESI low-z volume** of order 50–100 km/s (2M++ predicts 30–90; the data show a 90 km/s dipole toward Virgo at p ≈ 0.007). For v5b this is the whole residual at 0.01 < z < 0.025 after the pipeline bias; it is the floor that no TF model can remove and should be handled by a velocity model (the deferred C_vv term), not by the TF fit.
- **Low-mass galaxies** (logV < 1.9, M_r > −18.5) are off the bright-end relation by 0.1–0.15 dex and should not be given TF distances from a bright-end calibration; the Hier phase-space cut already removes them, which is why Hier's low-z offset is a third of Hyper's.

## Reproduction

```
conda activate tfpv
python diagnose_lowz.py t6_mock
python diagnose_lowz.py t1_shape
python diagnose_lowz.py t3_lever
python diagnose_lowz.py t5_external          # needs data/external/pvhub with the 2M++ maps
python diagnose_lowz.py t4_spatial           # add: --catalogs v5b t4_spatial --grouped  for the full-covariance check
python diagnose_lowz.py t2_scatter; python diagnose_lowz.py t8_size; python diagnose_lowz.py t7_pairs
python diagnose_lowz.py report               # -> output/lowz_diagnosis/tables.md
```

The 2M++ maps are Git-LFS files; they were fetched from `https://media.githubusercontent.com/media/KSaid-1/pvhub/master/data/` into `data/external/pvhub/data/`. Cosmicflows-4 distances were not used (the EDD export is form-driven); the 2M++ comparison is the external test.
