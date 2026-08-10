#!/usr/bin/env python3
"""
Convert DESI-DR1_TF_pv_cat_v15.fits to JSON format for tophat.stan

This script reads the DESI Tully-Fisher data and converts it to the format
expected by tophat.stan.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Training sample size used when neither n_objects nor train_fraction is given.
_DEFAULT_N_OBJECTS = 5000
_DEFAULT_RANDOM_SEED = 42


def process_desi_tf_data(
    fits_file,
    data_output_file,
    init_output_file,
    haty_max=-16,
    haty_min=-1.0e9,
    plane_cut=False,
    slope_plane=None,
    intercept_plane=None,
    intercept_plane2=None,
    n_objects=None,
    random_seed=_DEFAULT_RANDOM_SEED,
    target_main_count=None,
    train_fraction=None,
    source=None,
    *,
    z_col="Z_DESI",
    z_col_candidates=(
        "zobs",
        "ZOBS",
        "Z",
        "ZHELIO",
        "Z_CMB",
        "ZDESI",
        "ZTRUE",
        "Z_DESI",
    ),
    z_obs_min=None,  # <<< NEW: Minimum redshift for inclusion
    z_obs_max=None,
    fixed_init=None,
):
    """
    Process DESI TF data: convert to Stan JSON format and create initial conditions.
    (Modified to correctly load/propagate z_obs and apply an optional redshift cut.)
    """

    # Validate plane cut parameters
    if plane_cut and (slope_plane is None or intercept_plane is None):
        raise ValueError(
            "slope_plane and intercept_plane must be provided when plane_cut=True"
        )

    # If two-sided, enforce ordering c1 < c2
    two_sided = plane_cut and (intercept_plane2 is not None)
    if two_sided and not (intercept_plane < intercept_plane2):
        raise ValueError(
            f"For a two-sided parallel cut, require intercept_plane < intercept_plane2. "
            f"Got {intercept_plane} and {intercept_plane2}."
        )

    # ============================================================================
    # SECTION 1: READ FITS DATA
    # ============================================================================
    print(f"Reading FITS file: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        names = set(data.dtype.names or ())

        # Resolve redshift column name
        if z_col in names:
            z_col_use = z_col
        else:
            z_col_use = None
            for cand in z_col_candidates:
                if cand in names:
                    z_col_use = cand
                    break
            if z_col_use is None:
                raise ValueError(
                    f"Could not find a redshift column. Tried z_col={z_col!r} and candidates "
                    f"{z_col_candidates}. Available columns include: {sorted(list(names))[:30]} ..."
                )

        # Extract velocity, magnitude, and redshift data
        if "logV" in names:
            _logV = np.asarray(data["logV"], dtype=float)
            _logV_err = np.asarray(data["logV_ERR"], dtype=float)
        else:
            _logV = np.asarray(data["LOGVROT"], dtype=float)
            _logV_err = np.asarray(data["LOGVROT_ERR"], dtype=float)

        from mag_utils import get_mag_cols

        col_abs, col_abs_err, col_app = get_mag_cols(names)

        R_ABSMAG_SB26 = np.asarray(data[col_abs], dtype=float)
        R_ABSMAG_SB26_ERR = np.asarray(data[col_abs_err], dtype=float)
        R_MAG_SB26 = np.asarray(data[col_app], dtype=float)

        # [COLOR] z-band apparent magnitude and its photometric uncertainty.
        # sigma_z = Z_MAG_SB26_ERR_CORR only: it is the measurement noise on hat_z given
        # latent z (Eq. C9). sigma_y does NOT contribute here; combining them would
        # double-count sigma_y in the B_i covariance matrix entries.
        if "Z_MAG_SB26_CORR" in names:
            if "Z_MAG_SB26_ERR_CORR" not in names:
                raise KeyError(
                    "Found Z_MAG_SB26_CORR but required column Z_MAG_SB26_ERR_CORR "
                    "is missing. Cannot construct sigma_z for color.stan."
                )
            Z_MAG_SB26 = np.asarray(data["Z_MAG_SB26_CORR"], dtype=float)
            Z_MAG_SB26_ERR = np.asarray(data["Z_MAG_SB26_ERR_CORR"], dtype=float)
        elif "Z_MAG_SB26" in names:
            if "Z_MAG_SB26_ERR" not in names:
                raise KeyError(
                    "Found Z_MAG_SB26 but required column Z_MAG_SB26_ERR "
                    "is missing. Cannot construct sigma_z for color.stan."
                )
            Z_MAG_SB26 = np.asarray(data["Z_MAG_SB26"], dtype=float)
            Z_MAG_SB26_ERR = np.asarray(data["Z_MAG_SB26_ERR"], dtype=float)
        else:
            raise KeyError(
                "No z-band apparent magnitude column found (tried Z_MAG_SB26_CORR "
                "and Z_MAG_SB26). This column is required for color.stan."
            )

        # [COLOR] z-band absolute magnitude: Z_ABSMAG = R_ABSMAG - (R_MAG - Z_MAG)
        # sigma_z = Z_MAG_SB26_ERR directly (photometric noise on z-band only)
        Z_ABSMAG_SB26 = R_ABSMAG_SB26 - (R_MAG_SB26 - Z_MAG_SB26)
        Z_ABSMAG_SB26_ERR = Z_MAG_SB26_ERR

        # [2COLOR] g-band apparent magnitude and photometric uncertainty
        if "G_MAG_SB26_CORR" in names:
            if "G_MAG_SB26_ERR_CORR" not in names:
                raise KeyError(
                    "Found G_MAG_SB26_CORR but required column G_MAG_SB26_ERR_CORR "
                    "is missing. Cannot construct sigma_g for 2color.stan."
                )
            G_MAG_SB26 = np.asarray(data["G_MAG_SB26_CORR"], dtype=float)
            G_MAG_SB26_ERR = np.asarray(data["G_MAG_SB26_ERR_CORR"], dtype=float)
        elif "G_MAG_SB26" in names:
            if "G_MAG_SB26_ERR" not in names:
                raise KeyError(
                    "Found G_MAG_SB26 but required column G_MAG_SB26_ERR "
                    "is missing. Cannot construct sigma_g for 2color.stan."
                )
            G_MAG_SB26 = np.asarray(data["G_MAG_SB26"], dtype=float)
            G_MAG_SB26_ERR = np.asarray(data["G_MAG_SB26_ERR"], dtype=float)
        else:
            G_MAG_SB26 = None
            G_MAG_SB26_ERR = None

        # [2COLOR] g-band absolute magnitude: G_ABSMAG = R_ABSMAG - (R_MAG - G_MAG)
        if G_MAG_SB26 is not None:
            G_ABSMAG_SB26 = R_ABSMAG_SB26 - (R_MAG_SB26 - G_MAG_SB26)
            G_ABSMAG_SB26_ERR = G_MAG_SB26_ERR
        else:
            G_ABSMAG_SB26 = None
            G_ABSMAG_SB26_ERR = None

        z_all_raw = np.asarray(data[z_col_use], dtype=float)

        # [SPLIT] Galaxy identifier for train/holdout tracking
        sga_id_all_raw = np.asarray(data["SGA_ID"], dtype=float) if "SGA_ID" in names else np.arange(len(_logV), dtype=float)

        # [MOCK] MAIN flag. Only read for the mock source: the frozen trapezoid
        # cuts were DERIVED from MAIN rows only (selection_ellipse.py filters to
        # MAIN when source="fullmocks"), so applying them to the whole file mixes
        # in a population the ellipse never saw -- 637 of the reference mock's
        # 90,756 cut-passing galaxies are DWARF (MAIN is exactly ~DWARF). Small
        # (0.70%), but it makes the mock inconsistent with DR2, whose Phase A
        # applies no MAIN filter so derivation and application already agree.
        #
        # Gated on source rather than mere column presence: the DR2 per-population
        # FITS files carry their own pipeline-WRITTEN MAIN column (from a previous
        # color_predict.py run), and filtering on that would silently re-select a
        # DR2 run against a stale selection.
        main_all_raw = None
        if source == "fullmocks":
            if "MAIN" not in names:
                raise KeyError(
                    'source="fullmocks" but the catalog has no MAIN column. The '
                    'frozen mock selection cuts were derived from MAIN rows, so '
                    'this file cannot be selected consistently with them.'
                )
            main_all_raw = np.asarray(data["MAIN"], dtype=bool)

    total_rows = len(_logV)

    # Reference velocity (km/s)
    V0 = 100.0

    # Filter out invalid data
    valid_mask = (
        np.isfinite(_logV)
        & np.isfinite(_logV_err)
        & np.isfinite(R_ABSMAG_SB26)
        & np.isfinite(R_ABSMAG_SB26_ERR)
        & np.isfinite(z_all_raw)
        & (_logV_err > 0)
        # Non-negativity, per band. A negative error bar is unphysical, and
        # 2color.stan declares every sigma as vector<lower=0>, so one reaching
        # input.json makes CmdStan reject the whole data file at step 6 -- after
        # the run has already spent its GPU allocation. The r-band check was here
        # from the start; z and g were checked only for isfinite, which a negative
        # value passes. That asymmetry was live: 508 of the 676 v2.0.8 mocks carry
        # a negative Z_MAG_SB26_ERR (881 rows), which is where step 4 gets its
        # sigma_z, so roughly 45 runs would have failed once such a row landed in
        # the 5,000-galaxy training draw (~5% chance per row). 133 files also have
        # a negative R_ABSMAG_SB26_ERR, already excluded here.
        & (R_ABSMAG_SB26_ERR >= 0)
        & np.isfinite(Z_ABSMAG_SB26)      # [COLOR]
        & np.isfinite(Z_ABSMAG_SB26_ERR)  # [COLOR]
        & (Z_ABSMAG_SB26_ERR >= 0)        # [COLOR]
    )
    # [2COLOR] require finite g-band if available
    if G_ABSMAG_SB26 is not None:
        valid_mask = (valid_mask & np.isfinite(G_ABSMAG_SB26)
                      & np.isfinite(G_ABSMAG_SB26_ERR)
                      & (G_ABSMAG_SB26_ERR >= 0))

    # [MOCK] Restrict to MAIN so the frozen cuts are applied to the same
    # population they were fit on (see where main_all_raw is read).
    if main_all_raw is not None:
        _n_before = int(valid_mask.sum())
        valid_mask = valid_mask & main_all_raw
        _n_after = int(valid_mask.sum())
        print(f"  MAIN filter (source=fullmocks): {_n_after} of {_n_before} valid "
              f"rows are MAIN ({_n_before - _n_after} non-MAIN dropped)")

    _logV = _logV[valid_mask]
    _logV_err = _logV_err[valid_mask]
    R_ABSMAG_SB26 = R_ABSMAG_SB26[valid_mask]
    R_ABSMAG_SB26_ERR = R_ABSMAG_SB26_ERR[valid_mask]
    Z_ABSMAG_SB26 = Z_ABSMAG_SB26[valid_mask]         # [COLOR]
    Z_ABSMAG_SB26_ERR = Z_ABSMAG_SB26_ERR[valid_mask] # [COLOR]
    if G_ABSMAG_SB26 is not None:
        G_ABSMAG_SB26 = G_ABSMAG_SB26[valid_mask]         # [2COLOR]
        G_ABSMAG_SB26_ERR = G_ABSMAG_SB26_ERR[valid_mask] # [2COLOR]
    rz_color_all = (R_MAG_SB26 - Z_MAG_SB26)[valid_mask]  # apparent r-z color
    z_all_raw = z_all_raw[valid_mask]
    sga_id_all = sga_id_all_raw[valid_mask]  # [SPLIT]

    valid_rows = len(_logV)

    # x = log10(V / V0); already stored as log10(V) so subtract log10(V0)
    x_all = _logV - np.log10(V0)
    sigma_x_all = _logV_err

    # Magnitude data
    y_all = R_ABSMAG_SB26
    sigma_y_all = R_ABSMAG_SB26_ERR
    z_absmag_all = Z_ABSMAG_SB26         # [COLOR]
    sigma_z_absmag_all = Z_ABSMAG_SB26_ERR  # [COLOR]
    g_absmag_all = G_ABSMAG_SB26         # [2COLOR] (may be None)
    sigma_g_absmag_all = G_ABSMAG_SB26_ERR  # [2COLOR]

    # Redshift data (aligned to x_all/y_all by construction)
    zobs_all = z_all_raw

    # ============================================================================
    # SECTION 2: APPLY SELECTION CUTS (NOW INCLUDES haty_min AND z_obs_min)
    # ============================================================================
    x_data, y_data, sigma_x_data, sigma_y_data, z_data = [], [], [], [], []
    z_absmag_data, sigma_z_absmag_data = [], []  # [COLOR]
    g_absmag_data, sigma_g_absmag_data = [], []  # [2COLOR]
    sga_id_data = []                              # [SPLIT]

    # Track filtering statistics
    y_filtered_rows = 0
    z_min_filtered_rows = 0  # rows surviving z_obs_min cut
    z_filtered_rows = 0      # rows surviving both z cuts
    color_filtered_rows = 0  # rows surviving r-z color cut
    plane_pass_rows = 0

    for i in range(len(x_all)):
        x_val = x_all[i]
        y_val = y_all[i]

        # Apply BOTH y limits (magnitudes: "brighter" is more negative)
        if (y_val < haty_max) and (y_val > haty_min):
            y_filtered_rows += 1

            # ---- REDSHIFT CUTS ----
            if (z_obs_min is not None) and (zobs_all[i] <= z_obs_min):
                continue
            z_min_filtered_rows += 1
            if (z_obs_max is not None) and (zobs_all[i] > z_obs_max):
                continue
            z_filtered_rows += 1
            # -----------------------


            if plane_cut:
                lower_bound_oblique = slope_plane * x_val + intercept_plane
                lower_bound = max(haty_min, lower_bound_oblique)

                if not two_sided:
                    # One‑sided: lower_bound < y
                    if lower_bound < y_val:
                        x_data.append(x_val)
                        y_data.append(y_val)
                        sigma_x_data.append(sigma_x_all[i])
                        sigma_y_data.append(sigma_y_all[i])
                        z_data.append(zobs_all[i])
                        z_absmag_data.append(z_absmag_all[i])          # [COLOR]
                        sigma_z_absmag_data.append(sigma_z_absmag_all[i])  # [COLOR]
                        if g_absmag_all is not None:
                            g_absmag_data.append(g_absmag_all[i])          # [2COLOR]
                            sigma_g_absmag_data.append(sigma_g_absmag_all[i])  # [2COLOR]
                        sga_id_data.append(sga_id_all[i])                  # [SPLIT]
                        plane_pass_rows += 1
                else:
                    # Two‑sided: lower_bound < y < min(haty_max, upper_bound_oblique)
                    upper_bound_oblique = slope_plane * x_val + intercept_plane2
                    upper_bound = min(haty_max, upper_bound_oblique)

                    if (lower_bound < y_val) and (y_val < upper_bound):
                        x_data.append(x_val)
                        y_data.append(y_val)
                        sigma_x_data.append(sigma_x_all[i])
                        sigma_y_data.append(sigma_y_all[i])
                        z_data.append(zobs_all[i])
                        z_absmag_data.append(z_absmag_all[i])          # [COLOR]
                        sigma_z_absmag_data.append(sigma_z_absmag_all[i])  # [COLOR]
                        if g_absmag_all is not None:
                            g_absmag_data.append(g_absmag_all[i])          # [2COLOR]
                            sigma_g_absmag_data.append(sigma_g_absmag_all[i])  # [2COLOR]
                        sga_id_data.append(sga_id_all[i])                  # [SPLIT]
                        plane_pass_rows += 1
            else:
                # No plane cut (just the y‑range and optional redshift cut)
                x_data.append(x_val)
                y_data.append(y_val)
                sigma_x_data.append(sigma_x_all[i])
                sigma_y_data.append(sigma_y_all[i])
                z_data.append(zobs_all[i])
                z_absmag_data.append(z_absmag_all[i])          # [COLOR]
                sigma_z_absmag_data.append(sigma_z_absmag_all[i])  # [COLOR]
                if g_absmag_all is not None:
                    g_absmag_data.append(g_absmag_all[i])          # [2COLOR]
                    sigma_g_absmag_data.append(sigma_g_absmag_all[i])  # [2COLOR]
                sga_id_data.append(sga_id_all[i])                  # [SPLIT]

    # Convert to numpy arrays for calculations
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    sigma_x = np.array(sigma_x_data, dtype=float)
    sigma_y = np.array(sigma_y_data, dtype=float)
    z_obs = np.array(z_data, dtype=float)
    z_absmag = np.array(z_absmag_data, dtype=float)          # [COLOR]
    sigma_z_absmag = np.array(sigma_z_absmag_data, dtype=float)  # [COLOR]
    if g_absmag_all is not None:
        g_absmag = np.array(g_absmag_data, dtype=float)          # [2COLOR]
        sigma_g_absmag = np.array(sigma_g_absmag_data, dtype=float)  # [2COLOR]
    else:
        g_absmag = None
        sigma_g_absmag = None
    sga_ids_main = np.array(sga_id_data, dtype=float)  # [SPLIT]

    N_after_cuts = len(x)

    # Resolve a fractional training size against the post-selection count. An
    # absolute n_objects silently stops meaning "40%" whenever the selection is
    # re-derived and N_after_cuts moves, so populations with their own selection
    # ellipse specify train_fraction instead. This only picks the number; the
    # chosen galaxies are still recorded explicitly as train_sga_ids below.
    if train_fraction is not None and not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")

    if train_fraction is not None and n_objects is not None:
        print(
            f"  WARNING: both n_objects={n_objects} and train_fraction={train_fraction} "
            f"given; using the explicit n_objects and ignoring train_fraction."
        )
        train_fraction = None
    elif train_fraction is not None:
        n_objects = int(round(train_fraction * N_after_cuts))
        print(
            f"  train_fraction={train_fraction} of {N_after_cuts} selected "
            f"-> n_objects={n_objects}"
        )
    elif n_objects is None:
        n_objects = _DEFAULT_N_OBJECTS  # legacy default when neither is specified

    # Draw a fixed-size subsample, or subsample randomly for training only
    if target_main_count is not None:
        # [SUBSET] Draw exactly `target_main_count` galaxies from the post-cut
        # sample. This is the mock analysis's sizing mechanism: it makes the
        # mock's MAIN count match the real DR2 product
        # (DESI-DR2_TF_pv_cat_v5b.fits, MAIN = 17234) so the two samples carry
        # comparable statistical weight.
        #
        # It is a single draw, not an iterative or partitioned one, because the
        # trapezoid cuts are frozen for this analysis: the post-cut population
        # is fully determined before we choose, so the target is reachable
        # exactly in one step. A fraction-of-the-file scheme cannot do this —
        # the post-cut count would fall out of the fraction rather than being
        # chosen (1/5 of the reference mock gives ~18.1k, 1/6 ~15.1k; the target
        # would need 1/5.27).
        if N_after_cuts < target_main_count:
            sys.exit(
                f"ERROR: target_main_count={target_main_count} exceeds the "
                f"{N_after_cuts} galaxies passing the selection cuts in "
                f"{fits_file}. This file cannot produce a sample matched to the "
                f"target, so step 4 fails rather than emitting an unmatched run. "
                f"Lower target_main_count, widen the cuts, or exclude this file."
            )

        rng = np.random.default_rng(random_seed)
        idx = np.sort(
            rng.choice(N_after_cuts, size=target_main_count, replace=False)
        )
        x = x[idx]
        y = y[idx]
        sigma_x = sigma_x[idx]
        sigma_y = sigma_y[idx]
        z_obs = z_obs[idx]
        z_absmag = z_absmag[idx]
        sigma_z_absmag = sigma_z_absmag[idx]
        if g_absmag is not None:
            g_absmag = g_absmag[idx]
            sigma_g_absmag = sigma_g_absmag[idx]
        subset_sga_ids = sga_ids_main[idx].tolist()
        N_subset = len(subset_sga_ids)
        assert N_subset == target_main_count, (
            f"internal error: drew {N_subset}, expected {target_main_count}"
        )
        print(
            f"  Subsample: {N_subset} of {N_after_cuts} cut-passing galaxies "
            f"(target_main_count={target_main_count}, random_seed={random_seed})"
        )

        # Training draw within the subsample; the remainder is the holdout.
        if n_objects is not None and n_objects < N_subset:
            rng2 = np.random.default_rng(random_seed + 1)
            train_idx = np.sort(rng2.choice(N_subset, size=n_objects, replace=False))
            x = x[train_idx]
            y = y[train_idx]
            sigma_x = sigma_x[train_idx]
            sigma_y = sigma_y[train_idx]
            z_obs = z_obs[train_idx]
            z_absmag = z_absmag[train_idx]
            sigma_z_absmag = sigma_z_absmag[train_idx]
            if g_absmag is not None:
                g_absmag = g_absmag[train_idx]
                sigma_g_absmag = sigma_g_absmag[train_idx]
            train_sga_ids = [subset_sga_ids[i] for i in train_idx]
            print(f"  Training: {n_objects}, holdout: {N_subset - n_objects}")
        else:
            train_sga_ids = subset_sga_ids
            if n_objects is not None:
                print(
                    f"  WARNING: n_objects={n_objects} >= the {N_subset} drawn "
                    f"galaxies, so ALL of them are training and this run has NO "
                    f"holdout (empty ANALYSIS set; step 8 predictions are "
                    f"in-sample)."
                )
    elif n_objects is not None and n_objects < N_after_cuts:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(N_after_cuts, size=n_objects, replace=False)
        idx.sort()
        x = x[idx]
        y = y[idx]
        sigma_x = sigma_x[idx]
        sigma_y = sigma_y[idx]
        z_obs = z_obs[idx]
        z_absmag = z_absmag[idx]          # [COLOR]
        sigma_z_absmag = sigma_z_absmag[idx]  # [COLOR]
        if g_absmag is not None:
            g_absmag = g_absmag[idx]          # [2COLOR]
            sigma_g_absmag = sigma_g_absmag[idx]  # [2COLOR]
        train_sga_ids = sga_ids_main[idx].tolist()  # [SPLIT] record which galaxies are training
        print(
            f"  Subsampled from {N_after_cuts} to {n_objects} objects (random_seed={random_seed})"
        )
    else:
        train_sga_ids = None  # [SPLIT] no split: all galaxies are training

    # Convert back to lists for JSON serialization
    x_data = x.tolist()
    y_data = y.tolist()
    sigma_x_data = sigma_x.tolist()
    sigma_y_data = sigma_y.tolist()
    z_obs_data = z_obs.tolist()
    z_absmag_data = z_absmag.tolist()          # [COLOR]
    sigma_z_absmag_data = sigma_z_absmag.tolist()  # [COLOR]
    if g_absmag is not None:
        g_absmag_data = g_absmag.tolist()          # [2COLOR]
        sigma_g_absmag_data = sigma_g_absmag.tolist()  # [2COLOR]

    N_total = len(x)

    # ============================================================================
    # SECTION 3: CREATE STAN DATA DICTIONARY
    # ============================================================================
    N_bins = 1

    mu_y_TF = float(np.mean(y)) if N_total > 0 else 0.0
    tau = 1.5 * float(np.std(y, ddof=1)) if N_total > 1 else 1.0

    stan_data = {
        "N_bins": N_bins,
        "N_total": N_total,
        "x": x_data,
        "sigma_x": sigma_x_data,
        "y": y_data,
        "sigma_y": sigma_y_data,
        "haty_max": float(haty_max),
        "haty_min": float(haty_min),
        "y_min": float(haty_min) - 0.5,
        "y_max": float(haty_max) + 0.5,
        "mu_y_TF": mu_y_TF,
        "tau": tau,
        "z_obs": z_obs_data,  # now defined, aligned, and JSON‑serializable
        **( {"z_obs_min": float(z_obs_min)} if z_obs_min is not None else {}),
        **( {"z_obs_max": float(z_obs_max)} if z_obs_max is not None else {}),

        # [COLOR] z-band absolute magnitudes for color.stan
        "z": z_absmag_data,
        "sigma_z": sigma_z_absmag_data,
        "c_bar_obs": float(np.mean(y - z_absmag)) if N_total > 0 else 0.0,
    }

    # [2COLOR] g-band fields for 2color.stan
    if g_absmag is not None:
        stan_data["g"] = g_absmag_data
        stan_data["sigma_g"] = sigma_g_absmag_data
        stan_data["c_bar_g_obs"] = float(np.mean(y - g_absmag)) if N_total > 0 else 0.0

    # [SPLIT] Partition provenance. These are what the staleness check below
    # compares against, so they must round-trip through input.json -- previously
    # they were never written, so _old_partition read back as all-None and the
    # warning fired on every regeneration regardless of whether anything changed.
    #
    # Stan ignores *numeric* JSON keys that are not declared data variables, but
    # it REJECTS string values outright ("Variable: <name>, error: string values
    # not allowed"), which kills step 6. So nothing written here may be a string
    # -- record paths and other text in output/<run>/config.json instead.
    stan_data["n_objects"] = n_objects
    stan_data["random_seed"] = random_seed
    if train_fraction is not None:
        stan_data["train_fraction"] = train_fraction

    # [SPLIT] record training galaxy IDs so color_predict.py can identify holdout
    if train_sga_ids is not None:
        stan_data["train_sga_ids"] = train_sga_ids
        if target_main_count is not None:
            stan_data["subset_sga_ids"] = subset_sga_ids
            # Provenance for reproducible reconstruction of the subsample. The
            # authoritative record is subset_sga_ids itself (reconstruction
            # needs no RNG replay); these fields let the draw be regenerated
            # from scratch and audited via --verify_subset.
            #
            # Numeric only. `fits_file` deliberately is NOT written here: a
            # string value makes CmdStan reject the whole data file with
            # "Variable: fits_file, error: string values not allowed", which
            # kills step 6 for every target_main_count run. --verify_subset
            # falls back to output/<run>/config.json, which records the path.
            stan_data["target_main_count"] = target_main_count
            stan_data["N_after_cuts"] = int(N_after_cuts)
            print(f"  Train/holdout split: {len(train_sga_ids)} training, "
                  f"{len(subset_sga_ids) - len(train_sga_ids)} holdout within subset")
        else:
            print(f"  Train/holdout split: {len(train_sga_ids)} training, "
                  f"{N_after_cuts - len(train_sga_ids)} in same z-range not selected")

    if plane_cut:
        stan_data["slope_plane"] = float(slope_plane)
        stan_data["intercept_plane"] = float(intercept_plane)
        if two_sided:
            stan_data["intercept_plane2"] = float(intercept_plane2)

    # [SPLIT] Warn loudly if this overwrites an input.json fit to a different
    # partition — otherwise stale MCMC chains fit under the old partition can
    # keep being used silently by downstream steps (color_predict.py etc.).
    if os.path.exists(data_output_file):
        with open(data_output_file) as f:
            _old = json.load(f)
        _partition_keys = ["target_main_count", "n_objects", "random_seed",
                           "train_fraction"]
        _old_partition = {k: _old.get(k) for k in _partition_keys}
        _new_partition = {
            "target_main_count": target_main_count,
            "n_objects": n_objects, "random_seed": random_seed,
            "train_fraction": train_fraction,
        }
        if _old_partition != _new_partition:
            print(f"  WARNING: overwriting {data_output_file} whose partition "
                  f"metadata differs from this run: old={_old_partition} "
                  f"new={_new_partition}. Any existing MCMC chains/init/metric "
                  f"in this run dir were fit to the OLD partition and must be "
                  f"regenerated (step6; also step5d if this config has no "
                  f"fixed_init) before further use.")

    # Guard: CmdStan rejects a data file containing ANY string value, with
    # "Variable: <name>, error: string values not allowed", and it does so at
    # step 6 -- hours after step 4 wrote the file, and only for the runs whose
    # extra provenance happened to include text. Fail here instead, where the
    # offending key is obvious. (This caught `fits_file`, which broke step 6 for
    # every target_main_count run.)
    _str_keys = sorted(k for k, v in stan_data.items() if isinstance(v, str))
    if _str_keys:
        sys.exit(
            f"ERROR: refusing to write {data_output_file}: CmdStan rejects "
            f"string values in a data file, so these keys would make step 6 "
            f"fail: {_str_keys}. Record text in output/<run>/config.json "
            f"instead."
        )

    with open(data_output_file, "w") as f:
        json.dump(stan_data, f, indent=2)

    # ============================================================================
    # SECTION 4: CALCULATE STANDARDIZATION AND LINEAR REGRESSION
    # ============================================================================
    fixed_init_data = None
    if N_total > 0:
        mean_x = np.mean(x)
        sd_x = np.std(x, ddof=1)

        x_std = (x - mean_x) / sd_x

        if fixed_init is not None:
            # Unit conversion only, not a re-fit: slope_orig/intercept_orig are
            # frozen, data-independent physical-unit values (see fixed_init
            # file); slope_std/intercept_std are this run's own local
            # standardized-coordinate representation of that same fixed line,
            # derived via the exact inverse of the transform below. Bound-safe
            # by construction — see 2color.stan's slope_std/intercept_std
            # bounds, which reduce to slope_orig in [-9,-4] and intercept_orig
            # in [-24,-14] regardless of sd_x/mean_x.
            with open(fixed_init) as f:
                fixed_init_data = json.load(f)
            slope_orig = fixed_init_data["slope_orig"]
            intercept_orig = fixed_init_data["intercept_orig"]
            slope_std = slope_orig * sd_x
            intercept_std = intercept_orig + slope_orig * mean_x
        else:
            slope_std, intercept_std = np.polyfit(x_std, y, deg=1)
            slope_orig = slope_std / sd_x
            intercept_orig = intercept_std - slope_std * mean_x / sd_x
        intercept_std_vec = [float(intercept_std)]
    else:
        slope_std = 0.0
        intercept_std = 0.0
        intercept_std_vec = [0.0]
        slope_orig = 0.0
        intercept_orig = 0.0
        mean_x = 0.0
        sd_x = 1.0
        x_std = np.array([])

    # ============================================================================
    # SECTION 5: CREATE INITIAL CONDITIONS DICTIONARY
    # ============================================================================
    init_data = {
        "slope_std": float(slope_std),
        "intercept_std": intercept_std_vec,
        "slope_orig": float(slope_orig),
        "intercept_orig": float(intercept_orig),
        "sigma_int_x": 0.1,
        "sigma_int_y": 0.1,
        "log_sigma_int_z": -2.3,
        "mean_x": float(mean_x),
        "sd_x": float(sd_x),
        "alpha_kcorr_r": -0.5,
        # [COLOR] color model init parameters
        "gamma_tau_c": -0.19,
        "delta_c": 0.065,
        "mu_c": float(np.mean(y - z_absmag)) if N_total > 0 else 0.3,
        "log_tau_c": -3.3,
        "alpha_kcorr_z": -0.5,
    }

    # [2COLOR] g-band init parameters.
    # The 2color model (2color.stan) uses a rank-1 intrinsic covariance
    # S = w w^T over (y,z,g) with a single loading vector w (|w| ~ 0.35 mag).
    # Start from a moderate isotropic loading; step5d MAP optimization refines it.
    if g_absmag is not None:
        init_data["delta_g"] = 0.0
        init_data["mu_g"] = float(np.mean(y - g_absmag)) if N_total > 0 else 0.0
        init_data["alpha_kcorr_g"] = -0.5
        # Asymmetric start: a perfectly symmetric w = [c,c,c] gives a degenerate
        # gradient for S = w wᵀ that can make the MAP optimizer report a
        # non-finite gradient and stall (seen on the spiral population).
        init_data["w"] = [0.15, 0.20, 0.25]

    if fixed_init_data is not None:
        # Overlay every other frozen physical-unit key verbatim (all
        # unconstrained/fixed-range in 2color.stan, so no per-run bound risk).
        # slope_orig/intercept_orig are excluded: already consumed above to
        # derive this run's own slope_std/intercept_std.
        for k, v in fixed_init_data.items():
            if k in ("slope_orig", "intercept_orig"):
                continue
            init_data[k] = v
        print(
            f"  fixed_init: {fixed_init} -> slope_std={slope_std:.6g} "
            f"intercept_std={intercept_std:.6g} (from slope_orig={slope_orig:.6g} "
            f"intercept_orig={intercept_orig:.6g}, mean_x={mean_x:.6g}, sd_x={sd_x:.6g})"
        )

    with open(init_output_file, "w") as f:
        json.dump(init_data, f, indent=2)

    if fixed_init_data is not None:
        map_init_output_file = os.path.join(
            os.path.dirname(init_output_file), "init_MAP.json"
        )
        with open(map_init_output_file, "w") as f:
            json.dump(init_data, f, indent=2)
        print(f"MAP-quality init file (from fixed_init): {map_init_output_file}")

    # ============================================================================
    # SECTION 6: PRINT SUMMARY STATISTICS
    # ============================================================================
    print("\nData conversion complete!")
    print(f"Stan data output file: {data_output_file}")
    print(f"Initial conditions output file: {init_output_file}")

    print("\nFiltering:")
    print(f"  Total rows in FITS: {total_rows}")
    print(f"  Valid rows (finite, positive velocities, finite z): {valid_rows}")
    print(f"  Rows with {haty_min} < y < {haty_max}: {y_filtered_rows}")

    if z_obs_min is not None:
        print(f"  Rows with z_obs > {z_obs_min}: {z_min_filtered_rows}")
    if z_obs_max is not None:
        print(f"  Rows with z_obs <= {z_obs_max}: {z_filtered_rows}")


    if plane_cut:
        if not two_sided:
            print(
                f"  Rows passing plane cut (max({haty_min}, bar_s*x+c1) < y): {plane_pass_rows}"
            )
            print(
                f"  Rows filtered out by plane cut: {z_filtered_rows - plane_pass_rows}"
            )
            print(f"  Plane parameters: bar_s = {slope_plane}, c1 = {intercept_plane}")
        else:
            print(f"  Rows passing two‑sided plane cut: {plane_pass_rows}")
            print(
                f"  Rows filtered out by plane cut: {z_filtered_rows - plane_pass_rows}"
            )
            print(
                f"  Plane parameters: bar_s = {slope_plane}, c1 = {intercept_plane}, c2 = {intercept_plane2}"
            )

    print(f"  Rows filtered out (by y cut only): {valid_rows - y_filtered_rows}")
    print(f"  haty_max (selection upper threshold): {haty_max}")
    print(f"  haty_min (selection lower threshold): {haty_min}")

    print("\nSummary:")
    print(f"  Number of redshift bins: {N_bins}")
    print(f"  Final sample size: {N_total}")
    if N_total > 0:
        print(f"  z_obs range: [{np.min(z_obs):.6f}, {np.max(z_obs):.6f}]")

    # Return data for plotting (plot doesn't use z, but we return it for completeness)
    return (
        x_all,
        y_all,
        sigma_x_all,
        sigma_y_all,
        zobs_all,
        x,
        y,
        sigma_x,
        sigma_y,
        z_obs,
        total_rows,
        N_total,
    )


def plot_desi_tf_data(
    x_all,
    y_all,
    sigma_x_all,
    sigma_y_all,
    x_selected,
    y_selected,
    sigma_x_selected,
    sigma_y_selected,
    haty_max=None,
    haty_min=None,
    slope_plane=None,
    intercept_plane=None,
    intercept_plane2=None,
    output_file="desi_tf_scatter_plot.png",
):
    """
    Create scatter plot showing complete sample (low alpha) and selected sample (high alpha).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot complete sample with low alpha
    ax.errorbar(
        x_all,
        y_all,
        xerr=sigma_x_all,
        yerr=sigma_y_all,
        fmt="o",
        markersize=2,
        alpha=0.2,
        color="gray",
        elinewidth=0.3,
        capsize=0,
        label=f"Complete sample (N = {len(x_all)})",
    )

    # Plot selected sample with high alpha
    ax.errorbar(
        x_selected,
        y_selected,
        xerr=sigma_x_selected,
        yerr=sigma_y_selected,
        fmt="o",
        markersize=3,
        alpha=0.8,
        color="blue",
        elinewidth=0.5,
        capsize=0,
        label=f"Selected sample (N = {len(x_selected)})",
    )

    if haty_max is not None:
        ax.axhline(
            y=haty_max,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"$\\hat{{y}}_{{\\rm max}}$ = {haty_max}",
        )
    if haty_min is not None:
        ax.axhline(
            y=haty_min,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"$\\hat{{y}}_{{\\rm min}}$ = {haty_min}",
        )

    # Plot one or two parallel plane‑cut boundaries if present
    if slope_plane is not None and intercept_plane is not None and len(x_all) > 0:
        x_range = np.array([np.min(x_all) - 0.1, np.max(x_all) + 0.1])

        y_plane1 = slope_plane * x_range + intercept_plane
        ax.plot(
            x_range,
            y_plane1,
            "g--",
            linewidth=2,
            alpha=0.8,
            label=f"Plane cut 1: y = {slope_plane:.1f}x + {intercept_plane:.1f}",
        )

        if intercept_plane2 is not None:
            y_plane2 = slope_plane * x_range + intercept_plane2
            ax.plot(
                x_range,
                y_plane2,
                "g-.",
                linewidth=2,
                alpha=0.8,
                label=f"Plane cut 2: y = {slope_plane:.1f}x + {intercept_plane2:.1f}",
            )

    ax.set_xlabel(r"$\hat{x}$ = log($V_{0.4R26}/V_0$)", fontsize=12)
    ax.set_ylabel(
        r"$\hat{y}$ = $R\_ABSMAG\_SB26$ (absolute magnitude)",
        fontsize=12,
    )
    # ax.set_title("DESI DR1 Tully‑Fisher Data", fontsize=14, fontweight="bold")

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {output_file}")

    print("\nPlot summary:")
    print(f"  Complete sample: {len(x_all)} galaxies")
    print(f"  Selected sample: {len(x_selected)} galaxies")
    if len(x_all) > 0:
        print(f"  Complete x range: [{np.min(x_all):.3f}, {np.max(x_all):.3f}]")
        print(f"  Complete y range: [{np.min(y_all):.3f}, {np.max(y_all):.3f}]")
    if len(x_selected) > 0:
        print(
            f"  Selected x range: [{np.min(x_selected):.3f}, {np.max(x_selected):.3f}]"
        )
        print(
            f"  Selected y range: [{np.min(y_selected):.3f}, {np.max(y_selected):.3f}]"
        )


def verify_subset(run_dir):
    """Drift check for a ``target_main_count`` run: regenerate the subsample draw
    from the provenance in ``run_dir`` and compare it to the recorded IDs.

    The authoritative record of a subsample is ``subset_sga_ids`` in
    ``input.json`` — reconstruction never needs an RNG replay. This check exists
    because *regeneration* does: NumPy does not promise ``Generator`` stream
    stability across versions, so if a future numpy (or a change to the selection
    code) alters the draw, this reports it instead of letting the two silently
    diverge.

    Regeneration goes through ``process_desi_tf_data`` itself rather than a
    reimplementation of the draw, so the check cannot pass while the real code
    path has drifted. Returns a process exit status (0 = match).
    """
    import tempfile

    input_path = os.path.join(run_dir, "input.json")
    config_path = os.path.join(run_dir, "config.json")
    for p in (input_path, config_path):
        if not os.path.exists(p):
            print(f"ERROR: {p} not found — cannot verify {run_dir}")
            return 1
    with open(input_path) as f:
        recorded = json.load(f)
    with open(config_path) as f:
        cfg = json.load(f)

    if "subset_sga_ids" not in recorded:
        print(f"{input_path} has no subset_sga_ids: this run drew no fixed-size "
              f"subsample, so there is nothing to verify.")
        return 0

    target = recorded.get("target_main_count", cfg.get("target_main_count"))
    fits_file = recorded.get("fits_file", cfg.get("fits_file"))
    seed = recorded.get("random_seed", cfg.get("random_seed"))
    print(f"Verifying {run_dir}")
    print(f"  fits_file         : {fits_file}")
    print(f"  target_main_count : {target}")
    print(f"  random_seed       : {seed}")
    print(f"  N_after_cuts      : {recorded.get('N_after_cuts')}")

    with tempfile.TemporaryDirectory() as tmp:
        process_desi_tf_data(
            fits_file,
            os.path.join(tmp, "input.json"),
            os.path.join(tmp, "init.json"),
            haty_max=cfg.get("haty_max"),
            haty_min=cfg.get("haty_min"),
            plane_cut=True,
            slope_plane=cfg.get("slope_plane"),
            intercept_plane=cfg.get("intercept_plane"),
            intercept_plane2=cfg.get("intercept_plane2"),
            n_objects=cfg.get("n_objects"),
            train_fraction=cfg.get("train_fraction"),
            random_seed=seed,
            target_main_count=target,
            source=cfg.get("source"),
            z_obs_min=cfg.get("z_obs_min"),
            z_obs_max=cfg.get("z_obs_max"),
            fixed_init=cfg.get("fixed_init"),
        )
        with open(os.path.join(tmp, "input.json")) as f:
            regenerated = json.load(f)

    status = 0
    for key in ("subset_sga_ids", "train_sga_ids"):
        old = recorded.get(key)
        new = regenerated.get(key)
        if old == new:
            print(f"  MATCH    {key}: {len(old) if old is not None else 0} ids identical")
        else:
            n_old = len(old) if old is not None else 0
            n_new = len(new) if new is not None else 0
            n_common = len(set(old or []) & set(new or []))
            print(f"  MISMATCH {key}: recorded {n_old} ids, regenerated {n_new}, "
                  f"{n_common} in common")
            status = 1
    if status == 0:
        print("  -> subsample regenerates exactly; no drift.")
    else:
        print("  -> DRIFT: the recorded subset_sga_ids in input.json remain "
              "authoritative; do not regenerate this run's input.json without "
              "re-running the downstream chain.")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DESI TF data for Stan.")
    parser.add_argument(
        "--config", default=None,
        help="Path to JSON config (e.g. configs/dr1_v3.json)",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Run name; outputs go to output/<run>/ with standard filenames",
    )
    parser.add_argument(
        "--input", default=None, help="Input FITS file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output data JSON file (default: DESI_input.json or output/<run>/input.json)",
    )
    parser.add_argument(
        "--init",
        default=None,
        help="Output init JSON file (default: DESI_init.json or output/<run>/init.json)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Output plot PNG file (default: DESI_input.png or output/<run>/data.png)",
    )
    parser.add_argument(
        "--haty_max",
        type=float,
        default=None,
        help="Upper apparent magnitude selection limit",
    )
    parser.add_argument(
        "--haty_min",
        type=float,
        default=None,
        help="Lower apparent magnitude selection limit",
    )
    # default=None is required for config_utils.apply_config to fill these from
    # --config (it only fills slots that are still None). They previously carried
    # hard defaults, which silently shadowed the "n_objects"/"random_seed" keys in
    # every config file. The documented defaults are applied after apply_config.
    parser.add_argument(
        "--n_objects", type=int, default=None,
        help=f"Training sample size (default {_DEFAULT_N_OBJECTS}; "
             f"mutually exclusive with --train_fraction)",
    )
    parser.add_argument(
        "--train_fraction", type=float, default=None,
        help="Training sample size as a fraction of the post-selection count, e.g. 0.4. "
             "Use instead of --n_objects when the selection is re-derived and an "
             "absolute count would stop meaning the intended fraction.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help=f"Random seed for reproducible subsampling (default {_DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--source", default=None,
        help="Data origin: 'DESI' (real catalogs) or 'fullmocks' (AbacusSummit "
             "mocks). For 'fullmocks', step 4 restricts to MAIN rows so the "
             "frozen selection cuts are applied to the population they were "
             "derived from. Default: from --config, else 'DESI'.",
    )
    parser.add_argument(
        "--target_main_count", type=int, default=None,
        help="Draw exactly this many galaxies from the post-selection-cut "
             "sample as the analysed subsample (they become MAIN). Used by the "
             "mock batch to match the real DR2 MAIN count. Seeded by "
             "--random_seed; the drawn IDs are recorded in input.json.",
    )
    parser.add_argument(
        "--verify_subset", metavar="RUN_DIR", default=None,
        help="Drift check: regenerate the subsample draw from the provenance "
             "recorded in RUN_DIR/input.json and RUN_DIR/config.json and compare "
             "it to the recorded subset_sga_ids/train_sga_ids. Exits 0 on a "
             "match, 1 on a mismatch. Does not modify RUN_DIR.",
    )
    parser.add_argument(
        "--z_obs_min", type=float, default=None, help="Minimum redshift"
    )
    parser.add_argument("--z_obs_max", type=float, default=None, help="Maximum redshift")
    parser.add_argument(
        "--slope_plane", type=float, default=None, help="Slope of oblique selection cut"
    )
    parser.add_argument(
        "--intercept_plane",
        type=float,
        default=None,
        help="Intercept of lower oblique cut (c1)",
    )
    parser.add_argument(
        "--intercept_plane2",
        type=float,
        default=None,
        help="Intercept of upper oblique cut (c2)",
    )
    parser.add_argument(
        "--fixed_init",
        type=str,
        default=None,
        help="Path to a JSON file of fixed physical-unit init values "
             "(slope_orig, intercept_orig, sigma_int_x, w, ...). When set, "
             "skips the per-run np.polyfit regression: mean_x/sd_x are still "
             "computed fresh from this run's own training x, but "
             "slope_std/intercept_std are derived from the fixed "
             "slope_orig/intercept_orig via the exact inverse of the usual "
             "transform. Also writes output/<run>/init_MAP.json directly, "
             "skipping the need for step5d's MAP optimization.",
    )

    args = parser.parse_args()

    if args.verify_subset is not None:
        sys.exit(verify_subset(args.verify_subset))

    from config_utils import apply_config
    cfg = apply_config(args)
    # Applied after apply_config so a config's "random_seed" still wins over it.
    # n_objects is deliberately left as None here: process_desi_tf_data needs to
    # tell "unset" from "explicitly set" to resolve --train_fraction precedence.
    if args.random_seed is None:
        args.random_seed = _DEFAULT_RANDOM_SEED
    if cfg.get("fits_file") and not args.input:
        args.input = cfg["fits_file"]
    if cfg.get("run") and not args.run:
        args.run = cfg["run"]

    # Fall back to a safe default for input FITS if nothing was provided
    if not args.input:
        args.input = "data/DESI-DR1_TF_pv_cat_v15.fits"

    run_dir: str | None = None
    config: dict = {}

    if args.run is not None:
        run_dir = os.path.join("output", args.run)
        os.makedirs(run_dir, exist_ok=True)
        output_json = args.output or os.path.join(run_dir, "input.json")
        init_json = args.init or os.path.join(run_dir, "init.json")
        plot_file = args.plot or os.path.join(run_dir, "data.png")
        config = {
            "fits_file": args.input,
            "source": args.source,
            "haty_max": args.haty_max,
            "haty_min": args.haty_min,
            "n_objects": args.n_objects,
            "train_fraction": args.train_fraction,
            "random_seed": args.random_seed,
            # Recorded so --verify_subset can regenerate the draw from the run
            # dir alone, without the original pipeline config.
            "target_main_count": args.target_main_count,
            "z_obs_min": args.z_obs_min,
            "z_obs_max": args.z_obs_max,
            "slope_plane": args.slope_plane,
            "intercept_plane": args.intercept_plane,
            "intercept_plane2": args.intercept_plane2,
            "fixed_init": args.fixed_init,
            # Propagated so color_predict.py can see it: that script builds its
            # cfg from output/<run>/config.json (this dict), not from --config,
            # so a "dust_pickle" set only in the pipeline config never reached
            # the covariance code and the DR2 runs silently used the iron
            # default d_err_r. See color_predict.resolve_d_err_r().
            "dust_pickle": cfg.get("dust_pickle"),

        }
    else:
        output_json = args.output or "DESI_input.json"
        init_json = args.init or "DESI_init.json"
        plot_file = args.plot or "DESI_input.png"

    # Process data and get both complete and selected samples
    (
        x_all,
        y_all,
        sigma_x_all,
        sigma_y_all,
        z_all,
        x_sel,
        y_sel,
        sigma_x_sel,
        sigma_y_sel,
        z_sel,
        n_total_fits,
        n_training,
    ) = process_desi_tf_data(
        args.input,
        output_json,
        init_json,
        haty_max=args.haty_max,
        haty_min=args.haty_min,
        plane_cut=True,
        slope_plane=args.slope_plane,
        intercept_plane=args.intercept_plane,
        intercept_plane2=args.intercept_plane2,
        n_objects=args.n_objects,
        train_fraction=args.train_fraction,
        random_seed=args.random_seed,
        target_main_count=args.target_main_count,
        source=args.source,
        z_obs_min=args.z_obs_min,
        z_obs_max=args.z_obs_max,
        fixed_init=args.fixed_init,

    )

    if run_dir is not None:
        _rd: str = run_dir
        config["n_total_fits"] = n_total_fits
        config["n_training"] = n_training
        _config_path = os.path.join(_rd, "config.json")
        with open(_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config written to {_config_path}")

    # Create plot showing both complete and selected samples
    plot_desi_tf_data(
        x_all,
        y_all,
        sigma_x_all,
        sigma_y_all,
        x_sel,
        y_sel,
        sigma_x_sel,
        sigma_y_sel,
        haty_max=args.haty_max,
        haty_min=args.haty_min,
        slope_plane=args.slope_plane,
        intercept_plane=args.intercept_plane,
        intercept_plane2=args.intercept_plane2,
        output_file=plot_file,
    )
