def get_mag_cols(names):
    """
    Returns (abs_mag_col, abs_mag_err_col, app_mag_col) based on self-consistent priority.
    Group 1: R_ABSMAG_SB26_CORR, R_ABSMAG_SB26_ERR_CORR, R_MAG_SB26_CORR
    Group 2: R_ABSMAG_SB26, R_ABSMAG_SB26_ERR, R_MAG_SB26
    """
    # Prefer Group 1 (corrected magnitudes and errors)
    if "R_ABSMAG_SB26_CORR" in names:
        if "R_ABSMAG_SB26_ERR_CORR" not in names:
            raise KeyError(
                "Found R_ABSMAG_SB26_CORR but missing R_ABSMAG_SB26_ERR_CORR."
            )
        if "R_MAG_SB26_CORR" not in names:
            raise KeyError("Found R_ABSMAG_SB26_CORR but missing R_MAG_SB26_CORR.")
        return "R_ABSMAG_SB26_CORR", "R_ABSMAG_SB26_ERR_CORR", "R_MAG_SB26_CORR"

    # Fallback to Group 2 (uncorrected)
    elif "R_ABSMAG_SB26" in names:
        if "R_ABSMAG_SB26_ERR" not in names:
            raise KeyError("Found R_ABSMAG_SB26 but missing R_ABSMAG_SB26_ERR.")
        if "R_MAG_SB26" not in names:
            raise KeyError("Found R_ABSMAG_SB26 but missing R_MAG_SB26.")
        return "R_ABSMAG_SB26", "R_ABSMAG_SB26_ERR", "R_MAG_SB26"

    else:
        raise KeyError(
            f"Required magnitude columns not found (neither Group 1 nor Group 2 present). "
            f"Names present: {list(names)}"
        )
