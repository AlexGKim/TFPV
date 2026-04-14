"""Shared configuration loading for the TFR pipeline.

Usage in any pipeline script::

    parser.add_argument("--config", default=None,
        help="Path to JSON config (e.g. configs/dr1_v3.json)")
    # All config-overrideable args must use default=None in add_argument()
    args = parser.parse_args()
    cfg = apply_config(args)   # fills None slots from config, then PIPELINE_DEFAULTS
    # args.z_obs_min etc. are now guaranteed non-None

Design
------
Priority order (highest wins):
  1. Explicitly passed CLI flag  (left as-is by apply_config — it was already set)
  2. Value from --config JSON    (fills any None arg whose name matches a config key)
  3. PIPELINE_DEFAULTS fallback  (fills any None arg that is still None after step 2)

This replaces the per-script ``if "--xyz" not in sys.argv`` boilerplate pattern.
The trick is simple: every overrideable argparse argument must use ``default=None``.
"""

import json
import os

# ---------------------------------------------------------------------------
# Central defaults for all config-overrideable parameters.
# Each script's add_argument() must use default=None for these keys so that
# apply_config() can distinguish "user set this" from "argparse default".
# ---------------------------------------------------------------------------
PIPELINE_DEFAULTS: dict = {
    "source":      "DESI",     # data origin: "DESI", "fullmocks", "ariel"
    "exe":         "tophat",   # Stan binary name
    "model":       "tophat",   # posterior model: "tophat" or "normal"
    "n_sigma":     3.0,        # GMM ellipse sigma for selection_ellipse
    "n_sigma_perp": 3.0,       # perpendicular cut width (sigma units)
    "haty_min":    -23.0,      # loose pre-filter lower bound (selection_ellipse)
    "haty_max":    -18.0,      # loose pre-filter upper bound (selection_ellipse)
    "z_obs_min":   0.03,       # minimum redshift cut
    "z_obs_max":   0.1,        # maximum redshift cut
    "n_init":      20,         # GMM random restarts
    "n_bins":      20,         # pull-plot magnitude bins
}

# Required keys that must be present in any top-level pipeline config.
REQUIRED_KEYS: list[str] = [
    "run",
    "fits_file",
    "exe",
    "source",
    "model",
    "z_obs_min",
    "z_obs_max",
    "haty_min",
    "haty_max",
    "n_sigma_perp",
    "n_sigma",
]


def apply_config(args, config_path_attr: str = "config") -> dict:
    """Load a JSON config file and apply its values to *args* for any slot that
    is still ``None`` after argparse parsing.

    Parameters
    ----------
    args:
        The ``argparse.Namespace`` returned by ``parser.parse_args()``.
    config_path_attr:
        Name of the attribute on *args* that holds the config file path
        (default ``"config"``).

    Returns
    -------
    dict
        The raw config dict (possibly empty if no ``--config`` was given).
        Returned so callers can inspect keys that have no argparse counterpart
        (e.g. ``slope_plane``, ``intercept_plane``, ``intercept_plane2``).
    """
    cfg: dict = {}
    config_path = getattr(args, config_path_attr, None)
    if config_path:
        with open(config_path) as fh:
            cfg = json.load(fh)

    # Step 1: fill None args from config
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, value)

    # Step 2: fill remaining None args from PIPELINE_DEFAULTS
    for key, default in PIPELINE_DEFAULTS.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, default)

    return cfg


def run_dir_from_args(args) -> str | None:
    """Return ``output/<run>`` if ``args.run`` is set, else ``None``."""
    run = getattr(args, "run", None)
    return os.path.join("output", run) if run else None
