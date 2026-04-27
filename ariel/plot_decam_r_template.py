"""
plot_decam_r_template.py

Plots the DECam r-band total system throughput (SVO Filter Profile Service)
as a shaded background, overlaid with a Kinney+1996 Sc spiral galaxy template
SED redshifted to z = 0, 0.02, 0.04, 0.06, 0.08, 0.10, each colored by
redshift using the 'plasma' colormap. H-alpha (6563 Å rest) is marked on
each SED curve with a small tick.

Data fetched at runtime:
  Filter:   http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=CTIO/DECam.r
  Template: https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/kc96/sc_template.fits
"""

import io
import urllib.request

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLOT_WAVE_MIN = 4000.0
PLOT_WAVE_MAX = 8000.0
REDSHIFTS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
HA_REST = 6563.0
CMAP_NAME = "plasma"
FILTER_URL = (
    "http://svo2.cab.inta-csic.es/theory/fps/getdata.php"
    "?format=ascii&id=CTIO/DECam.r"
)
TEMPLATE_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/"
    "cdbs/grid/kc96/sc_template.fits"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_decam_r() -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength_Ang, transmission) for DECam r total throughput."""
    with urllib.request.urlopen(FILTER_URL, timeout=15) as resp:
        lines = resp.read().decode().strip().splitlines()
    rows = [
        [float(v) for v in ln.split()]
        for ln in lines
        if not ln.startswith("#") and len(ln.split()) == 2
    ]
    data = np.array(rows)
    return data[:, 0], data[:, 1]


def load_kinney_sc() -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength_Ang, flux_flam) for the Kinney+1996 Sc template."""
    with urllib.request.urlopen(TEMPLATE_URL, timeout=20) as resp:
        raw = resp.read()
    hdul = fits.open(io.BytesIO(raw))
    wave = hdul[1].data["WAVELENGTH"].astype(float)
    flux = hdul[1].data["FLUX"].astype(float)
    hdul.close()
    return wave, flux


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------
def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def redshift_sed(
    wave_rest: np.ndarray,
    flux_rest: np.ndarray,
    z: float,
    wave_obs: np.ndarray,
) -> np.ndarray:
    """Redshift a rest-frame SED to z and resample onto wave_obs (NaN outside template range)."""
    wave_shifted = wave_rest * (1.0 + z)
    return np.interp(wave_obs, wave_shifted, flux_rest, left=np.nan, right=np.nan)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(
    filter_wave: np.ndarray,
    filter_trans: np.ndarray,
    sed_wave: np.ndarray,
    sed_flux: np.ndarray,
) -> None:
    wave_grid = np.linspace(PLOT_WAVE_MIN, PLOT_WAVE_MAX, 4000)

    trans_norm = np.interp(
        wave_grid, filter_wave, normalize(filter_trans), left=0.0, right=0.0
    )

    cmap = matplotlib.colormaps[CMAP_NAME]
    norm_z = mcolors.Normalize(vmin=REDSHIFTS[0], vmax=REDSHIFTS[-1])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter background
    ax.fill_between(wave_grid, trans_norm, color="steelblue", alpha=0.18,
                    label="DECam $r$ throughput", zorder=1)
    ax.plot(wave_grid, trans_norm, color="steelblue", lw=1.2, alpha=0.6, zorder=2)

    # Redshifted SEDs
    for z in REDSHIFTS:
        color = cmap(norm_z(z))
        flux_obs = redshift_sed(sed_wave, sed_flux, z, wave_grid)

        valid = np.isfinite(flux_obs)
        if valid.sum() == 0:
            continue
        flux_plot = flux_obs.copy()
        flux_plot[valid] = normalize(flux_obs[valid])

        ax.plot(wave_grid, flux_plot, color=color, lw=1.5, label=f"$z = {z:.2f}$", zorder=3)

        # H-alpha tick
        ha_obs = HA_REST * (1.0 + z)
        if PLOT_WAVE_MIN <= ha_obs <= PLOT_WAVE_MAX:
            ha_flux = float(np.interp(ha_obs, wave_grid[valid], flux_plot[valid]))
            ax.plot(ha_obs, ha_flux, marker="|", markersize=8, markeredgewidth=1.5,
                    color=color, zorder=4)

    # Reference H-alpha line
    ax.axvline(HA_REST, color="0.55", lw=0.8, ls="--", alpha=0.7, zorder=0)
    ax.text(HA_REST + 25, 0.97, r"H$\alpha$ (rest)", color="0.45", fontsize=8, va="top")

    ax.set_xlim(PLOT_WAVE_MIN, PLOT_WAVE_MAX)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"Observed Wavelength ($\AA$)", fontsize=12)
    ax.set_ylabel("Normalized Flux / Throughput", fontsize=12)
    ax.set_title(
        "DECam $r$-band throughput with Kinney+1996 Sc SED at $z = 0$–$0.10$",
        fontsize=12,
    )

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm_z)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Redshift $z$", fontsize=11)

    handles, labels_leg = ax.get_legend_handles_labels()
    filter_handle = [h for h, l in zip(handles, labels_leg) if "DECam" in l]
    if filter_handle:
        ax.legend(handles=filter_handle, labels=["DECam $r$ throughput"],
                  loc="upper left", fontsize=10)

    ax.grid(True, alpha=0.25, lw=0.6)
    plt.tight_layout()
    outfile = "decam_r_template.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading DECam r-band filter from SVO...")
    filter_wave, filter_trans = load_decam_r()
    print(f"  {len(filter_wave)} points, {filter_wave.min():.0f}–{filter_wave.max():.0f} Å, "
          f"peak T = {filter_trans.max():.4f}")

    print("Loading Kinney+1996 Sc template from MAST...")
    sed_wave, sed_flux = load_kinney_sc()
    print(f"  {len(sed_wave)} points, {sed_wave.min():.0f}–{sed_wave.max():.0f} Å")

    make_plot(filter_wave, filter_trans, sed_wave, sed_flux)


if __name__ == "__main__":
    main()
