#!/usr/bin/env python
"""Reconstruct the selection function behind the Malmquist bias in the DR2 TF sample,
and express it as a proxy selection function in angular diameter alone.

Run against data/DR2_TF_disks_v5.fits.  Three steps:

1. BACKWARD RECONSTRUCTION OF THE MAGNITUDE LIMIT.
   For each redshift slice the faint edge of the absolute-magnitude distribution is
   measured and converted to an apparent limit m_lim(z) = M_lim(z) + <mu>(z).  A pure
   flux limit gives m_lim = const; the data give ~16.95 mag, flat to 0.3 mag.

2. WHY THAT IS A PURE ANGULAR-DIAMETER CUT.
   The r-band size-luminosity relation has slope -5.01 -- i.e. the mean surface
   brightness inside D26 is independent of size (Freeman's law).  Write

       SB == m_r + 5 log10(theta)          [theta in arcsec]

   which is distance-independent: SB = M_r + 5 log10(D) + const.  Then

       m_r < m_lim   <=>   theta > 10**((SB - m_lim)/5)

   exactly.  Since SB is Gaussian with scatter sigma_SB, the flux limit becomes a
   SMOOTH, REDSHIFT-INDEPENDENT selection function in angular diameter:

       S(theta) = 1{theta > theta_hard} * Phi( (5 log10 theta + m_lim - SB0) / sigma_SB )

   theta_hard = 20" is the SGA/DESI-PV target cut, which is a separate hard floor.

3. VALIDATION.
   S is a function of theta only, so pushing a galaxy to redshift z changes only
   theta = D/d_A(z).  The sample is de-selected by weighting each galaxy 1/S(theta_i)
   and re-selected at redshift z with S(D_i/d_A(z)), giving the predicted <M_r>(z)
   and <logV>(z).  These are compared with the observed drift.

Distances come from the catalog's own MU_ZCMB, so no cosmology is assumed.
"""
import argparse

import numpy as np
from astropy.io import fits
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCSEC_PER_RAD = 206264.806
C_OBS = "#1f6fb2"
C_MOD = "#d1670b"


def load(path, z_max):
    d = fits.open(path)[1].data
    z, mu = d["Z_DESI_CMB"], d["MU_ZCMB"]
    o = dict(z=z, mu=mu, theta=2.0 * d["SMA_SB26"],
             M=d["R_ABSMAG_SB26_CORR"], m=d["R_MAG_SB26_CORR"], logV=d["LOGVROT"])
    dL = 10.0 ** (mu / 5.0 - 5.0)
    o["dA"] = dL / (1.0 + z) ** 2
    o["D"] = o["theta"] / ARCSEC_PER_RAD * o["dA"] * 1e3
    ok = np.ones(len(z), bool)
    for k in o:
        ok &= np.isfinite(o[k])
    ok &= (z > 1e-3) & (z < z_max)
    return {k: v[ok] for k, v in o.items()}


def S_theta(theta, m_lim, SB0, sigma_SB, theta_hard):
    """Selection probability as a function of angular diameter (arcsec)."""
    theta = np.asarray(theta, float)
    s = norm.cdf((5.0 * np.log10(np.clip(theta, 1e-3, None)) + m_lim - SB0) / sigma_SB)
    return np.where(theta > theta_hard, s, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", default="data/DR2_TF_disks_v5.fits")
    p.add_argument("--theta-hard", type=float, default=20.0)
    p.add_argument("--z-max", type=float, default=0.065)
    p.add_argument("--faint-pct", type=float, default=95.0)
    p.add_argument("--s-floor", type=float, default=0.05,
                   help="drop galaxies whose own S is below this (unstable 1/S weight)")
    p.add_argument("--out", default="output/selection_proxy_d26.png")
    args = p.parse_args()

    d = load(args.parent, args.z_max)
    n = len(d["z"])

    # --- 1. size-luminosity slope and the surface-brightness invariant -------------
    x, y = np.log10(d["D"]), d["M"]
    B, A = np.polyfit(x, y, 1)
    SB = d["m"] + 5.0 * np.log10(d["theta"])
    SB0, sigma_SB = SB.mean(), SB.std()
    print(f"parent N={n}   theta min {d['theta'].min():.2f}\"")
    print(f"size-luminosity : M_r = {A:+.3f} {B:+.3f} log10(D/kpc)   (-5 => flux limit == size limit)")
    print(f"surface bright. : SB = m_r + 5log10(theta) = {SB0:.3f} +/- {sigma_SB:.3f} mag")

    # --- 2. backward-reconstruct the apparent magnitude limit ---------------------
    edges = np.linspace(0.005, args.z_max, 13)
    cen = 0.5 * (edges[:-1] + edges[1:])
    obs_M, obs_V, mlim_z, nb = [], [], [], []
    for i in range(len(cen)):
        s = (d["z"] >= edges[i]) & (d["z"] < edges[i + 1])
        nb.append(s.sum())
        obs_M.append(d["M"][s].mean()); obs_V.append(d["logV"][s].mean())
        mlim_z.append(np.percentile(d["M"][s], args.faint_pct) + d["mu"][s].mean())
    obs_M, obs_V, mlim_z = map(np.asarray, (obs_M, obs_V, mlim_z))
    m_lim = float(np.median(mlim_z[cen > 0.015]))
    print(f"apparent limit  : m_lim = {m_lim:.2f} mag "
          f"(range {mlim_z.min():.2f}-{mlim_z.max():.2f}, spread {mlim_z.max()-mlim_z.min():.2f})")

    theta_half = 10 ** ((SB0 - m_lim) / 5.0)
    print(f"\nPROXY  S(theta) = 1{{theta>{args.theta_hard:g}\"}} * Phi((5log10 theta"
          f" + {m_lim:.2f} - {SB0:.2f}) / {sigma_SB:.2f})")
    print(f"       50% completeness at theta = {theta_half:.1f}\"; "
          f"S(20\")={S_theta(20.001,m_lim,SB0,sigma_SB,args.theta_hard):.2f} "
          f"S(30\")={S_theta(30,m_lim,SB0,sigma_SB,args.theta_hard):.2f} "
          f"S(60\")={S_theta(60,m_lim,SB0,sigma_SB,args.theta_hard):.2f}")

    # --- 3. validate ------------------------------------------------------------
    # Recovering the intrinsic population needs an EFFECTIVE-VOLUME weight, not just
    # 1/S: a galaxy enters the catalog anywhere in the volume where S(theta(z))>0, and
    # that volume grows steeply with size.  Weighting by 1/S alone leaves the parent
    # dominated by the big distant galaxies and flattens the very drift under test.
    #   V_eff,i = Int S(D_i/d_A(z)) dV/dz dz      (generalised 1/Vmax)
    order = np.argsort(d["z"])
    z_grid = np.linspace(1e-3, args.z_max, 400)
    dA_grid = np.interp(z_grid, d["z"][order], d["dA"][order])
    dC_grid = dA_grid * (1.0 + z_grid)                       # comoving distance
    dVdz = np.gradient(dC_grid ** 3, z_grid)                 # prop. to comoving volume element

    th_grid = d["D"][:, None] / (dA_grid[None, :] * 1e3) * ARCSEC_PER_RAD   # (N, nz)
    S_grid = S_theta(th_grid, m_lim, SB0, sigma_SB, args.theta_hard)
    V_eff = np.trapezoid(S_grid * dVdz[None, :], z_grid, axis=1)
    use = V_eff > 0
    w0 = np.zeros(n); w0[use] = 1.0 / V_eff[use]
    print(f"effective volume: {use.sum()}/{n} galaxies with V_eff>0; "
          f"weight dynamic range {w0[use].max()/w0[use].min():.3g}")

    dA_cen = np.interp(cen, d["z"][order], d["dA"][order])
    mod_M, mod_V = [], []
    for i in range(len(cen)):
        th_at_z = d["D"] / (dA_cen[i] * 1e3) * ARCSEC_PER_RAD
        w = w0 * S_theta(th_at_z, m_lim, SB0, sigma_SB, args.theta_hard)
        mod_M.append(np.sum(w * d["M"]) / np.sum(w))
        mod_V.append(np.sum(w * d["logV"]) / np.sum(w))
    mod_M, mod_V = np.asarray(mod_M), np.asarray(mod_V)

    print(f"\n{'z':>6} {'N':>5} {'theta_lim':>9} {'<M>obs':>8} {'<M>prox':>8} {'d':>6}"
          f" {'<lV>obs':>8} {'<lV>prox':>8} {'m_lim(z)':>8}")
    for i in range(len(cen)):
        th_lim = theta_half  # z-independent by construction
        print(f"{cen[i]:6.3f} {nb[i]:5d} {th_lim:9.1f} {obs_M[i]:8.3f} {mod_M[i]:8.3f}"
              f" {mod_M[i]-obs_M[i]:+6.3f} {obs_V[i]:8.3f} {mod_V[i]:8.3f} {mlim_z[i]:8.2f}")

    for nm, o, mo, u in [("<M_r>", obs_M, mod_M, "mag"), ("<logV>", obs_V, mod_V, "dex")]:
        do, dm = o[-1] - o[0], mo[-1] - mo[0]
        print(f"drift {nm:7s} observed {do:+.3f} {u}   proxy {dm:+.3f} {u}   recovered {100*dm/do:5.1f}%")

    # --- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))

    ax = axes[0]
    Mlim_z = mlim_z - np.array([d["mu"][(d["z"] >= edges[i]) & (d["z"] < edges[i + 1])].mean()
                                for i in range(len(cen))])
    ax.plot(cen, mlim_z, "o-", color=C_OBS, lw=2, ms=5, label="apparent $m_{\\rm lim}$")
    ax.plot(cen, Mlim_z - Mlim_z.mean() + m_lim, "s--", color=C_MOD, lw=2, ms=4,
            label="absolute $M_{\\rm lim}$ (offset)")
    ax.axhline(m_lim, color="#9a9a96", ls=(0, (4, 3)), lw=1.2)
    ax.text(0.97, 0.06, f"$m_{{\\rm lim}}={m_lim:.2f}$, flat to {mlim_z.max()-mlim_z.min():.2f} mag",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#6b6b67")
    # same vertical scale as the absolute limit's excursion, so "flat" is not an artifact of zoom
    ax.set_ylim(m_lim - 3.4, m_lim + 3.4)
    ax.set_xlabel("redshift $z_{\\rm CMB}$")
    ax.set_ylabel(f"limiting magnitude at the {args.faint_pct:g}th pct edge")
    ax.set_title("1. reconstructed flux limit\n(flat $\\Rightarrow$ pure magnitude selection)",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    hb = ax.hexbin(np.log10(d["theta"]), d["m"], gridsize=45, cmap="Blues",
                   bins="log", mincnt=1, linewidths=0)
    tt = np.logspace(np.log10(20), np.log10(400), 50)
    ax.plot(np.log10(tt), SB0 - 5 * np.log10(tt), color=C_MOD, lw=2,
            label=f"$m_r = {SB0:.2f} - 5\\log_{{10}}\\theta$")
    ax.axhline(m_lim, color="#555", ls=(0, (4, 3)), lw=1.2)
    ax.set_xlabel("$\\log_{10}(\\theta\\,/\\,{\\rm arcsec})$"); ax.set_ylabel("$m_r$")
    ax.set_title(f"2. constant surface brightness\n$\\sigma_{{SB}}={sigma_SB:.2f}$ mag",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.invert_yaxis()

    ax = axes[2]
    tg = np.logspace(np.log10(15), np.log10(200), 300)
    ax.plot(tg, S_theta(tg, m_lim, SB0, sigma_SB, args.theta_hard), color=C_MOD, lw=2.2)
    ax.axvline(args.theta_hard, color="#9a9a96", ls=(0, (4, 3)), lw=1.2)
    ax.text(args.theta_hard * 1.04, 0.5, f" target cut {args.theta_hard:g}$''$",
            fontsize=8.5, color="#6b6b67", rotation=90, va="center")
    ax.axvline(theta_half, color="#555", ls=":", lw=1.2)
    ax.text(theta_half * 1.05, 0.08, f" 50% at {theta_half:.0f}$''$", fontsize=8.5, color="#555")
    ax.set_xscale("log"); ax.set_xlabel("angular diameter $\\theta$ [arcsec]")
    ax.set_ylabel("$S(\\theta)$"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("3. proxy selection function\n(redshift-independent)", fontsize=9.5, loc="left")

    ax = axes[3]
    ax.plot(cen, obs_M, "o-", color=C_OBS, lw=2, ms=5, label="observed")
    ax.plot(cen, mod_M, "s--", color=C_MOD, lw=2, ms=5, label="proxy $S(\\theta)$")
    ax.set_xlabel("redshift $z_{\\rm CMB}$"); ax.set_ylabel("$\\langle M_r\\rangle$ [mag]")
    ax.set_title("4. Malmquist bias reproduced", fontsize=9.5, loc="left")
    ax.invert_yaxis(); ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    for ax in axes:
        ax.grid(True, color="#e6e6e3", lw=0.7); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c8c8c4")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
