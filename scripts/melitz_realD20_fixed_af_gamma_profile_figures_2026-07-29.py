#!/usr/bin/env python3
"""Figures for the 2026-07-29 real-D20 fixed-A/f gamma_d_prime profile campaign.
Reads results/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv (+ refinement/targets
CSVs) and writes three figures (PDF + PNG) to figures/.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
RESDIR = os.path.join(REPO, "results")
OUTDIR = os.path.join(REPO, "docs", "key_results")
FIGDIR = os.path.join(REPO, "figures")
os.makedirs(FIGDIR, exist_ok=True)

MAIN_CSV = os.path.join(RESDIR, "melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv")
REFINE_CSV = os.path.join(OUTDIR, "melitz_realD20_fixed_Af_gamma_profile_refinement_2026-07-29.csv")

df = pd.read_csv(MAIN_CSV)
df["classification"] = df["classification"].astype(str)
finite = df[df["classification"] == "FiniteSolved"].copy()
finite = finite.sort_values(["branch", "fraction_to_theoretical_endpoint"])

# Phase 4 refinement points (bounded, <=16 solves) -- plotted as markers only (not connected
# by the main corridor line, since they are targeted infill, not part of the predetermined
# grid), but genuinely useful to show real gaps in the raw grid (e.g. branch B has a real
# jump from Delta=0.39 to Delta=2.33 between two adjacent predetermined grid points).
refine = pd.read_csv(REFINE_CSV) if os.path.exists(REFINE_CSV) else pd.DataFrame()
if len(refine):
    refine = refine[refine["classification"] == "FiniteSolved"].copy()
    # kappa_ratio/gamma_d_prime/gains_from_trade_pct are now precomputed correctly (numeraire-
    # invariant kappa_ratio, gamma_d_prime = kappa_ratio^(sigma-1) under the natural wage
    # normalization) directly in the refinement CSV -- see 2026-07-29 correction.

branch_style = {
    "A_toward_min_gamma": dict(color="#2E6F9E", label="Branch A: calibration → theoretical min γ′"),
    "B_toward_max_gamma": dict(color="#C0562F", label="Branch B: calibration → theoretical max γ′"),
}

calib = finite[finite["grid_index"] == 0].iloc[0]

targets = [0.1, 0.5, 1.0, 2.0]

def nearest_rows_for_targets(sub):
    rows = []
    for t in targets:
        idx = (sub["delta_star"] - t).abs().idxmin()
        rows.append(sub.loc[idx])
    return rows

# theoretical endpoints (2026-07-29 correction: gamma_d_prime = kappa_ratio^(sigma-1) under the
# natural wage_ratio=1 normalization -- gamma_d_prime_min = lambda_dd EXACTLY, gamma_d_prime_max
# = 1 EXACTLY; GT values themselves were always numeraire-invariant and unaffected)
GT_ceiling_pct = 11.279224  # branch A open Delta->inf limit (gamma_d_prime -> lambda_dd = 0.835676)
GT_floor_pct = 0.0          # branch B ordinary finite theoretical point (gamma_d_prime -> 1)

# ============================================================================
# Figure 1: main economically relevant figure (Delta in [0, ~2.25])
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
for branch, style in branch_style.items():
    sub = finite[finite["branch"] == branch]
    subm = sub[sub["delta_star"] <= 2.25]
    ax.plot(subm["delta_star"], subm["gains_from_trade_pct"], "-o", ms=3.5, lw=1.4, **style)
    for r in nearest_rows_for_targets(sub):
        if r["delta_star"] <= 2.5:
            ax.annotate(f"{r['delta_star']:.2f}", (r["delta_star"], r["gains_from_trade_pct"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color=style["color"])

ax.plot(calib["delta_star"], calib["gains_from_trade_pct"], "k*", ms=14, zorder=5, label="Fréchet calibration")
if len(refine):
    for branch, style in branch_style.items():
        rsub = refine[refine["branch"] == branch]
        rsub = rsub[rsub["delta_star"] <= 2.25]
        if len(rsub):
            ax.scatter(rsub["delta_star"], rsub["gains_from_trade_pct"], marker="^", s=45,
                       facecolor=style["color"], edgecolor="k", linewidth=0.6, zorder=4,
                       label=f"{branch.split('_')[0]} refinement (Phase 4)")
ax.axhline(GT_ceiling_pct, color="#2E6F9E", ls=":", lw=1, alpha=0.6, label="theoretical GT max (open limit, branch A)")
ax.axhline(GT_floor_pct, color="#C0562F", ls=":", lw=1, alpha=0.6, label="theoretical GT min = 0% (branch B)")
ax.set_xlabel(r"$\Delta^*$")
ax.set_ylabel("Gains from trade, $100(1-\\kappa)$  (%)")
ax.set_title("Real D=20 fixed-A/f gamma profile: gains from trade vs. $\\Delta^*$\n(economically relevant region)")
ax.set_xlim(-0.05, 2.3)
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "melitz_realD20_fixed_Af_GT_vs_delta_main_2026-07-29.pdf"))
fig.savefig(os.path.join(FIGDIR, "melitz_realD20_fixed_Af_GT_vs_delta_main_2026-07-29.png"), dpi=200)
plt.close(fig)

# ============================================================================
# Figure 2: full verified finite profile, log-scaled Delta axis up to cap
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
for branch, style in branch_style.items():
    sub = finite[finite["branch"] == branch]
    sub = sub[sub["delta_star"] > 0]
    ax.plot(sub["delta_star"], sub["gains_from_trade_pct"], "-o", ms=3, lw=1.2, **style)
ax.plot(calib["delta_star"], calib["gains_from_trade_pct"], "k*", ms=14, zorder=5, label="Fréchet calibration")
if len(refine):
    for branch, style in branch_style.items():
        rsub = refine[refine["branch"] == branch]
        if len(rsub):
            ax.scatter(rsub["delta_star"], rsub["gains_from_trade_pct"], marker="^", s=35,
                       facecolor=style["color"], edgecolor="k", linewidth=0.5, zorder=4)
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta^*$ (log scale)")
ax.set_ylabel("Gains from trade, $100(1-\\kappa)$  (%)")
ax.set_title("Real D=20 fixed-A/f gamma profile: full verified finite profile\n(log-scaled $\\Delta^*$, up to evaluation cap=10)")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "melitz_realD20_fixed_Af_GT_vs_delta_full_2026-07-29.pdf"))
fig.savefig(os.path.join(FIGDIR, "melitz_realD20_fixed_Af_GT_vs_delta_full_2026-07-29.png"), dpi=200)
plt.close(fig)

# ============================================================================
# Figure 3: diagnostic, gamma_d_prime vs Delta* (all classifications, colored)
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
class_colors = {"FiniteSolved": "#2E6F9E", "AboveEvaluationCap": "#E0A72E",
                 "InfiniteDeltaCertified": "#B23A48", "NumericalFailure": "#7F7F7F"}
for branch in ("A_toward_min_gamma", "B_toward_max_gamma"):
    sub = df[df["branch"] == branch].sort_values("fraction_to_theoretical_endpoint")
    subf = sub[sub["classification"] == "FiniteSolved"]
    ax.plot(subf["gamma_d_prime"], subf["delta_star"], "-", lw=1.0, color="0.6", zorder=1)
    for cls, color in class_colors.items():
        s = sub[sub["classification"] == cls]
        if len(s):
            y = s["delta_star"] if cls == "FiniteSolved" else [0]*len(s)
            ax.scatter(s["gamma_d_prime"], y, s=16, color=color, label=cls if branch == "A_toward_min_gamma" else None, zorder=2)
ax.axvline(calib["gamma_d_prime"], color="k", ls="--", lw=1, alpha=0.6, label="calibration")
ax.set_xlabel(r"$\gamma_d'$")
ax.set_ylabel(r"$\Delta^*$ (FiniteSolved only; others plotted at 0 for visibility)")
ax.set_title("Diagnostic: $\\Delta^*$ vs $\\gamma_d'$ across the fixed-A/f profile")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "melitz_realD20_fixed_Af_delta_vs_gamma_2026-07-29.pdf"))
fig.savefig(os.path.join(FIGDIR, "melitz_realD20_fixed_Af_delta_vs_gamma_2026-07-29.png"), dpi=200)
plt.close(fig)

print("Figures written to", FIGDIR)
for f in sorted(os.listdir(FIGDIR)):
    print(" ", f)
