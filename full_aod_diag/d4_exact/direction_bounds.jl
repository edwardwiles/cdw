# ============================================================================
# Direction-aware outer gamma'_focal (gp) box constraints (addendum, task §"correct
# the outer gamma bounds for upper and lower runs").
#
# REMEDIATION UPDATE (live production-run finding, 2026-07-22): `direction_gamma_bounds` and
# `validate_gp_in_direction_box` are NO LONGER called by c10_d20_production_driver.jl's
# run_profile_checkpointed/run_polish_checkpointed (or c18_short_trajectory_comparison.jl) to
# construct the outer KNITRO gp box or to gate a start point. `frechet_benchmark_gp(ctx)` is
# exactly the model's own calibration gp value, i.e. exactly the value the natural start point
# (g_start) normally IS -- splitting the box there put the start point on the box's own boundary,
# forcing KNITRO's interior-point/barrier presolve to shift away from it before it could even
# evaluate; live-confirmed that Delta* is extremely sensitive to gp near calibration (a 1%
# deviation inflates Delta* from ~0.0026 to ~0.13, a ~50x jump), so the forced shift alone spikes
# the constraint violation and produces spurious "could not evaluate objective/constraints at
# initial point" warnings/stalls before real solving happens. There was no real correctness
# upside to the split either: the objective gradient in gp has an unambiguous sign, so the
# solver was never at risk of wandering into the "wrong" direction regardless of the box.
# Both functions are KEPT (used only by test_direction_bounds.jl and
# staged_delta5_realdata_validation.jl's informational logging) since deleting them would need
# touching those non-production callers for no behavioral benefit -- but do not wire either
# function back into a production KNITRO box/gate without re-litigating this finding.
#
# ============================================================================
# THE TRANSFORMATION AND ITS DIRECTION (addendum's audit items 1-2)
# ============================================================================
# theoretical_gammaprime_bounds (moments_gammanorm.jl, pre-existing, UNCHANGED by this
# file) documents, in its own docstring:
#
#   (kappa_min, kappa_max) = (0, 1 - lambda_dd^(1/(sigma-1)));
#   implied gp bounds (lambda_dd^(1/sigma), 1)
#
# i.e. kappa = 1 - gp^(sigma/(sigma-1)), with kappa=0 at gp=gp_hi=1 and
# kappa=kappa_max at gp=gp_lo=lambda_dd^(1/sigma). For sigma>1 (the standard CES
# case; sigma~2.5-2.9 in this repo's real calibrations), d(kappa)/d(gp) =
# -(sigma/(sigma-1))*gp^(sigma/(sigma-1)-1) < 0 for gp>0: **kappa is STRICTLY
# DECREASING in gp**. Smaller gp -> larger kappa.
#
# ============================================================================
# WHICH find_smallest VALUE IS "UPPER" (addendum's audit items 3-4)
# ============================================================================
# Larger kappa is, by this repo's own established and EXPLICITLY-CALIBRATED
# convention, "upper": run_d4_optimized_fd.jl's own comment reads
#   `const FIND_SMALLEST = DIRECTION == "upper"   # calibrated post-hoc against
#    which gives larger kappa`
# and the real D=20 canonical-rerun frontier script (c_canon_run_one.jl) hardcodes
#   `find_smallest = true   # kappa-UPPER-bound branch (minimize gp)`
# Both independently confirm: **find_smallest=true (minimize gp) is the real upper-
# kappa direction; find_smallest=false (maximize gp) is the real lower-kappa
# direction.** This is also independently confirmed by this repo's own real,
# validated, cross-checked numbers (Continuation 8's registered incumbents):
# "upper" kappa=0.17246 has gp=0.89264 (BELOW the Frechet benchmark g_F=0.96097);
# "lower" kappa=0.00439 has gp=0.99674 (ABOVE g_F) -- gp is smaller for the larger-
# kappa ("upper") result, exactly matching the decreasing-kappa-in-gp relationship
# above.
#
# **THIS IS THE OPPOSITE of what c10_d20_production_driver.jl's OWN
# `direction = find_smallest ? :lower : :upper` line (in both do_checkpoint
# closures) computes**, and the opposite of `run_staged_delta5_continuation`'s
# HARDCODED `run_polish_checkpointed(stage_label, false, g, zfree; ...)` (always
# find_smallest=false, i.e. always the real LOWER-kappa direction, regardless of
# which direction the caller actually wants). Both are real, substantive label/
# logic bugs, fixed in this commit (see staged_delta5.jl and
# c10_d20_production_driver.jl's own diffs) -- NOT merely a bounds-tightening
# exercise. This plausibly contributes to (though does not by itself fully
# reproduce, see the handoff doc's own caveat) the original task's reported
# kappa 0.0806->0.0031 staged-continuation pathology: a chain of calls ostensibly
# representing "upper" but silently always running find_smallest=false would
# systematically walk gp UP (toward gp_hi=1, kappa->0), the wrong direction for an
# upper-kappa search, regardless of the separate incumbent-seeding bug already
# fixed on this branch.
#
# **IMPORTANT DIVERGENCE FROM THE ADDENDUM'S OWN LITERAL BOX FORMULAS**, flagged
# explicitly rather than silently resolved: the addendum's own text states
# "Upper: gamma_f^F <= gp <= lambda_ff^(1/(sigma-1))" and "Lower: 0 <= gp <=
# gamma_f^F". Given the confirmed decreasing-kappa-in-gp relationship and the
# confirmed find_smallest<->upper/lower mapping above, this is backwards: the
# UPPER (larger-kappa) run needs SMALLER gp (below g_F, toward gp_lo), and the
# LOWER (smaller-kappa) run needs LARGER gp (above g_F, toward gp_hi=1) -- the
# reverse of the addendum's stated ranges. This file implements the EVIDENCED
# direction (matching real established multi-session results), not the addendum's
# literal formula. See docs/fullA_driver_delta5_diagnostics_handoff.md's gamma-
# bounds section for the full three-way evidence (algebra + real D4/D20 numbers +
# this repo's own explicit "calibrated post-hoc" code comments) and flag this
# prominently for the user's own review before any production merge.
#
# Also note: this file reuses ctx.bounds.γp_lo/γp_hi (= theoretical_gammaprime_
# bounds' own (lambda_dd^(1/sigma), 1)) as the OUTER endpoints rather than
# introducing a new lambda_ff^(1/(sigma-1)) formula -- that pre-existing formula
# is independently verified self-consistent here (kappa_max is exactly achieved
# at gp_lo by construction) and is what ctx/the production driver's box already
# uses everywhere; γ_f^F only need cut the box, not redefine its outer edges.
# ============================================================================

"γ_f^F: the Frechet-benchmark/calibration-implied focal gamma'_focal value, post-clamp -- i.e. what's actually stored in ctx and used everywhere else. Same quantity this repo's own gamma-profile investigation calls `g_F`/`gF_ctx` (c8_gammainterp_benchmark_check.jl)."
frechet_benchmark_gp(ctx) = ctx.θ0_up[3 + ctx.D]

"""
    direction_gamma_bounds(ctx, find_smallest::Bool) -> (lo::Float64, hi::Float64)

Closed-interval box for the outer gp variable, split at the Frechet benchmark
`frechet_benchmark_gp(ctx)`, in the EVIDENCED (not the addendum's literal) direction:
`find_smallest=true` ("upper", the real larger-kappa branch) gets
`[ctx.bounds.γp_lo, γ_f^F]`; `find_smallest=false` ("lower") gets
`[γ_f^F, ctx.bounds.γp_hi]`. Neither endpoint is at a formula singularity (γp_lo/γp_hi
are the model's own theoretical extremes, γ_f^F is an interior calibration point), so
no epsilon padding is needed or used -- both bounds are literal, closed values.
"""
function direction_gamma_bounds(ctx, find_smallest::Bool)
    gF = frechet_benchmark_gp(ctx)
    return find_smallest ? (ctx.bounds.γp_lo, gF) : (gF, ctx.bounds.γp_hi)
end

"""
    validate_gp_in_direction_box(gp, ctx, find_smallest::Bool; label::String="", what::String="point")

Hard-errors (does NOT clamp) if `gp` lies outside the correct direction-specific box.
Used to reject an incompatible starting point, resumed checkpoint, or fixed-g profile
target rather than silently coercing it into range -- per the task's explicit
"do not silently clamp" instruction. Closed-interval comparison (`<`/`>`, not `<=`/`>=`),
consistent with `direction_gamma_bounds` returning closed endpoints.
"""
function validate_gp_in_direction_box(gp::Real, ctx, find_smallest::Bool; label::String = "", what::String = "point")
    lo, hi = direction_gamma_bounds(ctx, find_smallest)
    (gp < lo || gp > hi) && error(
        "$(isempty(label) ? "" : "[$label] ")direction-box violation: $what has gp=$gp, but " *
        "find_smallest=$find_smallest (the $(find_smallest ? "upper" : "lower")-kappa direction) " *
        "requires gp in the CLOSED interval [$lo, $hi] (Frechet benchmark g_F=$(frechet_benchmark_gp(ctx)), " *
        "theoretical bounds gp_lo=$(ctx.bounds.γp_lo), gp_hi=$(ctx.bounds.γp_hi)). Refusing to silently " *
        "clamp -- pass an explicit migration/override option if this incompatibility is expected " *
        "(e.g. a pre-fix checkpoint or start point from before this direction-box fix).")
    return nothing
end
