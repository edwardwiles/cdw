# Fixed-A_od*(=calibration) CM profile: how much of the CM-restricted upper bound comes from
# letting A_od move at all, vs. just the aggregate wedge gamma'_focal? For each nested grid L,
# fix zfree=0 (A_od=1 everywhere, the calibration level) and scan gamma'_focal only, computing
# Delta_dual via the SAME production bundle (Architecture B moments + Architecture C Hessian) the
# real outer run used. Interpolates for the gamma' at which Delta_dual~=1 (delta=1 budget), then
# reports the corresponding kappa -- a cheap (a handful of inner solves per L, no outer KNITRO
# loop at all) ballpark comparison against the real (A-free) outer-loop result.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
using Printf, LinearAlgebra, Statistics, Serialization

W = 80000; DELTA = 1.0
lp(xs...) = (println(xs...); flush(stdout))
lp(">>> building D20 real-data context, W=$W ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
lp(@sprintf(">>> ctx built in %.1fs. D=%d", time()-t0, D))

x_free_calib = ctx.θ0_up[ctx.free_idx]
gp_calib = x_free_calib[1]
A_calib = x_free_calib[2:end]   # the REAL calibration A_od levels -- NOT 1.0 (real-data level
# scaling, unlike D4's normalized synthetic setup; confirmed directly, ranges ~700 to ~4.9e11,
# c13_diag_fixedA_mismatch.jl). This IS "A^*" for this fixed-A profile, per the user's request.
σ = ctx.σ
kappa_of(gp) = 1 - gp^(σ/(σ-1))
lp(@sprintf(">>> calibration gamma'_focal=%.6f  kappa_calib=%.6f", gp_calib, kappa_of(gp_calib)))

snaps = nested_grid_sequence([10, 20, 50])

"Delta_dual at (gp, A_od=A_calib fixed) under the CM restriction at grid L, via the production
bundle. Exception-safe: a failed/unbounded inner solve (nStatus=-300, or any other error) is
reported as (Inf, -300) -- treated as 'past the delta=1 boundary' for bisection purposes, since a
structurally infeasible point certainly has Delta_dual > delta."
function delta_at_gp_fixed_A(pcx, gp::Float64, A_calib::Vector{Float64})
    x_free = vcat(gp, A_calib)
    try
        K, base = cm_production_value(x_free, pcx)
        return -base.ζstar, base.inner_status
    catch
        return Inf, -300
    end
end

results = Dict{Int,Any}()
for L in (10, 20, 50)
    lp("="^90)
    lp("L=$L fixed-A profile")
    lp("="^90)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])

    pts = NamedTuple[]
    function probe(gp)
        Δ, status = delta_at_gp_fixed_A(pcx, gp, A_calib)
        κ = kappa_of(gp)
        lp(@sprintf("  gp=%.6f  kappa=%.6f  Delta_dual=%s  status=%d", gp, κ, isfinite(Δ) ? @sprintf("%.6f", Δ) : "Inf/fail", status))
        push!(pts, (gp = gp, kappa = κ, Delta = Δ, status = status))
        return Δ
    end

    # ---- bracket search: gp_calib is known feasible (Delta<1). Step DOWN (lower gp -> higher
    # kappa, matches the outer loop's own established direction) with a doubling step until we
    # cross Delta=1 (or the solve fails -- treated as having crossed). ----
    gp_lo = gp_calib   # feasible end of the bracket (Delta < 1)
    Δ_lo = probe(gp_lo)
    gp_hi = NaN; Δ_hi = NaN   # infeasible end (Delta >= 1, or solve failure)
    h = 0.002
    gp_try = gp_calib
    for _ in 1:8
        gp_try -= h
        Δ_try = probe(gp_try)
        if !isfinite(Δ_try) || Δ_try >= DELTA
            gp_hi = gp_try; Δ_hi = Δ_try
            break
        else
            gp_lo = gp_try; Δ_lo = Δ_try
        end
        h *= 2
    end

    gp_star = NaN; kappa_star = NaN
    if isnan(gp_hi)
        lp("  ** never crossed Delta_dual=1 within the doubling search range -- widen further (not done, ballpark only) **")
    else
        # ---- bisect ~8 times between (gp_lo, feasible) and (gp_hi, infeasible/over-budget) ----
        a, b = gp_lo, gp_hi
        for _ in 1:8
            mid = (a + b) / 2
            Δ_mid = probe(mid)
            if isfinite(Δ_mid) && Δ_mid < DELTA
                a = mid
            else
                b = mid
            end
        end
        gp_star = a   # the feasible-side bracket endpoint -- a slightly conservative (Delta<=1) ballpark
        kappa_star = kappa_of(gp_star)
    end
    if !isnan(gp_star)
        lp(@sprintf("  BISECTED: gp*=%.6f  kappa*(fixed-A, Delta*~=1)=%.6f", gp_star, kappa_star))
    end
    results[L] = (pts = pts, gp_star = gp_star, kappa_star = kappa_star)
end

lp("")
lp("="^90)
lp("SUMMARY: fixed-A* CM-restricted kappa vs. the real free-A outer-loop result")
lp("="^90)
free_A_kappa = Dict(10 => 0.0651531041750657, 20 => 0.06131771825928012, 50 => 0.059132996954901595)
for L in (10, 20, 50)
    r = results[L]
    fk = free_A_kappa[L]
    if !isnan(r.kappa_star)
        gap = fk - r.kappa_star
        pct = 100 * gap / r.kappa_star
        lp(@sprintf("  L=%-2d  fixed-A kappa*=%.6f  free-A kappa=%.6f  gap=%.6f (+%.1f%%)", L, r.kappa_star, fk, gap, pct))
    else
        lp(@sprintf("  L=%-2d  fixed-A kappa*=UNRESOLVED  free-A kappa=%.6f", L, fk))
    end
end

serialize(joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation", "fixed_A_profile.jls"), results)
lp("DONE")
