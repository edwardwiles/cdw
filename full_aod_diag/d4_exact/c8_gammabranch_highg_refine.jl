# ============================================================================
# Continuation 8, workstream C: high-g branch REFINEMENT, triggered by a
# genuine finding in c8_gammabranch_highg_sweep.jl's Phase 4 -- at g equal to
# the CURRENT lower incumbent's own g (0.9967391744173478), an INDEPENDENT
# start (continuation warm-started from a distant g=0.999 point) found
# Delta=0.8922699, strictly BELOW the lower incumbent's own Delta=1.0000009
# at the SAME g. This means the lower incumbent's own A is a genuine but
# SUBOPTIMAL local basin -- not the global constrained minimizer profile_Delta(g)
# -- exactly the scenario the standing brief flagged as something to check,
# not assume.
#
# This script builds the TRUE fine-continuation profile in [0.994, 0.999]
# (small step 0.0005, always warm-starting from the immediately-previous g's
# best A -- the same discipline the low-g branch and Phase 2 already used
# successfully with zero failures in this exact region), taking the BEST of
# {continuation, calib, lower_incumbent-anchor} at EVERY step (not just at a
# few checkpoints), to avoid the basin-dependence Phase 4 exposed. Locates
# the Delta=delta=1 crossing on this robustly-tracked profile via bisection.
# ============================================================================
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using Printf

const COMMIT_C8 = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8, "c8_gammabranch_highg_refine")
mkpath(OUTDIR)
const MOMENT_REPR = :compressed

"At each g, take the best of {continuation (from zf_prev), calib (fresh), lower_incumbent (fresh, fixed anchor)} -- 3 cheap starts, robust to the single-continuation-path basin trap Phase 4 exposed."
function best_of_three(g::Float64, zf_prev::Vector{Float64}; maxtime::Float64 = 15.0)
    r_cont  = profile_delta_at_gamma_c8(g, zf_prev, ctx, pe; moment_repr = MOMENT_REPR, maxtime_real = maxtime, hessopt_tag = "sr1")
    r_calib = profile_delta_at_gamma_c8(g, ZFREE_CALIB_C8, ctx, pe; moment_repr = MOMENT_REPR, maxtime_real = maxtime, hessopt_tag = "sr1")
    r_lower = profile_delta_at_gamma_c8(g, ZFREE_LOWER_INCUMBENT, ctx, pe; moment_repr = MOMENT_REPR, maxtime_real = maxtime, hessopt_tag = "sr1")
    cands = [(kind="continuation", r=r_cont), (kind="calib", r=r_calib), (kind="lower_incumbent", r=r_lower)]
    feas = filter(c -> c.r.best_zfree !== nothing && isfinite(c.r.best_Delta), cands)
    isempty(feas) && return (g = g, Delta = NaN, kind = "NONE", zfree = nothing, all = cands)
    best = feas[argmin([c.r.best_Delta for c in feas])]
    return (g = g, Delta = best.r.best_Delta, kind = best.kind, zfree = copy(best.r.best_zfree), all = cands)
end

println("="^78)
println("Fine robust profile 0.994 -> 0.999, step 0.0005, best-of-3 every step")
println("="^78)
g_grid = collect(0.9940:0.0005:0.9990)
rows = NamedTuple[]
zf_prev = copy(ZFREE_INCUMBENT_C8)
for g in g_grid
    global zf_prev
    b = best_of_three(g, zf_prev; maxtime = 15.0)
    @printf("  g=%.6f  Delta=%.6e  Delta-delta=%+.4e  best_kind=%-16s  (cont=%.4e calib=%.4e lower=%.4e)\n",
        g, b.Delta, b.Delta - ctx.δ, b.kind,
        b.all[1].r.best_Delta === nothing ? NaN : (isfinite(b.all[1].r.best_Delta) ? b.all[1].r.best_Delta : NaN),
        isfinite(b.all[2].r.best_Delta) ? b.all[2].r.best_Delta : NaN,
        isfinite(b.all[3].r.best_Delta) ? b.all[3].r.best_Delta : NaN)
    push!(rows, (g = g, Delta = b.Delta, kind = b.kind))
    if b.zfree !== nothing
        zf_prev = b.zfree
    end
end

open(joinpath(OUTDIR, "fine_robust_profile.csv"), "w") do io
    println(io, "g,Delta,Delta_minus_delta,best_kind")
    for r in rows
        println(io, r.g, ",", r.Delta, ",", r.Delta - ctx.δ, ",", r.kind)
    end
end
println("\nWrote ", joinpath(OUTDIR, "fine_robust_profile.csv"))

# ---- bisect the Delta=delta crossing on this robust profile ----
feas_rows = filter(r -> isfinite(r.Delta), rows)
below = filter(r -> r.Delta <= ctx.δ, feas_rows)
above = filter(r -> r.Delta > ctx.δ, feas_rows)
if isempty(below) || isempty(above)
    println("WARNING: robust profile does not straddle Delta=delta on this grid -- cannot bisect, inspect fine_robust_profile.csv directly")
else
    g_below = maximum(r.g for r in below)   # largest g still feasible
    g_above = minimum(r.g for r in above)   # smallest g already infeasible
    @printf("\nBisecting between g_below=%.6f (Delta=%.6e) and g_above=%.6f (Delta=%.6e)\n",
        g_below, [r.Delta for r in below if r.g==g_below][1], g_above, [r.Delta for r in above if r.g==g_above][1])

    zf_lo = copy(zf_prev)   # fallback if we can't recover the exact zfree at g_below; refined below
    # recover a good warm-start AT g_below by re-solving there (cheap, 3-start best-of-three again)
    b_lo = best_of_three(g_below, ZFREE_LOWER_INCUMBENT; maxtime = 15.0)
    zf_lo = b_lo.zfree === nothing ? ZFREE_LOWER_INCUMBENT : b_lo.zfree

    g_a, g_b = g_below, g_above
    zf_a = zf_lo
    bisect_trace = NamedTuple[]
    for it in 1:10
        g_mid = (g_a + g_b) / 2
        b = best_of_three(g_mid, zf_a; maxtime = 15.0)
        Δ = b.Delta
        @printf("  iter %2d: g=%.9f  Delta=%s  Delta-delta=%s  best_kind=%s  |bracket|=%.3e\n",
            it, g_mid, isfinite(Δ) ? @sprintf("%.6e", Δ) : "NaN/infeasible",
            isfinite(Δ) ? @sprintf("%+.4e", Δ-ctx.δ) : "N/A", b.kind, g_b - g_a)
        push!(bisect_trace, (iter = it, g = g_mid, Delta = Δ, kind = b.kind))
        if isfinite(Δ) && Δ <= ctx.δ
            g_a = g_mid
            b.zfree !== nothing && (zf_a = b.zfree)
        else
            g_b = g_mid
        end
    end
    @printf("\nRobust high-g crossing bracket: [%.9f, %.9f]  width=%.3e\n", g_a, g_b, g_b-g_a)
    @printf("Robust high-g crossing point estimate (feasible-side edge): g = %.9f\n", g_a)
    kappa_new = 1 - g_a^(ctx.σ/(ctx.σ-1))
    kappa_incumbent = 0.005428799948779983
    @printf("Implied kappa at this g: %.8f  vs existing lower incumbent kappa=%.8f  (%.1f%% %s)\n",
        kappa_new, kappa_incumbent, 100*abs(kappa_new-kappa_incumbent)/kappa_incumbent,
        kappa_new < kappa_incumbent ? "SMALLER/better" : "LARGER/worse")

    open(joinpath(OUTDIR, "highg_bisection_trace.csv"), "w") do io
        println(io, "iter,g,Delta,best_kind")
        for r in bisect_trace
            println(io, r.iter, ",", r.g, ",", r.Delta, ",", r.kind)
        end
    end
    println("Wrote ", joinpath(OUTDIR, "highg_bisection_trace.csv"))
end
