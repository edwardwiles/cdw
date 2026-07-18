# ============================================================================
# Continuation 8, workstream C: low-g branch. Refine the unique
# profile_Delta(g) = delta = 1 crossing near the current upper incumbent
# (g approx 0.8926, Delta=0.9999924058, doc#1's "unique low-g crossing").
#
# Two-stage design (documented rationale, not an ad-hoc shortcut):
#   Stage 1 -- CHEAP bisection: single continuation-warm-started local solve
#     per g (compressed mode, per the bench decision), driving the bracket
#     width down fast. This is what determines the numerical g-precision of
#     the root.
#   Stage 2 -- ROBUSTNESS check: full 9-start multistart (reusing
#     c8_gammabranch_core.jl's make_starts_c8/multistart_profile_at_g, the
#     SAME pattern gamma_profile_multistart.jl established) at the Stage-1
#     root and its two bracket neighbors, to get a spread-based uncertainty
#     estimate independent of the bisection's own tolerance.
# ============================================================================
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using Printf, Statistics

const COMMIT_C8 = strip(strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String)))
const LOWG_OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8, "c8_gammabranch_lowg_bracket")
mkpath(LOWG_OUTDIR)
const MOMENT_REPR = :compressed   # per c8_gammabranch_compressed_vs_dense_bench.jl: ~1.8-1.9x faster, Delta matches dense to 8 digits

"Single-start, continuation-warm-started profile_Delta(g) evaluation (cheap inner evaluator for bisection)."
function f_delta_minus_1(g::Float64, zf_start::Vector{Float64}; maxtime::Float64 = 20.0)
    res = profile_delta_at_gamma_c8(g, zf_start, ctx, pe; moment_repr = MOMENT_REPR, maxtime_real = maxtime, hessopt_tag = "sr1")
    return res.best_Delta - ctx.δ, res
end

println("="^78)
println("STAGE 1: bisection on profile_Delta(g) - delta, low-g branch")
println("="^78)

# ---- confirm the bracket fresh under current code (not trusting historical numbers) ----
g_a = 0.8859910981141633   # historically infeasible (Delta=1.3817, priority4 doc)
g_b = G_INCUMBENT_C8        # 0.8926359584642946, historically Delta=0.9999924058 (barely feasible)

fa, res_a = f_delta_minus_1(g_a, ZFREE_INCUMBENT_C8; maxtime = 25.0)
fb, res_b = f_delta_minus_1(g_b, ZFREE_INCUMBENT_C8; maxtime = 25.0)
@printf("  g_a=%.10f  Delta-delta=%.6e  (knitro_status=%d n_eval=%d)\n", g_a, fa, res_a.knitro_status, res_a.n_eval)
@printf("  g_b=%.10f  Delta-delta=%.6e  (knitro_status=%d n_eval=%d)\n", g_b, fb, res_b.knitro_status, res_b.n_eval)

if sign(fa) == sign(fb)
    error("low-g bracket does not straddle Delta=delta under fresh evaluation (fa=$fa, fb=$fb) -- widen the bracket before bisecting")
end

# regula-falsi / bisection hybrid, warm-started continuation from the closer endpoint each step
bisect_trace = NamedTuple[]
push!(bisect_trace, (iter = 0, g = g_a, f = fa, kind = "endpoint_a"))
push!(bisect_trace, (iter = 0, g = g_b, f = fb, kind = "endpoint_b"))

zf_a, zf_b = copy(res_a.best_zfree), copy(res_b.best_zfree)
g_lo, f_lo, zf_lo = g_a, fa, zf_a   # f_lo > 0 side (infeasible, Delta>delta)
g_hi, f_hi, zf_hi = g_b, fb, zf_b   # f_hi < 0 side (feasible, Delta<=delta)
if f_lo < 0
    g_lo, f_lo, zf_lo, g_hi, f_hi, zf_hi = g_hi, f_hi, zf_hi, g_lo, f_lo, zf_lo
end

const N_BISECT = 14
for it in 1:N_BISECT
    global g_lo, f_lo, zf_lo, g_hi, f_hi, zf_hi
    # secant step, clamped into the bracket (falls back to bisection midpoint if secant lands outside)
    g_secant = g_hi - f_hi * (g_lo - g_hi) / (f_lo - f_hi)
    g_mid = (g_lo + g_hi) / 2
    g_try = (g_secant > min(g_lo, g_hi) && g_secant < max(g_lo, g_hi)) ? g_secant : g_mid
    zf_start = abs(g_try - g_hi) < abs(g_try - g_lo) ? zf_hi : zf_lo
    f_try, res_try = f_delta_minus_1(g_try, zf_start; maxtime = 20.0)
    @printf("  iter %2d: g=%.12f  Delta-delta=%+.6e  |bracket|=%.3e  (status=%d n_eval=%d)\n",
        it, g_try, f_try, abs(g_hi - g_lo), res_try.knitro_status, res_try.n_eval)
    push!(bisect_trace, (iter = it, g = g_try, f = f_try, kind = "bisect"))
    if f_try > 0
        g_lo, f_lo, zf_lo = g_try, f_try, copy(res_try.best_zfree)
    else
        g_hi, f_hi, zf_hi = g_try, f_try, copy(res_try.best_zfree)
    end
    abs(g_hi - g_lo) < 1e-9 && break
end

g_root_stage1 = g_hi   # the feasible-side endpoint (Delta<=delta), i.e. find_smallest's own convention
@printf("\nSTAGE 1 bracket after %d iterations: [%.12f, %.12f]  width=%.3e\n", N_BISECT, min(g_lo,g_hi), max(g_lo,g_hi), abs(g_hi-g_lo))
@printf("Stage-1 point estimate of the root (feasible-side bracket edge): g = %.12f\n", g_root_stage1)

open(joinpath(LOWG_OUTDIR, "bisection_trace.csv"), "w") do io
    println(io, "iter,g,Delta_minus_delta,kind")
    for r in bisect_trace
        println(io, r.iter, ",", r.g, ",", r.f, ",", r.kind)
    end
end

println("\n", "="^78)
println("STAGE 2: 9-start multistart robustness check at the root and its bracket neighbors")
println("="^78)

g_check = sort(unique([g_lo, g_root_stage1, g_hi]))
ms_rows = NamedTuple[]
for g in g_check
    rng_seed = 71000 + round(Int, g * 1e7)
    ms = multistart_profile_at_g(g, zf_hi, ctx, pe;
            anchors = [("incumbent", ZFREE_INCUMBENT_C8), ("calib", ZFREE_CALIB_C8)],
            moment_repr = MOMENT_REPR, maxtime_per_start = 15.0, seed = rng_seed)
    feas = filter(r -> r.feasible, ms.all)
    deltas = [r.Delta for r in feas]
    @printf("  g=%.10f  n_feasible=%d/%d  min=%.8f  max=%.8f  spread=%.3e  best_start=%s\n",
        g, length(feas), length(ms.all), isempty(deltas) ? NaN : minimum(deltas), isempty(deltas) ? NaN : maximum(deltas),
        isempty(deltas) ? NaN : (maximum(deltas)-minimum(deltas)), ms.best === nothing ? "NONE" : ms.best.kind)
    push!(ms_rows, (g = g, ms = ms, feas = feas))
end

# spread-based uncertainty: at g closest to the stage-1 root estimate
idx_closest = argmin(abs.([r.g for r in ms_rows] .- g_root_stage1))
spread_row = ms_rows[idx_closest]
deltas_at_root = [r.Delta for r in spread_row.feas]
delta_spread = isempty(deltas_at_root) ? NaN : maximum(deltas_at_root) - minimum(deltas_at_root)
# translate Delta-spread into an implied g-uncertainty via the local secant slope from stage 1
local_slope = abs((f_hi - f_lo) / (g_hi - g_lo))   # |dDelta/dg| estimate from the final bisection step
g_uncertainty_from_multistart = local_slope > 0 ? delta_spread / local_slope : NaN

@printf("\nDelta multistart spread at g closest to root (g=%.10f): %.3e\n", spread_row.g, delta_spread)
@printf("Local |dDelta/dg| estimate (final bisection step): %.4f\n", local_slope)
@printf("Implied g-uncertainty from multistart spread: %.3e\n", g_uncertainty_from_multistart)
@printf("Bisection bracket width (numerical precision floor): %.3e\n", abs(g_hi-g_lo))

open(joinpath(LOWG_OUTDIR, "multistart_at_root.csv"), "w") do io
    println(io, "g,start_kind,Delta,feasible,knitro_status,n_eval,wall")
    for r in ms_rows, s in r.ms.all
        println(io, r.g, ",", s.kind, ",", s.Delta, ",", s.feasible, ",", s.knitro_status, ",", s.n_eval, ",", s.wall)
    end
end

println("\nWrote:")
println("  ", joinpath(LOWG_OUTDIR, "bisection_trace.csv"))
println("  ", joinpath(LOWG_OUTDIR, "multistart_at_root.csv"))

println("\nSUMMARY (low-g branch):")
@printf("  root g* (feasible-side, Delta<=delta) ~= %.9f\n", g_root_stage1)
@printf("  bracket = [%.9f, %.9f], width = %.3e\n", min(g_lo,g_hi), max(g_lo,g_hi), abs(g_hi-g_lo))
@printf("  multistart Delta-spread at root ~= %.3e  =>  implied g-uncertainty ~= %.3e\n", delta_spread, g_uncertainty_from_multistart)
