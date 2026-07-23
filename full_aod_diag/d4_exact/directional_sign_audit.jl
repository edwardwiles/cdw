# Part I sign/directional audit, 2026-07-23. Independently re-derives and re-verifies the sign
# convention between the production fixed-dual outer gradient (evalResult.jac) and an
# independently reoptimized (fresh inner dual re-solve at each probe) central-FD secant of the
# CANONICAL Delta_dual value, per docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md Section 6.
#
# Does NOT trust overnight_cm_cplus_d4_gate.jl's own Section 4 "true_secant" formula -- rebuilds
# it from scratch here and cross-checks both versions (as that file wrote it, and the
# non-negated form Section 6's own derivation implies) against each other and against a
# high-accuracy independent finite difference.
include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
using Printf, LinearAlgebra, Random

const OUTCSV = joinpath(@__DIR__, "..", "..", "results", "cm_cplus_followup", "directional_audit.csv")
mkpath(dirname(OUTCSV))

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
const L = 10
contrasts = :orthonormal
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts)
ctx_cm = pcx.ctx_cm; aug = pcx.aug; bins = pcx.bins; cctx = pcx.cctx
ws = build_lfix_factorized_workspace(D, W)

x_free_calib = ctx.θ0_up[ctx.free_idx]

"Canonical Delta_dual: fresh inner solve at x_free (archC_verified_state's own explicit obj recompute at converged duals). This is the SAME function/formula cb_F! reports to KNITRO -- see algebra trace Section 2."
function delta_dual_fresh(x_free)
    base_here, verify_here = archC_verified_state(x_free, ctx_cm, cctx)
    return verify_here.Delta_dual, base_here
end

function theta_full_of(x_free)
    return CS.reconstruct_full(x_free, ctx_cm.m)
end

"w-space (gravity-pivot free coords) -> x_free, matching overnight_cm_cplus_d4_gate.jl's own construction exactly."
function xfree_from_w(w)
    z = pivot_expand(w[2:end], pe)
    return vcat(w[1], vec(exp.(z)))
end

function w_of(xf)
    z0 = log.(reshape(xf[2:end], D, D))
    return vcat(xf[1], pivot_reduce(z0, pe))
end

w0 = w_of(x_free_calib)
base0, verify0 = archC_verified_state(x_free_calib, ctx_cm, cctx)
@printf "Base point: Delta_dual = %.10e, inner nStatus check via verify.inner_status = %s\n" verify0.Delta_dual string(get(verify0, :inner_status, missing))

cache_ref0 = build_lfix_base_cache_cm(x_free_calib, ctx_cm, base0, ctx, aug, bins)
cache_cp0 = build_lfix_base_cache_cm_C!(ws, x_free_calib, ctx_cm, base0, ctx, aug, bins)

# Sanity check #1: lfix_from_q(cache.q0, cache.ζstar) must equal Delta_dual AT THE BASE POINT
# (h=0, no probe) -- confirms the two scalars share not just a formula but the SAME VALUE at the
# anchor, which Section 6 asserts but did not numerically re-verify.
lfix_at_base = lfix_from_q(cache_ref0.q0, cache_ref0.ζstar)
@printf "Sanity #1: lfix_from_q(q0,zeta*) at base = %.10e  vs  Delta_dual at base = %.10e  |diff| = %.3e\n" lfix_at_base verify0.Delta_dual abs(lfix_at_base - verify0.Delta_dual)
@assert abs(lfix_at_base - verify0.Delta_dual) < 1e-8 "BLOCKER: L_fix and Delta_dual disagree at the anchor point itself -- the two scalars are not even the same quantity at h=0."

winner0, _, gap0 = compute_winners(theta_full_of(x_free_calib), ctx)

hs = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
coords = unique([2, 3, min(5, D2), D2])  # a handful of free A-block coordinates (w-space index 1 is gamma; 2.. are A-block)

rows = NamedTuple[]

for k in coords
    println()
    println("="^100)
    println("coordinate k = $k")
    println("="^100)
    for h in hs
        wp = copy(w0); wp[k] += h
        wm = copy(w0); wm[k] -= h
        xfp = xfree_from_w(wp); xfm = xfree_from_w(wm)

        Dp, basep = delta_dual_fresh(xfp)
        Dm, basem = delta_dual_fresh(xfm)

        # (A) the formula AS WRITTEN in overnight_cm_cplus_d4_gate.jl Section 4 (extra leading minus)
        true_secant_asis = -(Dp - Dm) / (2h)
        # (B) the formula Section 6's own derivation implies (no extra minus): matches Lp,Lm order/sign exactly
        true_secant_corrected = (Dp - Dm) / (2h)

        Lp_ref = lfix_incremental_at(cache_ref0, ctx_cm, pe, w0, k, w0[k] + h)
        Lm_ref = lfix_incremental_at(cache_ref0, ctx_cm, pe, w0, k, w0[k] - h)
        secant_ref = (Lp_ref - Lm_ref) / (2h)

        Lp_cp = lfix_incremental_at_C(cache_cp0, ctx_cm, pe, w0, k, w0[k] + h)
        Lm_cp = lfix_incremental_at_C(cache_cp0, ctx_cm, pe, w0, k, w0[k] - h)
        secant_cp = (Lp_cp - Lm_cp) / (2h)

        winner_p, _, _ = compute_winners(theta_full_of(xfp), ctx)
        winner_m, _, _ = compute_winners(theta_full_of(xfm), ctx)
        nswitch_p = count(winner_p .!= winner0)
        nswitch_m = count(winner_m .!= winner0)

        sign_match_asis = sign(true_secant_asis) == sign(secant_ref)
        sign_match_corrected = sign(true_secant_corrected) == sign(secant_ref)

        @printf "  h=%.1e  true_asis=%+.6e  true_corrected=%+.6e  fixed_ref=%+.6e  fixed_cp=%+.6e  | winner-switches(+h)=%d (-h)=%d | sign_match(asis)=%s sign_match(corrected)=%s\n" h true_secant_asis true_secant_corrected secant_ref secant_cp nswitch_p nswitch_m sign_match_asis sign_match_corrected

        push!(rows, (k=k, h=h, Dp=Dp, Dm=Dm, true_secant_asis=true_secant_asis,
                      true_secant_corrected=true_secant_corrected, fixed_dual_ref=secant_ref,
                      fixed_dual_cp=secant_cp, nswitch_p=nswitch_p, nswitch_m=nswitch_m,
                      sign_match_asis=sign_match_asis, sign_match_corrected=sign_match_corrected))
    end
end

open(OUTCSV, "w") do io
    println(io, "k,h,Dp,Dm,true_secant_asis,true_secant_corrected,fixed_dual_ref,fixed_dual_cp,nswitch_p,nswitch_m,sign_match_asis,sign_match_corrected")
    for r in rows
        println(io, "$(r.k),$(r.h),$(r.Dp),$(r.Dm),$(r.true_secant_asis),$(r.true_secant_corrected),$(r.fixed_dual_ref),$(r.fixed_dual_cp),$(r.nswitch_p),$(r.nswitch_m),$(r.sign_match_asis),$(r.sign_match_corrected)")
    end
end
println()
println("Wrote $OUTCSV ($(length(rows)) rows)")

# Convergence classification: for each k, as h -> 0 (restricted to winner-stable probes), does
# true_secant_corrected converge toward fixed_dual, or does true_secant_asis?
println()
println("="^100)
println("Per-coordinate convergence summary (winner-stable rows only, smallest available h)")
println("="^100)
for k in coords
    sub = filter(r -> r.k == k && r.nswitch_p == 0 && r.nswitch_m == 0, rows)
    isempty(sub) && (println("  k=$k: NO winner-stable probe found at any tested h -- cannot classify, see winner-boundary-nonsmoothness bucket"); continue)
    best = sub[argmin([r.h for r in sub])]
    ratio_corrected = best.true_secant_corrected / best.fixed_dual_ref
    ratio_asis = best.true_secant_asis / best.fixed_dual_ref
    @printf "  k=%d  h=%.1e  true_corrected/fixed_dual = %+.4f   true_asis/fixed_dual = %+.4f\n" k best.h ratio_corrected ratio_asis
end

# ============================================================================
# I.3: directional predictions, not only coordinate equality. Compare the
# fixed-dual Jacobian's LINEAR prediction (t*dot(g_ref, dir)) against the
# independently reoptimized Delta_dual_fresh(w0 + t*dir) - Delta_dual_fresh(w0)
# over a small symmetric t-grid, for several normalized directions.
# ============================================================================
println()
println("="^100)
println("SECTION I.3: directional predictions (fixed-dual-Jacobian-implied vs independently reoptimized)")
println("="^100)

g_ref_full, _ = cm_production_gradient(x_free_calib, pcx, ctx, pe; base = base0, threaded = false, h_mode = :fixed, h0 = 0.01)
Random.seed!(4242)
dir_grad = g_ref_full ./ norm(g_ref_full)
dir_sparse = zeros(D2); dir_sparse[3] = 1.0
top3 = sortperm(abs.(g_ref_full), rev = true)[1:3]
dir_topmag = zeros(D2); dir_topmag[top3] .= sign.(g_ref_full[top3]); dir_topmag ./= norm(dir_topmag)
dir_switchheavy = randn(D2); dir_switchheavy ./= norm(dir_switchheavy)

directions = [("grad_direction", dir_grad), ("sparse_e3", dir_sparse),
              ("top3_magnitude", dir_topmag), ("random_switch_heavy", dir_switchheavy)]

# symmetric magnitude grid (mirrors I.2's bandwidth grid, extended finer) so a winner-stable
# regime can actually be located; a single-sided step at t=0.002-0.02 already crosses many
# decision boundaries along combined directions (unlike single coordinates), so this must be
# checked down to the same 1e-4 floor used in I.2, not assumed comparable at coarse t.
tmags = [0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4]
D0 = verify0.Delta_dual
dir_rows = NamedTuple[]
for (dname, dir) in directions
    println("-"^100)
    println("direction: $dname")
    predicted_slope = dot(g_ref_full, dir)
    for t in tmags
        wp = w0 .+ t .* dir; wm = w0 .- t .* dir
        xfp = xfree_from_w(wp); xfm = xfree_from_w(wm)
        Dp_, _ = delta_dual_fresh(xfp); Dm_, _ = delta_dual_fresh(xfm)
        actual_secant = (Dp_ - Dm_) / (2t)          # symmetric, correctly-signed (matches I.2's true_secant_corrected convention)
        onesided_change = Dp_ - D0                   # what a single +t KNITRO-style step would actually see
        winner_p, _, _ = compute_winners(theta_full_of(xfp), ctx)
        winner_m, _, _ = compute_winners(theta_full_of(xfm), ctx)
        nsw_p = count(winner_p .!= winner0); nsw_m = count(winner_m .!= winner0)
        sign_match = sign(actual_secant) == sign(predicted_slope)
        @printf "  t=%.1e  actual_secant=%+.6e  predicted_slope=%+.6e  onesided_Δ(+t)=%+.6e  sign_match=%s  nswitch(+t)=%d nswitch(-t)=%d\n" t actual_secant predicted_slope onesided_change sign_match nsw_p nsw_m
        push!(dir_rows, (direction = dname, t = t, actual_secant = actual_secant, predicted_slope = predicted_slope, onesided_change = onesided_change, nswitch_p = nsw_p, nswitch_m = nsw_m, sign_match = sign_match))
    end
end

OUTCSV2 = joinpath(@__DIR__, "..", "..", "results", "cm_cplus_followup", "directional_predictions.csv")
open(OUTCSV2, "w") do io
    println(io, "direction,t,actual_secant,predicted_slope,onesided_change,nswitch_p,nswitch_m,sign_match")
    for r in dir_rows
        println(io, "$(r.direction),$(r.t),$(r.actual_secant),$(r.predicted_slope),$(r.onesided_change),$(r.nswitch_p),$(r.nswitch_m),$(r.sign_match)")
    end
end
println()
println("Per-direction convergence at smallest winner-stable t:")
for (dname, _) in directions
    sub = filter(r -> r.direction == dname && r.nswitch_p == 0 && r.nswitch_m == 0, dir_rows)
    if isempty(sub)
        println("  $dname: NO winner-stable magnitude found down to 1e-4 -- genuinely winner-switch-heavy direction, classify as winner-boundary nonsmoothness, not sign failure")
    else
        best = sub[argmin([r.t for r in sub])]
        @printf "  %s: t=%.1e  actual_secant=%+.6e  predicted_slope=%+.6e  ratio=%+.4f\n" dname best.t best.actual_secant best.predicted_slope (best.actual_secant/best.predicted_slope)
    end
end
println()
println("Wrote $OUTCSV2 ($(length(dir_rows)) rows)")

println("DONE")
