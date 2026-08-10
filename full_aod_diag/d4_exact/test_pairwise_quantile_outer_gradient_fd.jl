# ================================================================================================
# Reoptimized-FD validation gate for the pairwise-quantile-independence restriction's OUTER gradient.
#
# This is the gate the handover doc (docs/PAIRWISE_QUANTILE_OUTER_LOOP_INTEGRATION_HANDOVER_
# 2026-08-10.md, task list item 4) names as required before the outer gradient may be called
# correct. It is the direct analog of what this codebase already holds origin-ZC's own nu-gradient
# to: `d_delta_dual_d_eta_origin_fd` + `test_cm_originzc_pure_moments.jl`'s testset, where every FD
# probe RE-SOLVES the inner dual from scratch.
#
# What is checked, in order:
#   1. Delta_dual is actually produced by the new verified-state layer, and the solve verifies.
#   2. The analytic CUTOFF block vs reoptimized FD, at MATCHED bandwidth (see below).
#   3. The analytic ECONOMIC block vs reoptimized FD -- i.e. that this restriction being active does
#      not disturb the shared, family-agnostic (g, A_od) gradient.
#   4. The COMBINED vector's layout: that vcat(g_econ, g_cut) lines up index-for-index with the
#      outer coordinate vector w = vcat(gp, zfree, raw_cutoffs) the driver actually hands KNITRO.
#      (Handover doc: "validate the COMBINED vector too, not just the cutoff piece in isolation.")
#
# ON MATCHED BANDWIDTH -- why this gate is not `h = 1e-4`:
# Delta_dual is a genuine STEP function of any single cutoff (it changes only when a cutoff crosses
# an actual draw). An FD probe smaller than the gap to the next draw crosses nothing and returns
# exactly 0.0 for every coordinate. So "analytic vs FD at h=1e-4" would report total disagreement
# for a perfectly correct gradient. Both sides must be secants of the same staircase over the same
# window, which is what `matched_raw_steps`/`cutoff_probe_points` provide (one shared helper, so the
# gate cannot drift out of sync with the thing it gates). See memory
# `feedback-fd-bandwidth-mismatch-looks-like-a-bug`.
#
# Even at matched bandwidth these two quantities are NOT expected to agree to solver tolerance, and
# the pass thresholds below say so explicitly rather than being tuned until green:
#   - analytic = FIXED-dual secant (zeta, lambda frozen at the base solve's optimum)
#   - FD       = REOPTIMIZED secant (inner dual re-solved at each probe)
# They agree to first order by the envelope theorem; they differ at second order in the probe width,
# which is deliberately NOT infinitesimal here. So the gate checks SIGN agreement and RELATIVE
# agreement on the coordinates that carry real signal, not bitwise equality -- and it checks the
# correlation across the whole probed set, which is what would actually break under a sign error, a
# layout/transpose error, or a chain-rule error.
#
# Usage:  julia --project=. full_aod_diag/d4_exact/test_pairwise_quantile_outer_gradient_fd.jl [L] [min_crossed]
# ================================================================================================
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "shared_a_gradient.jl", "operator_verification.jl", "cm_screen_bridge.jl",
          "pairwise_quantile_cutoff_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_cutoff_gradient.jl",
          "pairwise_quantile_outer_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool, detail::AbstractString = "")
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
    flush(stdout)
end

const PQ_L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const MIN_CROSSED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200

println("=== pairwise-quantile OUTER gradient: reoptimized-FD gate ===")
println("L = ", PQ_L, "  min_crossed = ", MIN_CROSSED)
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
layout = PairwiseQuantileCutoffLayout(ctx.D, PQ_L)
pcx = build_pairwise_quantile_production_context(ctx, layout; min_crossed = MIN_CROSSED)
ctx_cm = pcx.ctx_cm
W = size(ctx.U, 1)
println("W = ", W, "  D = ", ctx.D, "  n_raw = ", n_raw(layout), "  n_econ_free = ", length(x_free_calib))
flush(stdout)

function quantile_naive(v::AbstractVector{Float64}, p::Float64)
    s = sort(v); n = length(s)
    return s[clamp(round(Int, p * n), 1, n)]
end
raw_cutoffs = zeros(n_raw(layout))
for o in 1:ctx.D
    b = raw_index(layout, o, 1)
    Uo = @view ctx.U[:, o]
    q = [quantile_naive(Uo, r / PQ_L) for r in 1:PQ_L-1]
    raw_cutoffs[b] = log(q[1])
    for k in 2:PQ_L-1
        gap = log(q[k]) - log(q[k-1])
        raw_cutoffs[b+k-1] = gap > 0 ? log(expm1(gap)) : -5.0
    end
end

# ---------------------------------------------------------------------------------------------
# 1. The objective value the outer loop consumes
# ---------------------------------------------------------------------------------------------
println("\n=== 1. verified state produces Delta_dual ===")
t0 = time()
base, verify = archPQ_verified_state(x_free_calib, raw_cutoffs, ctx_cm)
println("inner_status = ", verify.inner_status, "  Delta_dual = ", verify.Delta_dual,
        "  Delta_primal = ", verify.Delta_primal, "  (", round(time() - t0, digits = 2), "s)")
println("primal_dual_gap = ", verify.primal_dual_gap, "  mean_m_resid = ", verify.mean_m_resid,
        "  max_abs_moment_kkt_resid = ", verify.max_abs_moment_kkt_resid)
println("block KKT: E = ", verify.kkt_resid_E, "  marginalbin = ", verify.kkt_resid_marginalbin,
        "  pairindep = ", verify.kkt_resid_pairindep)
check("Delta_dual is finite", isfinite(verify.Delta_dual), "Delta_dual=$(verify.Delta_dual)")
check("verify carries r_current of length W", length(verify.r_current) == W)
cls = classify_inner_result(verify)
println("classify_inner_result = ", cls,
        cls == VerifiedSolved ? "" : "  reasons=$(verification_rejection_reasons(verify))")
check("inner solve is VerifiedSolved under the SHARED acceptance gate", cls == VerifiedSolved)

# ---------------------------------------------------------------------------------------------
# 2. CUTOFF block vs reoptimized FD at matched bandwidth
# ---------------------------------------------------------------------------------------------
println("\n=== 2. cutoff block: analytic vs reoptimized FD (matched bandwidth) ===")

# The gate runs at TWO cutoff points, because they test different things:
#
#  (a) the NATURAL starting point (cutoffs at each origin's own empirical quantiles). Confirmed
#      live (debug_pq_cutoff_sign_isolate.jl EXPERIMENT 2, 16/16 cutoffs): Delta* rises in BOTH
#      directions here, i.e. this point is a local MINIMUM of Delta* in every cutoff coordinate.
#      That is not an accident -- the restriction's targets are the fixed constants 1/L and 1/L^2,
#      so cutoffs AT the empirical quantiles are exactly where the unweighted draws already come
#      closest to satisfying it. The true gradient is therefore near zero here and any finite probe
#      is curvature-dominated, which makes it a WEAK test of a gradient's correctness however
#      reassuring it looks.
#
#  (b) a deliberately OFF-optimum point (cutoffs pushed away from the empirical quantiles), where
#      the true gradient is large and unambiguous. This is the point that actually discriminates a
#      correct gradient from a wrong one, and it is where the pass thresholds below are enforced.
function run_cutoff_gate(label::AbstractString, raw::Vector{Float64}, enforce::Bool)
    println("\n--- cutoff gate at: ", label, " ---")
    b, v = archPQ_verified_state(x_free_calib, raw, ctx_cm)
    println("Delta_dual = ", v.Delta_dual, "  class = ", classify_inner_result(v))
    enforce && check("[$label] inner solve VerifiedSolved", classify_inner_result(v) == VerifiedSolved)
    ga_full = pairwise_quantile_cutoff_gradient_vec(b, v, ctx_cm, raw; min_crossed = MIN_CROSSED)
    enforce && check("[$label] analytic cutoff gradient all-finite", all(isfinite, ga_full))

    t = time()
    (gf_full, h_used, n_probed) = d_delta_dual_d_cutoff_fd(x_free_calib, raw, ctx_cm;
        min_crossed = MIN_CROSSED, verbose = false)
    println("probed ", n_probed, "/", n_raw(layout), " coords (", round(time() - t, digits = 1),
            "s, ", 2 * n_probed, " inner solves)")
    pr = [j for j in 1:n_raw(layout) if isfinite(gf_full[j])]
    @printf("%6s %14s %14s %14s %10s\n", "coord", "analytic", "reopt-FD", "h_used", "ratio")
    for j in pr
        @printf("%6d %14.6e %14.6e %14.6e %10.4f\n", j, ga_full[j], gf_full[j], h_used[j],
                gf_full[j] == 0.0 ? NaN : ga_full[j] / gf_full[j])
    end
    ga = ga_full[pr]; gf = gf_full[pr]
    scale = maximum(abs, gf)
    sig = [i for i in eachindex(gf) if abs(gf[i]) > 0.10 * scale]
    r_corr = length(ga) > 1 ? cor(ga, gf) : NaN
    rel = norm(ga .- gf) / max(norm(gf), eps())
    sign_ok = isempty(sig) ? 0 : count(i -> sign(ga[i]) == sign(gf[i]), sig)
    ratios = isempty(sig) ? Float64[] : [ga[i] / gf[i] for i in sig]
    println("\nFD scale=", scale, "  signal coords (>10% of scale)=", length(sig),
            "  sign agreement=", sign_ok, "/", length(sig))
    isempty(ratios) || println("ratio analytic/FD on signal coords: min=", minimum(ratios),
                               " median=", median(ratios), " max=", maximum(ratios))
    println("correlation(analytic, FD) = ", r_corr, "   relative L2 error = ", rel)
    if enforce
        check("[$label] sign agreement on all signal-carrying coordinates", sign_ok == length(sig),
              "$sign_ok / $(length(sig))")
        check("[$label] median ratio analytic/FD within [0.75, 1.35]",
              !isempty(ratios) && 0.75 <= median(ratios) <= 1.35,
              isempty(ratios) ? "no signal coords" : "median=$(median(ratios))")
        check("[$label] correlation(analytic, FD) > 0.95", isfinite(r_corr) && r_corr > 0.95, "r=$r_corr")
        check("[$label] relative L2 error < 0.35", rel < 0.35, "rel=$rel")
        # Negative control: with the documented Delta_dual = -f sign flip removed, the SAME
        # comparison must fail decisively. This is what makes the checks above evidence about the
        # SIGN, not merely about the magnitude.
        r_wrong = length(ga) > 1 ? cor(-ga, gf) : NaN
        check("[$label] NEGATIVE CONTROL: unflipped gradient anti-correlates with FD",
              isfinite(r_wrong) && r_wrong < -0.95, "r_wrong=$r_wrong")
    end
    return (ga = ga_full, gf = gf_full, corr = r_corr, rel = rel)
end

# (a) natural starting point -- reported, not enforced (see comment above: near-zero true gradient).
res_a = run_cutoff_gate("empirical-quantile start (local min, weak test)", copy(raw_cutoffs), false)

# (b) off-optimum point: shift every origin's first raw coordinate (which shifts that origin's whole
# cutoff ladder in log-space) well away from the empirical quantiles, and widen the first gap.
raw_off = copy(raw_cutoffs)
for o in 1:ctx.D
    b = raw_index(layout, o, 1)
    raw_off[b] += 0.35
    n_cutoffs(layout) >= 2 && (raw_off[b+1] += 0.25)
end
res_b = run_cutoff_gate("off-optimum cutoffs (real gradient signal)", raw_off, true)

g_cut = pairwise_quantile_cutoff_gradient_vec(base, verify, ctx_cm, raw_cutoffs; min_crossed = MIN_CROSSED)
check("analytic cutoff gradient is not identically zero at the base point",
      maximum(abs, g_cut) > 0.0, "max|g_cut|=$(maximum(abs, g_cut))")

# --- 2c. Is the residual analytic-vs-FD gap a BANDWIDTH artifact or a systematic bias? ---------
# The off-optimum gate above passes but with a systematically low median ratio. Two explanations
# have opposite consequences and must be distinguished, not assumed:
#   - bandwidth artifact: analytic (fixed-dual) and FD (reoptimized) secants differ at SECOND order
#     in the probe width, so the gap must SHRINK as min_crossed shrinks. Harmless: at real campaign
#     W the same min_crossed is a far smaller fraction of the draws.
#   - systematic bias: a missing chain-rule/scale factor would show a gap that does NOT shrink.
# This sweep reports which one it is, at the off-optimum point where the signal is real.
println("\n--- 2c. bandwidth sensitivity of the analytic-vs-FD gap (off-optimum point) ---")
@printf("%10s %14s %14s %14s\n", "min_crossed", "median ratio", "correlation", "rel L2 err")
for mc in (25, 50, 100, 200)
    bb, vv = archPQ_verified_state(x_free_calib, raw_off, ctx_cm)
    gaa = pairwise_quantile_cutoff_gradient_vec(bb, vv, ctx_cm, raw_off; min_crossed = mc)
    (gff, _, _) = d_delta_dual_d_cutoff_fd(x_free_calib, raw_off, ctx_cm; min_crossed = mc)
    pr = [j for j in 1:n_raw(layout) if isfinite(gff[j])]
    a = gaa[pr]; f = gff[pr]
    sc = maximum(abs, f)
    sg = [i for i in eachindex(f) if abs(f[i]) > 0.10 * sc]
    @printf("%10d %14.4f %14.4f %14.4f\n", mc,
            isempty(sg) ? NaN : median([a[i] / f[i] for i in sg]),
            length(a) > 1 ? cor(a, f) : NaN,
            norm(a .- f) / max(norm(f), eps()))
    flush(stdout)
end

# ---------------------------------------------------------------------------------------------
# 3. ECONOMIC block vs reoptimized FD (is the shared block disturbed by this restriction?)
# ---------------------------------------------------------------------------------------------
println("\n=== 3. economic block: analytic vs reoptimized FD ===")
g_full, meta = pairwise_quantile_production_gradient(x_free_calib, raw_cutoffs, pcx, ctx, pe;
    base = base, verify = verify, min_crossed = MIN_CROSSED, threaded = false, h_mode = :cached)
D2_econ = ctx.D * (hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D)
println("length(g_full) = ", length(g_full), "  D2_econ = ", D2_econ, "  n_raw = ", n_raw(layout))
check("combined gradient length == D*Ddest + n_raw(layout)", length(g_full) == D2_econ + n_raw(layout))
check("combined gradient is all-finite", all(isfinite, g_full))
check("combined gradient's cutoff tail == the standalone cutoff gradient",
      g_full[D2_econ+1:end] == g_cut)

# `economic_A_gradient!` returns the gradient ALREADY in the driver's own outer economic
# coordinates w_econ = (gp, zfree) -- index 1 is d/dgp (`meta.gamma_component`) and indices
# 2:D2_econ are d/dzfree in pivot-reduced z-space, with the gravity-pivot chain rule applied
# internally (shared_a_gradient.jl's own docstring: "same gravity-pivot chain rule
# (pivot_expand/pivot_reduce)"). So the FD must move w_econ, NOT x_free, and map back through the
# same `x_free_from_w` the driver uses. Getting this backwards would compare two different
# coordinate systems and manufacture a disagreement out of nothing.
xfw(w_econ) = vcat(w_econ[1], vec(exp.(pivot_expand(w_econ[2:end], pe))))
A_calib = reshape(collect(x_free_calib[2:end]), ctx.D, hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D)
w_econ0 = vcat(x_free_calib[1], pivot_reduce(log.(A_calib), pe))
check("w_econ round-trips to the calibration x_free", isapprox(xfw(w_econ0), collect(x_free_calib); rtol = 1e-8),
      "max abs diff = $(maximum(abs, xfw(w_econ0) .- collect(x_free_calib)))")

g_econ_A = g_full[1:D2_econ]

# --- 3a. The DECISIVE check on the restriction's effect on the economic block: q0 exactness ------
# `build_lfix_base_cache_pairwise_quantile` folds this restriction's G_R*lambda_R into the cache's
# q0 and cross-checks the result against `verify.r_current`, which is recomputed by a completely
# independent route (the verifier's own fresh forward pass). Both are the same quantity
# r = -zeta - E*lambda_E - G_R*lambda_R, so agreement to floating point is an EXACT statement that
# the economic block is being linearized about the right base point with the restriction active --
# far stronger than any finite-difference comparison, and the check that caught the real bug here
# (an un-folded q0, initially rationalized away as "the restriction rows don't depend on theta").
ensure_pq_bins!(ctx_cm, raw_cutoffs)
cache_pq = build_lfix_base_cache_pairwise_quantile(x_free_calib, ctx_cm, base; verify = verify)
q0_err = maximum(abs, cache_pq.q0 .- verify.r_current)
println("max|q0(with restriction fold) - independently recomputed r| = ", q0_err)
check("q0 restriction fold is exact against the independent verifier recompute", q0_err <= 1e-8,
      "err=$q0_err")
# And the control that gives that number its meaning: WITHOUT the fold, q0 is materially wrong.
cache_nofold = build_lfix_base_cache(x_free_calib, ctx_cm, base; validate_dense = false)
q0_err_nofold = maximum(abs, cache_nofold.q0 .- verify.r_current)
println("max|q0(WITHOUT the fold) - r| = ", q0_err_nofold, "   (control: must be large)")
check("CONTROL: omitting the restriction fold gives a materially wrong q0", q0_err_nofold > 1e-3,
      "err_nofold=$q0_err_nofold")

# --- 3b. Reference equivalence: the economic block is the SHARED one, driven identically ---------
# This mirrors how the codebase already gates this block (test_shared_a_gradient.jl), which checks
# `economic_A_gradient!` against `composite_gradient_at_fast` for BIT-IDENTITY on the same inputs,
# and deliberately does NOT finite-difference it. The reason matters here: the economic A-block
# gradient is itself an ADAPTIVE BANDWIDTH-SELECTED secant (`select_bandwidth`, composite_gradient.jl
# :196-213), whose own docstring states that a probe smaller than its h_floor=1e-4 "would just
# reproduce Method-A's known-wrong winner-boundary-dropping gradient". A small-h reoptimized FD
# therefore measures a DIFFERENT (and, per this codebase's own analysis, wrong) estimator -- it is
# not a ground truth for this block, and an earlier draft of this gate that treated it as one
# produced a confident-looking failure on exactly the coordinates where the winner-boundary term
# dominates. See memory `feedback-fd-bandwidth-mismatch-looks-like-a-bug`.
g_ref, meta_ref = composite_gradient_at_fast(x_free_calib, ctx_cm, pe; base = base, cache = cache_pq,
    h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
ws_chk = get_or_build_econ_a_grad_ws(cache_pq.W)
g_shared = zeros(D2_econ)
economic_A_gradient!(g_shared, base, ctx_cm, pe, ws_chk; cache = cache_pq, h_mode = :cached,
    bandwidth_cache = Dict{Int,Float64}())
println("max|economic_A_gradient! - composite_gradient_at_fast| = ", maximum(abs, g_shared .- g_ref))
check("economic block bit-identical to the reference implementation on the same cache",
      g_shared == g_ref, "max|diff|=$(maximum(abs, g_shared .- g_ref))")

# --- 3c. Matched-bandwidth reoptimized FD on the economic block: DIAGNOSTIC ONLY -----------------
# Reported, not enforced. Probing at the bandwidth `select_bandwidth` itself chose makes the two
# estimators comparable in principle, but the economic secant is a switching-mass-targeted probe
# whose reoptimized counterpart carries the same finite-window curvature gap quantified for the
# cutoff block in section 2c -- so this is informative, not a pass/fail criterion.
econ_coords = collect(2:min(5, D2_econ))
println("\nmatched-bandwidth economic FD (DIAGNOSTIC, not gated), coords ", econ_coords, ":")
@printf("%6s %14s %14s %12s %10s\n", "coord", "analytic", "reopt-FD", "h_used", "ratio")
for j in econ_coords
    h = meta_ref.h_used[j]
    h > 0 || continue
    wp = copy(w_econ0); wp[j] += h
    wm = copy(w_econ0); wm[j] -= h
    _, vp = archPQ_verified_state(xfw(wp), raw_cutoffs, ctx_cm)
    _, vm = archPQ_verified_state(xfw(wm), raw_cutoffs, ctx_cm)
    fd = (vp.Delta_dual - vm.Delta_dual) / (2h)
    @printf("%6d %14.6e %14.6e %12.4e %10.4f\n", j, g_econ_A[j], fd, h, fd == 0.0 ? NaN : g_econ_A[j] / fd)
    flush(stdout)
end

println()
println(ALL_PASS[] ? "ALL PAIRWISE-QUANTILE OUTER-GRADIENT FD CHECKS PASSED" : "SOME CHECKS FAILED")
exit(ALL_PASS[] ? 0 : 1)
