# ================================================================================================
# Reoptimized-FD validation gate for the pairwise-quantile-independence restriction's OUTER gradient,
# VERSION B: fixed cutoffs + FREE bin masses (2026-08-10).
#
# This is the gate that must pass before the outer gradient may be called correct. It is the direct
# analog of what this codebase already holds origin-ZC's own nu-gradient to
# (`d_delta_dual_d_eta_origin_fd` + `test_cm_originzc_pure_moments.jl`'s testset, where every FD
# probe RE-SOLVES the inner dual from scratch and the tolerance is tightened by SHRINKING h, never
# by loosening the threshold).
#
# THE BAR WENT UP FROM VERSION A, NOT DOWN -- read this before adjusting any tolerance below.
# Version A's outer coordinates were the quantile CUTOFFS, which sit inside indicator functions, so
# `Delta_dual` was a genuine STEP function of every one of them: an FD probe smaller than the gap to
# the next draw crossed nothing and returned exactly 0.0, both sides had to be matched-bandwidth
# secants of the same staircase, and ~20% relative agreement was the best available. Version B's
# coordinates shift moment TARGETS smoothly and never move a draw between bins, so `Delta_dual` is
# smooth in them, a plain small-`h` reoptimized central FD is a VALID ground truth, and agreement
# should be at the level origin-ZC's own gate achieves. A few percent is NOT acceptable here -- the
# loose version-A tolerance was a property of the step-function objective, not a standard to
# inherit. If this gate does not pass, do NOT reintroduce a bandwidth to make it look better.
#
# What is checked, in order:
#   1. Delta_dual is produced by the verified-state layer, and the solve verifies.
#   2. The analytic MASS block vs reoptimized FD, with an h-shrinking ladder, at a NON-UNIFORM mu
#      (see below), plus two negative controls that must fail.
#   3. The analytic ECONOMIC block: that this restriction being active does not disturb the shared,
#      family-agnostic (g, A_od) gradient -- gated by q0 exactness and bit-identity, NOT by a
#      small-h FD (see section 3's own comment).
#   4. The COMBINED vector's layout: that vcat(g_econ, g_mass) lines up index-for-index with the
#      outer coordinate vector w = vcat(gp, zfree, raw_masses) the driver hands KNITRO.
#
# WHY THE GATE IS ENFORCED AT A NON-UNIFORM mu. At `mu = 1/L` every bin's mass is equal, so the
# pair term's partner index is unobservable: `mu[p,b]` and `mu[p,a]` are numerically identical and
# a transcription error between them is invisible. The enforced point therefore perturbs the masses
# away from uniform, per origin and per bin. The uniform point is still reported, because it is the
# campaign's own starting point and worth seeing.
#
# Usage:  julia --project=. full_aod_diag/d4_exact/test_pairwise_quantile_outer_gradient_fd.jl [L] [cutoff_source]
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
          "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl", "cm_screen_bridge.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
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
const CUTOFF_SOURCE = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :empirical_quantile

println("=== pairwise-quantile OUTER gradient (FREE MASSES): reoptimized-FD gate ===")
println("L = ", PQ_L, "  cutoff_source = :", CUTOFF_SOURCE)
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
W = size(ctx.U, 1)
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)
# Floor at half the EXPECTED joint-cell occupancy (W/L^2): data-derived from this test's own W and
# L rather than a typed-in constant, and strict enough that a genuinely starved cell still fails.
const MIN_BIN_COUNT = max(10, W ÷ (2 * PQ_L^2))
pcx = build_pairwise_quantile_production_context(ctx, layout;
    cutoff_source = CUTOFF_SOURCE, min_bin_count = MIN_BIN_COUNT)
ctx_cm = pcx.ctx_cm
println("W = ", W, "  D = ", ctx.D, "  n_raw = ", n_raw(layout), "  n_econ_free = ", length(x_free_calib),
        "  min_bin_count = ", MIN_BIN_COUNT)
flush(stdout)

raw_uniform = uniform_mass_raw(layout)
# The enforced point: masses pushed away from uniform, differently at every (origin, bin), so the
# pair term's partner index and the stick-breaking chain rule are both genuinely exercised.
raw_off = copy(raw_uniform)
for o in 1:ctx.D, k in 1:n_free_bins(layout)
    raw_off[raw_index(layout, o, k)] += 0.30 * sin(2.7 * o + 1.3 * k)
end

# ---------------------------------------------------------------------------------------------
# 1. The objective value the outer loop consumes
# ---------------------------------------------------------------------------------------------
println("\n=== 1. verified state produces Delta_dual ===")
t0 = time()
base, verify = archPQ_verified_state(x_free_calib, raw_uniform, ctx_cm)
println("inner_status = ", verify.inner_status, "  Delta_dual = ", verify.Delta_dual,
        "  Delta_primal = ", verify.Delta_primal, "  (", round(time() - t0, digits = 2), "s)")
println("primal_dual_gap = ", verify.primal_dual_gap, "  mean_m_resid = ", verify.mean_m_resid,
        "  max_abs_moment_kkt_resid = ", verify.max_abs_moment_kkt_resid)
println("block KKT: E = ", verify.kkt_resid_E, "  marginalbin = ", verify.kkt_resid_marginalbin,
        "  pairindep = ", verify.kkt_resid_pairindep)
println("restriction residuals under the LFD: max marginal cum = ", verify.max_marginal_cumulative_residual,
        "  max joint cum = ", verify.max_cumulative_residual)
check("Delta_dual is finite", isfinite(verify.Delta_dual), "Delta_dual=$(verify.Delta_dual)")
check("verify carries r_current of length W", length(verify.r_current) == W)
check("verify carries the masses this solve ran at", size(verify.mu) == (ctx.D, n_free_bins(layout)))
check("at mu=1/L the verifier sees uniform masses", maximum(abs, verify.mu .- 1.0 / PQ_L) < 1e-14)
cls = classify_inner_result(verify)
println("classify_inner_result = ", cls,
        cls == VerifiedSolved ? "" : "  reasons=$(verification_rejection_reasons(verify))")
check("inner solve is VerifiedSolved under the SHARED acceptance gate", cls == VerifiedSolved)

# ---------------------------------------------------------------------------------------------
# 2. MASS block: exact closed form vs reoptimized FD
# ---------------------------------------------------------------------------------------------
println("\n=== 2. mass block: analytic (exact closed form) vs reoptimized FD ===")

"""
Runs the analytic-vs-reoptimized-FD comparison at `raw`, with an h-shrinking ladder. Returns the
best (smallest relative L2) result. `enforce` gates it.

The ladder exists for the reason `test_cm_originzc_pure_moments.jl`'s own does: at a fixed `h` the
central FD carries both truncation error (falling as h^2) and inner-solver-tolerance noise (rising
as 1/h), so the right response to disagreement is to move `h`, not to widen the threshold. See
memory `feedback-fd-bandwidth-mismatch-looks-like-a-bug`.
"""
function run_mass_gate(label::AbstractString, raw::Vector{Float64}, enforce::Bool,
                       hs::Tuple)
    println("\n--- mass gate at: ", label, " ---")
    b, v = archPQ_verified_state(x_free_calib, raw, ctx_cm)
    println("Delta_dual = ", v.Delta_dual, "  class = ", classify_inner_result(v))
    enforce && check("[$label] inner solve VerifiedSolved", classify_inner_result(v) == VerifiedSolved)
    ga = pairwise_quantile_mass_gradient_vec(b, v, ctx_cm, raw)
    enforce && check("[$label] analytic mass gradient all-finite", all(isfinite, ga))

    best = (h = NaN, rel = Inf, cosang = NaN, maxdiff = Inf, gf = fill(NaN, length(ga)))
    @printf("%10s %14s %14s %14s %10s\n", "h", "rel L2", "max|diff|", "cosine", "secs")
    for h in hs
        t = time()
        (gf, n_probed) = d_delta_dual_d_mass_fd(x_free_calib, raw, ctx_cm; h = h)
        rel = norm(ga .- gf) / max(norm(gf), eps())
        cosang = dot(ga, gf) / (norm(ga) * norm(gf) + 1e-300)
        maxdiff = maximum(abs, ga .- gf)
        @printf("%10.0e %14.3e %14.3e %14.10f %10.1f\n", h, rel, maxdiff, cosang, time() - t)
        flush(stdout)
        rel < best.rel && (best = (h = h, rel = rel, cosang = cosang, maxdiff = maxdiff, gf = gf))
    end

    println("\nbest: h=", best.h, "  relative L2 = ", best.rel, "  cosine = ", best.cosang,
            "  max|diff| = ", best.maxdiff)
    @printf("%6s %16s %16s %10s\n", "coord", "analytic", "reopt-FD", "ratio")
    for j in 1:min(length(ga), 12)
        @printf("%6d %16.8e %16.8e %10.5f\n", j, ga[j], best.gf[j],
                best.gf[j] == 0.0 ? NaN : ga[j] / best.gf[j])
    end

    if enforce
        check("[$label] cosine(analytic, reoptimized FD) > 0.9999", best.cosang > 0.9999,
              "cosine=$(best.cosang)")
        # Threshold: 1e-6. The handover set the expectation at ~1e-5 relative (origin-ZC's own
        # standard); the measured value at D=4/L=5 on 2026-08-10 was 4.1e-10 at h=1e-5, falling as
        # h^2 down the ladder (1.15e-6 -> 1.15e-8 -> 4.1e-10), so this gate sits STRICTER than the
        # stated expectation and still ~2400x above what the code actually achieves. It is not a
        # threshold tuned until green: version A could only reach ~0.2 here, and "a few percent"
        # would mean something is wrong.
        check("[$label] relative L2 error < 1e-6 (version A could only reach ~0.2)", best.rel < 1e-6,
              "rel=$(best.rel)")
        # Two negative controls. A gate nobody has watched fail is not known to be a gate, and these
        # two failure modes are exactly the ones this family has form for.
        #
        # (a) SIGN. `d_delta_dual_d_mu` differentiates Delta_dual directly (its -mean_m prefactor IS
        #     the Delta = -f sign), so unlike version A's cutoff gradient nothing is negated at the
        #     production layer. If someone "restores" a negation, this must fail.
        cos_flip = dot(-ga, best.gf) / (norm(ga) * norm(best.gf) + 1e-300)
        check("[$label] NEGATIVE CONTROL: a sign-flipped gradient anti-correlates with FD",
              cos_flip < -0.9999, "cosine_flipped=$cos_flip")
        # (b) PRODUCT TERM. Dropping the sum_{p != o} mu_p * lambda_pair term leaves the marginal
        #     duals alone -- a gradient that still looks plausible and still points roughly the right
        #     way. It must not survive the relative-L2 gate.
        op = ctx_cm.pq_op
        ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
        lM, lP = reshape_pq_duals(b.λstar, op, ncore1)
        d_mu_nopair = -v.m_mean .* collect(lM)
        ga_nopair = chain_mass_gradient_to_raw(d_mu_nopair, raw, ctx_cm.pq_mass_state.mu, ctx_cm.pq_layout)
        rel_nopair = norm(ga_nopair .- best.gf) / max(norm(best.gf), eps())
        check("[$label] NEGATIVE CONTROL: dropping the pair product term fails the same gate",
              rel_nopair > 1e-2, "rel_without_pair_term=$rel_nopair")
    end
    return (ga = ga, best = best)
end

# (a) the campaign's own starting point -- reported, not enforced: at uniform masses the pair term's
#     partner index is unobservable (every bin has the same mass), so agreement here is weak
#     evidence however good it looks.
res_uniform = run_mass_gate("uniform masses mu = 1/L (start point; weak test)", copy(raw_uniform), false,
                            (1e-4,))

# (b) the enforced point.
res_off = run_mass_gate("non-uniform masses (enforced)", copy(raw_off), true, (1e-3, 1e-4, 1e-5))

g_mass = pairwise_quantile_mass_gradient_vec(base, verify, ctx_cm, raw_uniform)
check("analytic mass gradient is not identically zero at the start point",
      maximum(abs, g_mass) > 0.0, "max|g_mass|=$(maximum(abs, g_mass))")

# ---------------------------------------------------------------------------------------------
# 3. ECONOMIC block (is the shared block disturbed by this restriction?)
# ---------------------------------------------------------------------------------------------
println("\n=== 3. economic block ===")
g_full, meta = pairwise_quantile_production_gradient(x_free_calib, raw_uniform, pcx, ctx, pe;
    base = base, verify = verify, threaded = false, h_mode = :cached)
D2_econ = ctx.D * (hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D)
println("length(g_full) = ", length(g_full), "  D2_econ = ", D2_econ, "  n_raw = ", n_raw(layout))
check("combined gradient length == D*Ddest + n_raw(layout)", length(g_full) == D2_econ + n_raw(layout))
check("combined gradient is all-finite", all(isfinite, g_full))
check("combined gradient's mass tail == the standalone mass gradient",
      g_full[D2_econ+1:end] == g_mass)

# --- 3a. The DECISIVE check on the restriction's effect on the economic block: q0 exactness ------
# `build_lfix_base_cache_pairwise_quantile` folds this restriction's G_R*lambda_R into the cache's
# q0 and cross-checks the result against `verify.r_current`, which is recomputed by a completely
# independent route (the verifier's own fresh forward pass). Both are the same quantity
# r = -zeta - E*lambda_E - G_R*lambda_R, so agreement to floating point is an EXACT statement that
# the economic block is being linearized about the right base point with the restriction active --
# far stronger than any finite-difference comparison, and the check that caught the real bug here
# (an un-folded q0, initially rationalized away as "the restriction rows don't depend on theta").
# The reparameterization does not retire this: q0 is a LEVEL, not a derivative. See memory
# `feedback-q0-restriction-fold-is-a-level-not-a-derivative`.
ensure_pq_masses!(ctx_cm, raw_uniform)
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
# and deliberately does NOT finite-difference it. The reason matters: the economic A-block gradient
# is itself an ADAPTIVE BANDWIDTH-SELECTED secant (`select_bandwidth`, composite_gradient.jl), whose
# own docstring states that a probe smaller than its h_floor=1e-4 "would just reproduce Method-A's
# known-wrong winner-boundary-dropping gradient". A small-h reoptimized FD therefore measures a
# DIFFERENT (and, per this codebase's own analysis, wrong) estimator -- it is not a ground truth for
# this block, and an earlier draft of this gate that treated it as one produced a confident-looking
# failure on exactly the coordinates where the winner-boundary term dominates. See memory
# `feedback-fd-bandwidth-mismatch-looks-like-a-bug`.
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
xfw(w_econ) = vcat(w_econ[1], vec(exp.(pivot_expand(w_econ[2:end], pe))))
A_calib = reshape(collect(x_free_calib[2:end]), ctx.D, hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D)
w_econ0 = vcat(x_free_calib[1], pivot_reduce(log.(A_calib), pe))
check("w_econ round-trips to the calibration x_free", isapprox(xfw(w_econ0), collect(x_free_calib); rtol = 1e-8),
      "max abs diff = $(maximum(abs, xfw(w_econ0) .- collect(x_free_calib)))")
g_econ_A = g_full[1:D2_econ]
econ_coords = collect(2:min(5, D2_econ))
println("\nmatched-bandwidth economic FD (DIAGNOSTIC, not gated), coords ", econ_coords, ":")
@printf("%6s %14s %14s %12s %10s\n", "coord", "analytic", "reopt-FD", "h_used", "ratio")
for j in econ_coords
    h = meta_ref.h_used[j]
    h > 0 || continue
    wp = copy(w_econ0); wp[j] += h
    wm = copy(w_econ0); wm[j] -= h
    _, vp = archPQ_verified_state(xfw(wp), raw_uniform, ctx_cm)
    _, vm = archPQ_verified_state(xfw(wm), raw_uniform, ctx_cm)
    fd = (vp.Delta_dual - vm.Delta_dual) / (2h)
    @printf("%6d %14.6e %14.6e %12.4e %10.4f\n", j, g_econ_A[j], fd, h, fd == 0.0 ? NaN : g_econ_A[j] / fd)
    flush(stdout)
end

println()
println(ALL_PASS[] ? "ALL PAIRWISE-QUANTILE OUTER-GRADIENT FD CHECKS PASSED" : "SOME CHECKS FAILED")
exit(ALL_PASS[] ? 0 : 1)
