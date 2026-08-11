# ================================================================================================
# VERSION-A / VERSION-B EQUIVALENCE ANCHOR for the pairwise-quantile-independence restriction.
#
# Version A made the quantile CUTOFFS free and pinned the moment targets at the constants `1/L`,
# `1/L^2`. Version B fixes the cutoffs and frees the bin MASSES `mu_{o,a}`, with targets `mu_{o,a}`
# and `mu_{o,a}*mu_{p,b}`. Set the fixed cutoffs to the draws' own empirical quantiles AND pin
# `mu_{o,a} = 1/L`, and the two moment matrices are IDENTICAL, element for element:
#
#     version A at its own start point : 1{b_o=a} - 1/L        1{b_o=a, b_p=b} - 1/L^2
#     version B here                   : 1{b_o=a} - mu_{o,a}   1{b_o=a, b_p=b} - mu_{o,a} mu_{p,b}
#                                        with mu == 1/L        so mu*mu == 1/L^2
#
# so `Delta*` must reproduce version A's to solver reproducibility. This is the cheapest and
# strongest single check that the mu-centering and the (o,a)/(p,b) indexing are right: a centering
# or indexing error moves `Delta*` far more than the tolerances below, and it costs ONE inner solve.
#
# REFERENCE VALUES (version A, recorded 2026-08-10 in
# docs/PAIRWISE_QUANTILE_OUTER_LOOP_STATUS_2026-08-10.md, at the real D=20 calibration point with
# the production scientific settings below):
#
#     W =  20,000  L =  3  ->  Delta_dual = 0.005235   (VerifiedSolved, 4.4 s)
#     W = 100,000  L =  5  ->  Delta_dual = 0.000706   (VerifiedSolved, 39.9 s)
#     W = 100,000  L = 10  ->  Delta_dual = 0.003011   (VerifiedSolved, 1084.6 s)
#
# The recorded figures carry 4 significant digits, so the gate below is a RELATIVE one at that
# resolution. A tighter (bit-level) comparison is available by re-running version A itself from a
# git worktree at the pre-reparameterization commit; when that is done, the observed agreement is
# reported in the session doc rather than hard-coded here as a magic constant.
#
# Also checked, and cheap: the fixed cutoffs under `:empirical_quantile` reproduce version A's own
# `pairwise_quantile_start_cutoffs` construction, and `mu = 1/L` decodes exactly.
#
# Run:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --project=. \
#     full_aod_diag/d4_exact/test_pairwise_quantile_version_ab_anchor.jl [W] [L] [expected_Delta]
# ================================================================================================
_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "multistart_seed_generator.jl",   # build_aspace_geometry / cm_w0_from_calibration
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, SpecialFunctions

lp(xs...) = (println(xs...); flush(stdout))
ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool, detail::AbstractString = "")
    global ALL_PASS[] &= cond
    lp(cond ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
end

const W_A   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
const L_A   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
# Version A's own recorded Delta_dual at (W,L), 4 significant digits (see header).
const REFERENCE = Dict((20_000, 3) => 0.005235, (100_000, 5) => 0.000706, (100_000, 10) => 0.003011)
const EXPECTED = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : get(REFERENCE, (W_A, L_A), NaN)
const GRAV = default_gravity_exclude_cells_brazil_korea()

lp("="^100)
lp("VERSION-A / VERSION-B EQUIVALENCE ANCHOR: W=", W_A, " L=", L_A,
   "  cutoffs=:empirical_quantile  mu=1/L")
lp("version-A reference Delta_dual = ", EXPECTED)
lp("="^100)

# Production scientific settings, matching exactly what the version-A reference values were recorded
# under (docs/PAIRWISE_QUANTILE_OUTER_LOOP_STATUS_2026-08-10.md): sigma=3.0, sobol_randomized,
# seed=20260719, exclude_row, inner_lower_limit=-10, Brazil-Korea gravity exclusions, delta=50 (so
# the early-abort threshold is Inf and cannot be mistaken for a failure).
ctx_raw = d20_real_setup_design(W = W_A, δ = 50.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("ctx: D=", ctx.D, " D_dest=", ctx.D_dest, " W=", ctx.W, " sigma=", ctx.σ)

layout = PairwiseQuantileMassLayout(ctx.D, L_A)
MIN_BIN_COUNT = max(10, W_A ÷ (2 * L_A^2))
pcx = build_pairwise_quantile_production_context(ctx, layout;
    cutoff_source = :empirical_quantile, min_bin_count = MIN_BIN_COUNT)
ctx_cm = pcx.ctx_cm

# --- 1. the fixed cutoffs reproduce version A's own empirical-quantile construction -------------
# Version A built its start point by taking each origin's empirical r/L quantile, then encoding it
# through the softplus-ordered transform (log q_1, log(expm1(gap_k))) and decoding it back. That
# round-trip is exact in exact arithmetic; this reproduces the same quantiles directly, and the
# check below confirms the two constructions agree to floating point.
function version_a_style_quantiles(U, L)
    W, D = size(U)
    Q = Matrix{Float64}(undef, L - 1, D)
    for o in 1:D
        Uo = sort(@view U[:, o])
        n = length(Uo)
        for r in 1:L-1
            Q[r, o] = Uo[clamp(round(Int, (r / L) * n), 1, n)]
        end
    end
    return Q
end
Q_a = version_a_style_quantiles(ctx.U, L_A)
check("fixed cutoffs == version A's own empirical-quantile construction",
      ctx_cm.pq_cutoffs == Q_a,
      "max|diff|=$(maximum(abs, ctx_cm.pq_cutoffs .- Q_a))")
# Bin-assignment checksum, printed so it can be compared against the version-A reference run's own
# (ref_versionA_anchor_point.jl in the detached version-A worktree). Agreement is NOT automatic:
# version A reached its cutoffs through a log/softplus encode-decode round trip, and a draw sitting
# exactly ON a cutoff (the empirical quantile IS one of the draws) could land in a different bin
# under a 1-ulp difference. If the checksums match, the two versions partition the same draws the
# same way and Delta* must agree to solver reproducibility, not merely to the 4 significant digits
# of the recorded reference.
lp("bin checksum: sum(bin) = ", sum(Int(b) for b in ctx_cm.pq_op.bin),
   "   hash(bin) = ", hash(ctx_cm.pq_op.bin))
lp("fixed cutoffs, origin 1 = ", ctx_cm.pq_cutoffs[:, 1])

# --- 1b. the ONE thing that stops this anchor being bit-exact, measured rather than assumed -------
# Version A never used the empirical quantiles directly: it ENCODED them into raw outer coordinates
# (`log q_1`, `log(expm1(gap_k))`) and DECODED them back (`exp`, `softplus`) on every outer point.
# That round trip is exact in real arithmetic and 1-ulp-lossy in floating point, and the empirical
# quantile IS one of the draws -- so a draw sitting exactly ON a cutoff can land in a different bin
# under a decoded cutoff that is one ulp below the quantile. This block reproduces version A's own
# transform (the production copy was deleted with the rest of the cutoff machinery, so it is
# restated here, marked as a diagnostic) and COUNTS the affected draws, so the residual Delta*
# difference below is attributed rather than hand-waved.
_va_softplus(x::Float64) = x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x))
function version_a_roundtrip_cutoffs(Q::Matrix{Float64})
    nc, D = size(Q)
    Qout = similar(Q)
    for o in 1:D
        raw = Vector{Float64}(undef, nc)
        raw[1] = log(Q[1, o])
        for k in 2:nc
            raw[k] = log(expm1(log(Q[k, o]) - log(Q[k-1, o])))
        end
        logq = Vector{Float64}(undef, nc)
        logq[1] = raw[1]
        for k in 2:nc
            logq[k] = logq[k-1] + _va_softplus(raw[k])
        end
        for k in 1:nc
            Qout[k, o] = exp(logq[k])
        end
    end
    return Qout
end
Q_rt = version_a_roundtrip_cutoffs(ctx_cm.pq_cutoffs)
op_rt = PairwiseQuantileOperator(ctx.U, L_A, Q_rt)
n_bin_diff = count(!=(0), Int.(op_rt.bin) .- Int.(ctx_cm.pq_op.bin))
n_assign = ctx.D * ctx.W
lp("version-A log/softplus round trip moves ", n_bin_diff, " of ", n_assign,
   " bin assignments (", round(100 * n_bin_diff / n_assign, sigdigits = 3), "%);",
   " max|Q_roundtrip - Q| = ", maximum(abs, Q_rt .- ctx_cm.pq_cutoffs))
# Every moved draw must be one that sits EXACTLY on a cutoff -- that is the whole mechanism. If a
# draw strictly inside a bin moved, the explanation would be wrong and something else is going on.
on_boundary = 0
for o in 1:ctx.D, w in 1:ctx.W
    op_rt.bin[w, o] == ctx_cm.pq_op.bin[w, o] && continue
    any(r -> ctx.U[w, o] == ctx_cm.pq_cutoffs[r, o], 1:L_A-1) && (global on_boundary += 1)
end
check("every bin assignment the round trip moves is a draw sitting exactly ON a cutoff",
      on_boundary == n_bin_diff, "$on_boundary of $n_bin_diff")
check("the round trip moves at most one draw per (origin, cutoff)",
      n_bin_diff <= ctx.D * (L_A - 1), "$n_bin_diff moved, $(ctx.D * (L_A - 1)) cutoffs")

# --- 2. mu = 1/L decodes exactly -----------------------------------------------------------------
mass0 = uniform_mass_raw(layout)
st_chk = PairwiseQuantileMassState(ctx.D, L_A)
set_pairwise_quantile_masses!(st_chk, mass0, layout)
check("uniform_mass_raw decodes to mu == 1/L exactly (max|mu - 1/L|)",
      maximum(abs, st_chk.mu .- 1.0 / L_A) < 1e-15 && maximum(abs, st_chk.mu_last .- 1.0 / L_A) < 1e-15,
      "max|mu-1/L|=$(maximum(abs, st_chk.mu .- 1.0/L_A))")

# --- 3. the moment matrix's centering constants ARE version A's constants ------------------------
# The direct statement of the equivalence, checked on the target vector itself rather than inferred
# from Delta*: every marginal row's constant must be exactly 1/L and every pair row's exactly 1/L^2.
tvec = zeros(n_total_rows(ctx.D, L_A))
pairwise_quantile_target_vector!(tvec, ctx_cm.pq_op, st_chk)
nm = n_marginal_rows(ctx.D, L_A)
check("marginal-row centering constants are exactly 1/L",
      maximum(abs, tvec[1:nm] .- 1.0 / L_A) < 1e-16,
      "max|t - 1/L|=$(maximum(abs, tvec[1:nm] .- 1.0/L_A))")
check("pair-row centering constants are exactly 1/L^2",
      maximum(abs, tvec[nm+1:end] .- 1.0 / L_A^2) < 1e-16,
      "max|t - 1/L^2|=$(maximum(abs, tvec[nm+1:end] .- 1.0/L_A^2))")

# --- 4. and therefore Delta* must reproduce version A's -----------------------------------------
geo = build_aspace_geometry(ctx)
w_cal = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
xf = x_free_from_w(vcat(w_cal[1], cm_z_from_a(w_cal[2:end], cm_fixed_theta(ctx),
    precompute_cm_aspace_xy(ctx), geo.pe)), geo.pe)

lp("\nstarting REAL KNITRO inner solve ...")
t0 = time()
base, verify = archPQ_verified_state(xf, mass0, ctx_cm)
wall = time() - t0
@printf("Delta_dual = %.17g   inner_status = %d   class = %s   n_fg = %d   n_hess = %d   %.1fs\n",
        verify.Delta_dual, verify.inner_status, string(classify_inner_result(verify)),
        verify.n_fg, verify.n_hess, wall)
lp("block KKT: E = ", verify.kkt_resid_E, "  marginalbin = ", verify.kkt_resid_marginalbin,
   "  pairindep = ", verify.kkt_resid_pairindep)
check("inner solve is VerifiedSolved", classify_inner_result(verify) == VerifiedSolved)

if isfinite(EXPECTED)
    rel = abs(verify.Delta_dual - EXPECTED) / abs(EXPECTED)
    @printf("relative difference vs version A's reference value: %.3e\n", rel)
    # Threshold 5e-4. Two things set it, neither of them a tuning knob:
    #   - the recorded reference in the status doc carries 4 significant digits, so 5e-4 is the
    #     resolution of that number itself;
    #   - even against a full-precision live re-run of version A, the two are NOT expected to be
    #     bit-identical, because of the round-trip effect measured in section 1b: a handful of draws
    #     out of D*W sit exactly on a cutoff and change bin under version A's 1-ulp-lossy
    #     encode/decode. That is a real (tiny) difference in the partition, not an error in either
    #     version, and it is reported above so the residual here is attributed rather than assumed.
    # A centering or indexing error would miss by orders of magnitude, not by this.
    check("Delta_dual reproduces version A's reference value to the resolution of that record",
          rel < 5e-4, "rel=$rel")
else
    lp("NOTE: no recorded version-A reference for (W=$W_A, L=$L_A) -- value reported, not gated. ",
       "Pass the reference as ARGS[3] to gate it.")
end

# --- 5. closing the loop: at version A's OWN (round-tripped) cutoffs, agreement is BIT-EXACT -----
# Sections 1b and 4 together say the residual difference is entirely the 2-draw partition change,
# but they say it by attribution. This checks it directly: rebuild the version-B context on `Q_rt`
# -- the cutoffs version A actually decoded, one ulp from the empirical quantiles -- so the two
# versions now partition the draws identically. The moment matrices are then element-for-element
# equal, and `Delta*` must agree to solver reproducibility, not to 4 significant digits.
#
# The context is assembled by hand here rather than through
# `build_pairwise_quantile_production_context`, deliberately: that function derives its cutoffs from
# an explicit `cutoff_source` and has no override, and adding one just to let a test inject cutoffs
# would put a silent back door in the production surface (CLAUDE.md). Everything below uses the same
# public constructors the production builder uses, in the same order.
if isfinite(EXPECTED) && n_bin_diff > 0
    lp("\n--- 5. re-solve at version A's OWN round-tripped cutoffs (identical partition) ---")
    aug_rt = build_pairwise_quantile_augmented_obj(ctx, layout, Q_rt)
    mass_rt = PairwiseQuantileMassState(ctx.D, L_A)
    hess_rt = PairwiseQuantileCoreHessCtx(aug_rt.ncore_econ, aug_rt.op, mass_rt, aug_rt.core_cf_ref)
    ctx_rt = merge(ctx, (obj = aug_rt.obj_pq, pq_op = aug_rt.op, pq_mass_state = mass_rt,
                         pq_core_cf_ref = aug_rt.core_cf_ref, pq_hess_ctx = hess_rt,
                         pq_econ_ctx = ctx, pq_layout = layout, pq_cutoffs = Q_rt,
                         pq_cutoff_source = :version_a_roundtrip_diagnostic,
                         pq_min_bin_count = MIN_BIN_COUNT))
    check("round-tripped context reproduces version A's bin assignment exactly",
          aug_rt.op.bin == op_rt.bin)
    _, verify_rt = archPQ_verified_state(xf, mass0, ctx_rt)
    @printf("Delta_dual at version A's own cutoffs = %.17g   (version A: %.17g)\n",
            verify_rt.Delta_dual, EXPECTED)
    rel_rt = abs(verify_rt.Delta_dual - EXPECTED) / abs(EXPECTED)
    @printf("relative difference: %.3e\n", rel_rt)
    # 1e-12: solver reproducibility, not "close". Both sides solve the SAME convex problem from the
    # same start with the same deterministic KNITRO settings, so anything above round-off here would
    # mean the moment matrices are not in fact identical.
    check("at an identical partition, version B reproduces version A to solver reproducibility",
          rel_rt < 1e-12, "rel=$rel_rt")
end

lp("")
lp(ALL_PASS[] ? "VERSION-A/B EQUIVALENCE ANCHOR PASSED" : "ANCHOR FAILED -- see above")
exit(ALL_PASS[] ? 0 : 1)
