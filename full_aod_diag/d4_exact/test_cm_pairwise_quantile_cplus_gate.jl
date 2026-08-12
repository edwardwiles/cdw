# ================================================================================================
# GATE: family #7's Backend C+ (factorized) outer gradient against its DENSE reference.
#
# `cm_pairwise_quantile_cplus.jl` recomputes the ECONOMIC half of the outer gradient through the
# factorized price representation (never materializing the W x D x Ddest tensors) instead of the
# dense `build_lfix_base_cache` path. The restriction half is untouched. So:
#
#   1. the two gradients must agree -- the SAME quantity by two representations;
#   2. the RESTRICTION tail must be BIT-IDENTICAL (nothing about it changed at all);
#   3. the q0 fold must still be exact against the independently recomputed r, in BOTH paths --
#      and for THIS family that means BOTH restriction blocks in both paths, which is checked with
#      the same negative control the dense-path gate uses;
#   4. C+ must not be slower, which is the entire point.
#
# Tolerance on (1): the two are different representations of the same arithmetic, not a
# reassociation of the same one, so bit-identity is NOT claimed. `c23_cplus_gate.jl` -- this repo's
# own established C+ gate -- holds its check to atol/rtol 1e-6; the same standard is used here, and
# the MEASURED agreement is printed so a regression is visible even inside tolerance.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. -t N \
#     full_aod_diag/d4_exact/test_cm_pairwise_quantile_cplus_gate.jl [W] [L] [G] [n_families] [contrasts]
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
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl",
          "cm_pairwise_quantile_cplus.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf

lp(xs...) = (println(xs...); flush(stdout))
ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool, detail::AbstractString = "")
    global ALL_PASS[] &= cond
    lp(cond ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  (" * detail * ")")
end

const W_G     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const L_G     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
const G_G     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
const NFAM_G  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2
const CONTR_G = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :orthonormal
const GRAV    = default_gravity_exclude_cells_brazil_korea()

lp("="^96)
lp("CM + PAIRWISE-QUANTILE Backend C+ GATE: W=", W_G, " L=", L_G, " G=", G_G,
   " families=", NFAM_G, " contrasts=:", CONTR_G, "  julia threads=", Threads.nthreads())
lp("="^96)

ctx_raw = d20_real_setup_design(W = W_G, δ = 50.0, find_smallest = true, draw_design = :pseudorandom,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0,
    inner_loop_opt = joinpath(dirname(_D4E), "ek_inner_cmpq.opt"))
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
cfg = CMPairwiseQuantileConfig(L = L_G, cm_grid_size = G_G, cm_moment_families = NFAM_G,
                               contrasts = CONTR_G, min_bin_count = max(10, W_G ÷ (2 * L_G^2)),
                               mass_start = :uniform)
pcx = build_cm_pairwise_quantile_production_context(ctx, cfg;
    inner_opt = joinpath(dirname(_D4E), "ek_inner_cmpq.opt"))
ctx_cm = pcx.ctx_cm
geo = build_aspace_geometry(ctx); pe = geo.pe
w_cal = cm_w0_from_calibration(ctx, pe, :powered_aspace)
xf = x_free_from_w(vcat(w_cal[1], cm_z_from_a(w_cal[2:end], cm_fixed_theta(ctx),
    precompute_cm_aspace_xy(ctx), pe)), pe)
mass0 = cmpq_uniform_mass_raw(L_G)

lp("solving once (both paths reuse this base/verify, so ONLY the gradient differs) ...")
base, verify = archCMPQ_verified_state(xf, mass0, ctx_cm)
lp("Delta_dual = ", verify.Delta_dual, "  class = ", classify_inner_result(verify),
   "  n_fg=", verify.n_fg, "  n_hess=", verify.n_hess)
check("inner solve VerifiedSolved", classify_inner_result(verify) == VerifiedSolved)

econ_ws = get_or_build_econ_a_grad_ws(W_G)
pool, wsC = cm_pairwise_quantile_cplus_workspaces(ctx_cm)
D2_econ = ctx.D * ctx.D_dest
kw = (threaded = true, h_mode = :cached)

# warm both paths (JIT), then measure
cm_pairwise_quantile_production_gradient(xf, mass0, pcx, ctx, pe; base = base, verify = verify,
    econ_ws = econ_ws, bandwidth_cache = Dict{Int,Float64}(), kw...)
cm_pairwise_quantile_production_gradient_cplus(xf, mass0, pcx, ctx, pe, pool, wsC; base = base,
    verify = verify, bandwidth_cache = Dict{Int,Float64}(), kw...)

# INTERLEAVED, and repeated: this box runs other campaigns and has been at load 70+ all session, so
# a single dense-then-cplus pair is not a measurement (memory: a single-run A/B here once produced
# an apparent 18% regression that was pure load drift). Alternating the arms makes both pay the same
# drift, and the MINIMUM over repeats is reported alongside the mean.
const NREP = 3
"Wrapped in a FUNCTION rather than looping at top level: a `for` body is a SOFT scope, so the
accumulating assignments would need `global` declarations that then conflict with any outer `local`.
A function body is a hard scope and the whole question disappears -- the same reason this family's
other gates wrap their try/catch flags."
function timed_reps(nrep::Int)
    td = Float64[]; tc = Float64[]
    gd = Float64[]; gc = Float64[]
    for rep in 1:nrep
        bwc_d = Dict{Int,Float64}(); bwc_c = Dict{Int,Float64}()
        t1 = @elapsed (gd, _) = cm_pairwise_quantile_production_gradient(xf, mass0, pcx, ctx, pe;
            base = base, verify = verify, econ_ws = econ_ws, bandwidth_cache = bwc_d, kw...)
        t2 = @elapsed (gc, _) = cm_pairwise_quantile_production_gradient_cplus(xf, mass0, pcx,
            ctx, pe, pool, wsC; base = base, verify = verify, bandwidth_cache = bwc_c, kw...)
        push!(td, t1); push!(tc, t2)
        @printf("  rep %d: dense %.3f s   cplus %.3f s\n", rep, t1, t2); flush(stdout)
    end
    return (td, tc, gd, gc)
end
td, tc, g_dense, g_cplus = timed_reps(NREP)

lp("")
check("both gradients have the same length",
      length(g_dense) == length(g_cplus) == D2_econ + n_cmpq_raw(L_G),
      "$(length(g_dense)) == $D2_econ + $(n_cmpq_raw(L_G))")
gd_e = g_dense[1:D2_econ];      gc_e = g_cplus[1:D2_econ]
gd_r = g_dense[D2_econ+1:end];  gc_r = g_cplus[D2_econ+1:end]
maxabs = maximum(abs, gd_e .- gc_e)
relerr = maxabs / max(maximum(abs, gd_e), eps())
@printf("economic block: max|dense - cplus| = %.3e   relative = %.3e\n", maxabs, relerr)
check("economic block agrees to the repo's own C+ standard (c23_cplus_gate: atol/rtol 1e-6)",
      isapprox(gd_e, gc_e; atol = 1e-6, rtol = 1e-6), "max|diff|=" * string(maxabs))
check("RESTRICTION tail is BIT-IDENTICAL (nothing about it changed)", gd_r == gc_r,
      "max|diff|=" * string(maximum(abs, gd_r .- gc_r)))

# q0 exactness must survive the port -- in BOTH paths, and for BOTH restriction blocks.
cache_d = build_lfix_base_cache_cmpq(xf, ctx_cm, base; verify = verify, econ_ws = econ_ws)
cache_c = build_lfix_base_cache_cmpq_C!(wsC, xf, ctx_cm, base; verify = verify)
ed = maximum(abs, cache_d.q0 .- verify.r_current)
ec = maximum(abs, cache_c.q0 .- verify.r_current)
@printf("q0 vs independently recomputed r:  dense %.3e   cplus %.3e\n", ed, ec)
check("q0 fold exact in the DENSE path", ed <= 1e-8, string(ed))
check("q0 fold exact in the C+ path", ec <= 1e-8, string(ec))
# ...and the SAME negative control the dense gate uses, now applied to the C+ cache: folding only
# the level+pair block must leave it badly wrong. Without this, the two checks above would both pass
# if `cmpq_restriction_q0_contribution!` silently lost its CM half.
op = ctx_cm.cmpq_ctx.op
λ_L, λ_P, _ = reshape_cmpq_duals_lambda(base.λstar, op, ctx_cm.cmpq_ctx.ncore1)
contrib_R = zeros(op.W)
cm_pq_forward!(contrib_R, λ_L, λ_P, op, ctx_cm.cmpq_mass_state, ctx_cm.cmpq_ctx.refIndex1)
cache_bare = build_lfix_base_cache_C!(wsC, xf, ctx_cm, base; validate_dense = false)
e_partial = maximum(abs, (cache_bare.q0 .+ contrib_R) .- verify.r_current)
@printf("NEGATIVE CONTROL, C+ with only the level+pair fold: %.3e (vs %.3e with both)\n", e_partial, ec)
check("NEGATIVE CONTROL: the C+ q0 needs the CM fold too", e_partial > 1e-3, string(e_partial))
# The two cache builders must also agree with EACH OTHER on q0, which is a stronger statement than
# both being within 1e-8 of r.
@printf("dense q0 vs cplus q0: max|diff| = %.3e\n", maximum(abs, cache_d.q0 .- cache_c.q0))
check("dense and C+ q0 agree with each other",
      isapprox(cache_d.q0, cache_c.q0; atol = 1e-8, rtol = 1e-8))

@printf("\nwall over %d interleaved reps:  dense min %.3f mean %.3f   cplus min %.3f mean %.3f   speedup(min) %.2fx\n",
        NREP, minimum(td), sum(td) / NREP, minimum(tc), sum(tc) / NREP, minimum(td) / minimum(tc))
check("C+ is not slower than the dense path (best-of on interleaved reps)",
      minimum(tc) <= minimum(td), @sprintf("%.3f vs %.3f", minimum(tc), minimum(td)))

lp("")
lp(ALL_PASS[] ? "ALL CM+PQ C+ GATE CHECKS PASSED" : "SOME C+ GATE CHECKS FAILED")
exit(ALL_PASS[] ? 0 : 1)
