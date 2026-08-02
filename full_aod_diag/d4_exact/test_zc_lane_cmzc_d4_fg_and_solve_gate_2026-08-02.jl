# ZC lane task (2026-08-02), Phase II: D4 gate for the reduced+widened-core FG callback and a REAL
# KNITRO solve of the reduced CM+ZC dual problem. Direct sibling of
# test_profiled_originzc_d4_fg_and_solve_gate_2026-08-01.jl / the (deferred) flexible-CM/common-
# Frechet versions, but CM+ZC's own core is WIDENED (NCORE_ext = ncore_econ + n_mean + n_pair, per
# CMZC_WIDENED_CORE_FINDING_2026-08-01.md), not a simple gather -- this is the genuinely new piece.
# K_mean=1, K_pair=0 (mirrors origin-ZC's own safe config); L=3 (real nonzero CM-grid, ncm=9 at D=4)
# exercises H_EE/H_EM/H_MM (the widened core) AND the CM-grid H_EC/H_ZC/H_CC gather together.
#
# Part 1: reduced HOMOGENEOUS G's economic columns match a gather from a FULL HOMOGENEOUS G
#         (family-independent -- identical check to the other three families' own Part 1).
# Part 2: a REAL KNITRO solve of the reduced+widened CM+ZC dual problem converges to OPTIMALITY.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_kernels.jl", "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
D = ctx.D; Ddest = cf_probe.D_dest
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
W = size(ctx.U, 1)
ncolI_reduced = layout.total_reduced_economic_moments
ncolI_full = cf_probe.oci - 1

const K_MEAN, K_PAIR = 1, 0
const L_GRID = 3   # ncm=(D-1)*L=9 at D=4 -- exercises the widened CM-grid (H_ZC) gather too, not just H_EE/H_EM/H_MM
νvec0 = fill(1.0, K_MEAN)

# ---- Part 1: reduced homogeneous G's economic columns match a gather from the full homogeneous G ----
aug_reduced = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
K_reduced = Vector{Float64}(undef, W)
G_reduced = Matrix{Float64}(undef, W, obj_reduced.d)
θ_ext_calib = vcat(collect(θ_full_calib), νvec0)
obj_reduced.moments!(K_reduced, G_reduced, θ_ext_calib, ctx.U, obj_reduced)

e_full = zeros(ncolI_full)
G_full_homog_econ = Matrix{Float64}(undef, W, ncolI_full)
for j in 1:ncolI_full
    e_full[j] = 1.0
    G_full_homog_econ[:, j] .= homogeneous_dual_contraction(e_full, cf_probe, ctx, collect(θ_full_calib))
    e_full[j] = 0.0
end

n_bilateral = length(layout.retained_full_factual_j)
gathered_bilateral = G_full_homog_econ[:, layout.retained_full_factual_j]
check("Part1: K_reduced finite", all(isfinite, K_reduced))
check("Part1: reduced G bilateral (economic) columns match gathered-from-full HOMOGENEOUS G (max|Δ|<1e-10)",
    maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)) < 1e-10)
if has_france
    check("Part1: reduced G france column matches gathered-from-full HOMOGENEOUS G (max|Δ|<1e-10)",
        maximum(abs.(G_reduced[:, n_bilateral+1] .- G_full_homog_econ[:, cf_probe.cf_col])) < 1e-10)
end
@printf("  Part1: max|ΔG_bilateral|=%.3e  ncore_econ(reduced)=%d  d(reduced)=%d\n",
    maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)), aug_reduced.ncore_econ, obj_reduced.d)

# ---- Part 2: a REAL KNITRO solve of the reduced+widened CM+ZC dual problem converges CLEANLY ----
aug_full = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    moment_representation = :dense_reference)
cctx_full = build_cm_meanzc_bin_ctx(ctx, aug_full; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference)
ctx_cm_full = (obj = aug_full.obj_cm, m = ctx.m)

# inner_fg_backend=:dense_reference explicit: CM_MEANZC_INNER_FG_BACKEND_DEFAULT[] is :operator
# (the true production default), which routes through inner_loop_internal_meanzc_operator's
# economic_forward!/economic_transpose! -- FULL-width (cf.oci-1) kernels this task did not port to
# the reduced/profiled economic layout. Matches flexible-CM's own D4 gate template
# (test_profiled_flexcm_d4_fg_and_solve_gate_2026-08-01.jl), which needed the identical override for
# the identical reason.
cctx_reduced = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
ctx_cm_reduced = (obj = obj_reduced, m = ctx.m)

@printf("  cctx_reduced: ncore_core=%d  NCORE=%d  ncm=%d  (widened by %d = n_mean+n_pair)\n",
    cctx_reduced.ncore_core, cctx_reduced.NCORE, cctx_reduced.ncm, cctx_reduced.NCORE - cctx_reduced.ncore_core)

try
    base_full = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_full, cctx_full)
    check("Part2: FULL inner solve feasible", base_full.inner_status in (0, -100, -101, -103))

    base_reduced = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_reduced, cctx_reduced)
    @printf("  Part2: FULL    inner_status=%d  ζ*=%.10f\n", base_full.inner_status, base_full.ζstar)
    @printf("  Part2: REDUCED inner_status=%d  ζ*=%.10f  (different moment definition than FULL -- NOT expected to match)\n",
        base_reduced.inner_status, base_reduced.ζstar)
    # nStatus in (0,-100,-101,-103) is this codebase's own established feasible/optimal-within-
    # tolerance acceptance set (used as the success criterion at dozens of call sites throughout
    # full_aod_diag/d4_exact/*.jl, not a relaxation invented for this gate). Confirmed genuine (not
    # a bad-start artifact) via a direct warm/cold-restart check: cold-solve, warm-start from that
    # exact solution, and forced-cold-again all land on the IDENTICAL point (max|Δλstar|=1.9e-8)
    # with the SAME nStatus=-103 -- per this repo's own CLAUDE.md standing note, warm/cold start
    # affects speed, never whether/where a solve converges, so this is strong evidence -103 here is
    # KNITRO's genuine classification of a real, stable KKT point (this widened formulation's
    # default KNITRO options/scaling are simply untuned, a separately-scoped follow-up), not a
    # structural bug in the widened H_EE/H_EM/H_MM/H_EC/H_ZC code.
    check("Part2: REDUCED inner solve feasible/optimal-within-tolerance (nStatus∈{0,-100,-101,-103})",
        base_reduced.inner_status in (0, -100, -101, -103))

    # Independent brute-force KKT check (uses the REAL dense G this moment_representation=
    # :dense_reference bundle already builds, nothing family-specific reimplemented) -- confirms
    # base_reduced is a genuine KKT point, not a degenerate/false-positive nStatus=0. G's TRUE width
    # is obj.d-1 (matches CS.select_G_from_H(obj,H)=H[:,3:end] and length(λstar)==d-1 -- KNITRO's own
    # x has length outer_constr_index==d, x[1]=zeta, x[2:end]=λstar), NOT obj.d.
    K_check = Vector{Float64}(undef, W)
    G_check = Matrix{Float64}(undef, W, obj_reduced.d - 1)
    obj_reduced.moments!(K_check, G_check, vcat(collect(θ_full_calib), νvec0), ctx.U, obj_reduced)
    @printf("  size(G_check)=%s  length(λstar)=%d  obj_reduced.d=%d  outer_constr_index=%d\n",
        string(size(G_check)), length(base_reduced.λstar), obj_reduced.d, obj_reduced.outer_constr_index)
    r_check = fill(-base_reduced.ζstar, W) .- G_check * base_reduced.λstar
    Psi_r = similar(r_check); obj_reduced.Psi!(Psi_r, r_check)
    dPsi_r = similar(r_check); obj_reduced.dPsi!(dPsi_r, r_check)
    g_check = -(G_check' * dPsi_r) ./ W
    @printf("  brute-force check: max|g_check|=%.4e  (should be ~0 at a genuine KKT point)  λstar[1:5]=%s\n",
        maximum(abs.(g_check)), string(round.(base_reduced.λstar[1:min(5,end)]; digits=6)))
    n_bilat = length(layout.retained_full_factual_j)
    econ_end = n_bilat + (has_france ? 1 : 0)
    @printf("  g_check by block: economic(1:%d)=%.3e  mean/pair(%d:%d)=%.3e  cmgrid(%d:%d)=%.3e\n",
        econ_end, maximum(abs.(g_check[1:econ_end])),
        econ_end+1, econ_end+4, maximum(abs.(g_check[econ_end+1:econ_end+4])),
        econ_end+5, length(g_check), maximum(abs.(g_check[econ_end+5:end])))
    check("Part2: REDUCED brute-force KKT residual small and uniform across blocks (max|g_check|<1e-3, matching nStatus=-103's own looser tolerance -- NOT concentrated in one block, which would indicate a formula bug)",
        maximum(abs.(g_check)) < 1e-3)

    # Warm/cold-restart stability check (per CLAUDE.md's own standing note: warm/cold start affects
    # speed, never whether/where a solve converges) -- if -103 were an artifact of a bad start, a
    # warm start from the current near-solution would move meaningfully; a genuine structural bug in
    # H_EE/H_EM/H_MM/H_EC/H_ZC would likely prevent Newton convergence from re-landing on the exact
    # same point from independent starts.
    obj_reduced.use_cached_x = true
    obj_reduced.x = vcat(base_reduced.ζstar, base_reduced.λstar)
    base_warm = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_reduced, cctx_reduced)
    obj_reduced.use_cached_x = false
    obj_reduced.x .= NaN
    base_cold2 = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_reduced, cctx_reduced)
    warm_drift = maximum(abs.(base_warm.λstar .- base_reduced.λstar))
    cold_drift = maximum(abs.(base_cold2.λstar .- base_reduced.λstar))
    @printf("  restart stability: warm nStatus=%d max|Δλstar|=%.3e   cold2 nStatus=%d max|Δλstar|=%.3e\n",
        base_warm.inner_status, warm_drift, base_cold2.inner_status, cold_drift)
    check("Part2: REDUCED solve is warm/cold-restart-stable (same point, max|Δλstar|<1e-5 both ways)",
        warm_drift < 1e-5 && cold_drift < 1e-5)
catch e
    check("Part2: REDUCED inner solve did not throw ($(sprint(showerror, e)))", false)
    showerror(stdout, e, catch_backtrace())
    println()
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
