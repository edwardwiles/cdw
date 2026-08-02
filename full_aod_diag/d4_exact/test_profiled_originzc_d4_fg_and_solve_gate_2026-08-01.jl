# Restricted-inner-endtoend task (2026-08-01), §8/§11/§12: D4 gate for the reduced FG callback and a
# REAL KNITRO solve of the reduced dual problem, ZC-only (origin-ZC) -- the direct sibling of
# test_profiled_flexcm_d4_fg_and_solve_gate_2026-08-01.jl / test_profiled_cmfrechet_d4_fg_and_solve_
# gate_2026-08-01.jl, using the SAME two-part structure. Uses K_mean=1, K_pair=0 -- the ONLY origin-ZC
# config confirmed safe against the separately-tracked "archOZ_base_state direct-call crash at K>1"
# issue (this session's own memory note) -- that issue is out of scope here (task's own instruction:
# "do not merge the separate experimental ZC-Hessian backend branch").
#
# Part 1: the reduced HOMOGENEOUS G's economic columns match a gather from a FULL HOMOGENEOUS G
# (family-independent -- identical to the other two families' own Part 1).
# Part 2: a REAL KNITRO inner solve of the reduced origin-ZC dual problem converges to OPTIMALITY
# (nStatus==0), exercising the NEW H_EE/H_EZ profiled branches in archA_partitioned_hess_cb_builder
# for the first time under a real solve.
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

layout_o = OriginByPowerLayout(ctx.D, 1, 0)   # K_mean=1, K_pair=0 -- the safe config
νfull0 = fill(1.0, ctx.D)   # factorial(1)=1 per origin, K_mean=1

# ---- Part 1: reduced homogeneous G matches a gather from the full homogeneous G (same family) ----
aug_reduced = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
n_eta_total = n_eta(layout_o)
K_reduced = Vector{Float64}(undef, W)
G_reduced = Matrix{Float64}(undef, W, obj_reduced.d)
θ_ext_calib = vcat(collect(θ_full_calib), νfull0)
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
check("Part1: reduced G bilateral columns match gathered-from-full HOMOGENEOUS G (max|Δ|<1e-10)",
    maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)) < 1e-10)
if has_france
    check("Part1: reduced G france column matches gathered-from-full HOMOGENEOUS G (max|Δ|<1e-10)",
        maximum(abs.(G_reduced[:, n_bilateral+1] .- G_full_homog_econ[:, cf_probe.cf_col])) < 1e-10)
end
@printf("  Part1: max|ΔG_bilateral|=%.3e\n", maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)))

# ---- Part 2: a REAL KNITRO solve of the reduced origin-ZC dual problem converges CLEANLY ----
aug_full = build_originzc_augmented_obj(ctx, CS, layout_o; moment_representation = :dense_reference)
octx_full = build_originzc_core_hess_ctx(aug_full, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                          zc_cross_hessian_backend = :winner_bin)
ctx_cm_full = (obj = aug_full.obj_cm, m = ctx.m, octx = octx_full)

octx_reduced = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                             zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
ctx_cm_reduced = (obj = obj_reduced, m = ctx.m, octx = octx_reduced)

try
    base_full = archOZ_base_state(x_free_calib, νfull0, ctx_cm_full)
    check("Part2: FULL inner solve feasible", base_full.inner_status in (0, -100, -101, -103))

    base_reduced = archOZ_base_state(x_free_calib, νfull0, ctx_cm_reduced)
    @printf("  Part2: FULL    inner_status=%d  ζ*=%.10f\n", base_full.inner_status, base_full.ζstar)
    @printf("  Part2: REDUCED inner_status=%d  ζ*=%.10f  (different moment definition than FULL -- NOT expected to match)\n",
        base_reduced.inner_status, base_reduced.ζstar)
    check("Part2: REDUCED inner solve converges to OPTIMALITY (nStatus==0, not merely feasible)",
        base_reduced.inner_status == 0)
catch e
    check("Part2: REDUCED inner solve did not throw ($(sprint(showerror, e)))", false)
    showerror(stdout, e, catch_backtrace())
    println()
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
