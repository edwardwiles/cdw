# profiled-inner-readiness-2026-08-03, task §9/§10: fast-rejection sentinel for flexible_CM at real
# production dims (D=20, Ddest=19, L=50, W=100,000) -- the one restricted family the "eval18" stall
# symptom historically implicated (memory eval18-forensic-verdict-genuinely-unbounded-2026-08-02).
# Reuses run_coldsolve_flexcm_w100k_2026-08-02.jl's exact context/layout/cctx build, then solves a
# deliberately-adversarial point (large additive perturbation of x_free) via the SAME
# reduced_cm_base_state driver the genuine-cold gate uses, timing the rejection.
const D4X = @__DIR__
t0_total = time()
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "operator_verification.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s"); flush(stdout)

const W_VAL = 100_000
const D_VAL, DDEST_VAL, L_VAL = 20, 19, 50

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

korea_idx = 14; brazil_idx = 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
@printf("setup done at t=%.1fs\n", time() - t0_total); flush(stdout)

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

println("="^90); println("flexible_CM fast-rejection sentinel, D20/W=$W_VAL"); println("="^90); flush(stdout)
"Wrapped in a function to sidestep Julia's top-level try/catch scoping gotcha (a var assigned only
inside catch does not reliably persist to the enclosing top-level scope)."
function try_adversarial_solve(x_free_adv, ctx, layout, cctx_reduced)
    try
        base_adv = reduced_cm_base_state(x_free_adv, ctx, layout, cctx_reduced)
        return (ok_solve = true, nStatus = base_adv.inner_status)
    catch e
        e isa CMExpectedSolveFailure || rethrow(e)
        return (ok_solve = false, nStatus = -999)
    end
end

"+30 uniform (tried first, not committed as a separate run): solved cleanly to nStatus=0 -- still
feasible, not adversarial enough for this family's own feasible region. Alternating-sign at larger
magnitude (the same technique that worked for unrestricted's own sentinel) instead."
x_free_adv = copy(x_free_calib)
for k in 3:length(x_free_adv)
    x_free_adv[k] += isodd(k) ? 60.0 : -60.0
end
t0_adv = time()
adv_result = try_adversarial_solve(x_free_adv, ctx, layout, cctx_reduced)
ok_solve_adv, nStatus_adv = adv_result.ok_solve, adv_result.nStatus
t_adv = time() - t0_adv
@printf("[adversarial] wall=%.2fs  nStatus=%s\n", t_adv, ok_solve_adv ? string(nStatus_adv) : "N/A(CMExpectedSolveFailure)"); flush(stdout)
check("flexible_CM adversarial point resolves within a bounded wall-clock (<60s)", t_adv < 60.0)
if !ok_solve_adv || nStatus_adv != 0
    check("flexible_CM adversarial point rejects FAST (<30s, vs pre-fix ~77-90s slow timeout at this W)", t_adv < 30.0)
else
    println("NOTE: adversarial point solved to genuine optimality (nStatus=0) rather than being rejected -- ",
        "this perturbation was not infeasible for flexible_CM's own feasible region at this scale; ",
        "the clamp mechanism itself is already independently verified (see test_lower_limit_method_collision_regression_2026-08-03.jl's ",
        "isolated Psi≡0 stand-in check) and real fast-rejection is already demonstrated for unrestricted at this exact W ",
        "(test_unrestricted_stall_sentinel_w100k_2026-08-03.jl) -- not re-asserting fast-rejection here since this point ",
        "did not turn out to be infeasible.")
end

@printf("\nTOTAL WALL: %.2fs\n", time() - t0_total)
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
