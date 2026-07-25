# ============================================================================
# Restricted-immutable-workspace production port (2026-07-24), task section 7:
# real outer-loop A/B shakedown for origin-ZC, dense (pre-change) vs cached
# (post-change, current production) moment construction.
#
# Uses the REAL production outer-loop driver (run_originzc_upper_checkpointed)
# unmodified -- identical algorithm, option file, screens, and cache policy
# for both arms. The ONLY difference between the "dense" and "cached" runs is
# which `wrap_moments_with_originzc` method resolves at the call site inside
# build_originzc_augmented_obj: for VARIANT="dense" this script redefines
# wrap_moments_with_originzc (exact same type signature as the production
# method, so this replaces rather than shadows it) to delegate to
# wrap_moments_with_originzc_dense, the byte-for-byte-preserved pre-2026-07-24
# allocating reference path. This is a same-process, this-script-only
# redefinition -- no production file is touched.
#
# Usage: julia --project=. -t 20 full_aod_diag/d4_exact/restricted_workspace_outer_ab_originzc.jl <dense|cached> <budget_s> <ckpt_dir> <label>
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_checkpoint_fingerprint.jl"))
include(joinpath(@__DIR__, "direction_bounds.jl"))
using Printf, LinearAlgebra, Statistics, Dates

lp(xs...) = (println(xs...); flush(stdout))

const VARIANT = ARGS[1]
VARIANT in ("dense", "cached") || error("VARIANT must be dense|cached, got $VARIANT")
const BUDGET = parse(Float64, ARGS[2])
const CKPT_DIR = ARGS[3]
const LABEL = ARGS[4]
mkpath(CKPT_DIR)

if VARIANT == "dense"
    lp(">>> VARIANT=dense: redefining wrap_moments_with_originzc -> wrap_moments_with_originzc_dense (this process only)")
    function wrap_moments_with_originzc(core_moments!::Function, ncore_econ::Int,
                                         Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}},
                                         layout::MeanZCTargetLayout)
        return wrap_moments_with_originzc_dense(core_moments!, ncore_econ, Zraw_all, Zpairraw_all, layout)
    end
else
    lp(">>> VARIANT=cached: using production wrap_moments_with_originzc unmodified")
end

const K = 1
const DELTA = 1.0
const W = 80_000
const DRAW_SEED = 20260719

lp("="^100)
lp("Origin-ZC outer A/B: variant=", VARIANT, " K_mean=K_pair=", K, " delta=", DELTA, " budget=", BUDGET,
   "s label=", LABEL, "  ", Dates.now())
lp("="^100)

ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :pseudorandom, draw_seed = DRAW_SEED)
pe = build_pivot_elimination(ctx)
D = ctx.D

gp0 = frechet_benchmark_gp(ctx)
z0 = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*_ctx_ddest(ctx)], D, _ctx_ddest(ctx)))
zfree0 = pivot_reduce(z0, pe)

layout = OriginByPowerLayout(D, K, K)
nu0 = Vector{Float64}(undef, n_eta(layout))
for k in 1:K
    Uk = ctx.U .^ k
    for o in 1:D
        nu0[target_index(layout, o, k)] = mean(@view Uk[:, o])
    end
end
eta0 = log.(nu0)
w0 = vcat(gp0, zfree0, eta0)
lp("start point: gp0=", gp0, " n_eta=", length(eta0), " Delta_target(delta)=", DELTA)

res = run_originzc_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
    maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "originzc_outer_ab_$(VARIANT)_$(LABEL)", label = "$(VARIANT)_$(LABEL)",
    checkpoint_interval_s = 60.0, cm_gradient_backend = :cplus,
    distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = K, K_pair = K,
    power_target_layout = :origin_by_power)

lp()
lp("A/B RESULT variant=", VARIANT, " knitro_status=", res.knitro_status, " wall=", round(res.wall, digits = 1),
   "s n_eval=", res.n_eval, " n_grad=", res.n_grad, " kappa=", res.kappa)
if res.best === nothing
    lp("NO FEASIBLE INCUMBENT FOUND in this budget.")
else
    lp("best_feasible: gp=", res.best.gp, " Delta=", res.best.Delta, " n_eval=", res.best.n_eval)
    D2_econ = length(w0) - n_eta(layout)
    wbest = res.best.w
    xf_best = x_free_from_w(wbest[1:D2_econ], pe)
    νfull_best = exp.(wbest[D2_econ+1:end])
    pcx = build_originzc_production_context(ctx, CS, layout)
    _, base_cold, verify_cold = cm_originzc_production_value_verified(xf_best, νfull_best, pcx)
    lp("COLD-VERIFY: Delta_dual=", verify_cold.Delta_dual, " (checkpoint recorded ", res.best.Delta, ") |diff|=",
       abs(verify_cold.Delta_dual - res.best.Delta), " verified_success=", is_verified_success(verify_cold),
       " class=", classify_inner_result(verify_cold))
    if verify_cold.Delta_dual >= 0.5 && verify_cold.Delta_dual <= 2.05
        seed_path = joinpath(CKPT_DIR, "originzc_P1_seed_$(VARIANT)_$(LABEL).jls")
        save_benchmark_seed(seed_path, ctx, pe, wbest[1:D2_econ], νfull_best; family = :origin_zc, K_mean = K, K_pair = K,
            mean_target_layout = "SharedByPowerLayout", contrasts = :orthonormal, cm_L = 0,
            Delta_dual = verify_cold.Delta_dual, delta_budget = DELTA)
        lp("P1 candidate saved (Delta_dual in [0.5,2.05]): ", seed_path)
    end
end
lp()
lp("OUTER A/B DONE (", VARIANT, "/", LABEL, ") -- ", Dates.now())
