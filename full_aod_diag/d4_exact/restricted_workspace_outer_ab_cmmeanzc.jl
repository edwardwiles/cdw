# ============================================================================
# Restricted-immutable-workspace production port (2026-07-24), task section 8:
# real outer-loop A/B shakedown for CM+mean/ZC, dense (pre-change) vs cached
# (post-change, current production) moment construction.
#
# Same monkey-patch approach as restricted_workspace_outer_ab_originzc.jl,
# applied to wrap_moments_with_cm_meanzc instead: uses the REAL production
# outer-loop driver (run_cm_upper_checkpointed, cm_extension=
# :cm_plus_equal_means_zero_covariance i.e. K_mean=K_pair=1) unmodified for
# both arms; only which wrap_moments_with_cm_meanzc method resolves at the
# call site changes.
#
# Usage: julia --project=. -t 20 full_aod_diag/d4_exact/restricted_workspace_outer_ab_cmmeanzc.jl <dense|cached> <budget_s> <ckpt_dir> <label>
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
    lp(">>> VARIANT=dense: redefining wrap_moments_with_cm_meanzc -> wrap_moments_with_cm_meanzc_dense (this process only)")
    function wrap_moments_with_cm_meanzc(core_moments!::Function, ncore_econ::Int, CM::Matrix{Float64},
                                          Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}};
                                          meanzc_basis::Symbol = :direct, refIndex1::Int = 1)
        return wrap_moments_with_cm_meanzc_dense(core_moments!, ncore_econ, CM, Zraw_all, Zpairraw_all;
                                                  meanzc_basis = meanzc_basis, refIndex1 = refIndex1)
    end
else
    lp(">>> VARIANT=cached: using production wrap_moments_with_cm_meanzc unmodified")
end

const K = 1
const DELTA = 1.0
const W = 80_000
const L = 50
const DRAW_SEED = 20260719
const CONTRASTS = :orthonormal

lp("="^100)
lp("CM+mean/ZC outer A/B: variant=", VARIANT, " K_mean=K_pair=", K, " delta=", DELTA, " budget=", BUDGET,
   "s label=", LABEL, "  ", Dates.now())
lp("="^100)

ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :pseudorandom, draw_seed = DRAW_SEED)
pe = build_pivot_elimination(ctx)
D = ctx.D
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

gp0 = frechet_benchmark_gp(ctx)
z0 = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*_ctx_ddest(ctx)], D, _ctx_ddest(ctx)))
zfree0 = pivot_reduce(z0, pe)
nu0 = [mean(ctx.U)]   # K_mean=1: scalar mean target, matches restricted_workspace_benchmark.jl's own nu0_meanzc convention
eta0 = log.(nu0)
w0 = vcat(gp0, zfree0, eta0)
lp("start point: gp0=", gp0, " n_eta(K_mean)=", length(eta0), " Delta_target(delta)=", DELTA)

res = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
    L = L, contrasts = CONTRASTS, probs = probs,
    maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "cmmeanzc_outer_ab_$(VARIANT)_$(LABEL)", label = "$(VARIANT)_$(LABEL)",
    checkpoint_interval_s = 60.0, cm_gradient_backend = :cplus,
    cm_extension = :cm_plus_equal_means_zero_covariance, meanzc_K_mean = 0, meanzc_K_pair = 0)
    # meanzc_K_mean/K_pair left at 0: the named arm :cm_plus_equal_means_zero_covariance already
    # resolves to (K_mean,K_pair)=(1,1) via meanzc_resolve_K -- passing explicit 1/1 here would be
    # redundant and (per that function's own validation) only tolerated if it MATCHES the named
    # arm's implied value, so leaving it at the 0 default is the documented, error-free spelling.

lp()
lp("A/B RESULT variant=", VARIANT, " knitro_status=", res.knitro_status, " wall=", round(res.wall, digits = 1),
   "s n_eval=", res.n_eval, " n_grad=", res.n_grad, " kappa=", res.kappa)
if res.best === nothing
    lp("NO FEASIBLE INCUMBENT FOUND in this budget.")
else
    lp("best_feasible: gp=", res.best.gp, " Delta=", res.best.Delta, " n_eval=", res.best.n_eval)
    K_mean_actual = 1
    D2_econ = length(w0) - K_mean_actual
    wbest = res.best.w
    xf_best = x_free_from_w(wbest[1:D2_econ], pe)
    nuvec_best = exp.(wbest[D2_econ+1:end])
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean_actual, K_pair = K_mean_actual,
        contrasts = CONTRASTS, meanzc_basis = :direct, probs = probs)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    base_cold, verify_cold = archC_meanzc_verified_state(xf_best, nuvec_best, ctx_cm, cctx)
    lp("COLD-VERIFY: Delta_dual=", verify_cold.Delta_dual, " (checkpoint recorded ", res.best.Delta, ") |diff|=",
       abs(verify_cold.Delta_dual - res.best.Delta), " verified_success=", is_verified_success(verify_cold),
       " class=", classify_inner_result(verify_cold))
    if verify_cold.Delta_dual >= 0.5 && verify_cold.Delta_dual <= 2.05
        seed_path = joinpath(CKPT_DIR, "cmmeanzc_P1_seed_$(VARIANT)_$(LABEL).jls")
        save_benchmark_seed(seed_path, ctx, pe, wbest[1:D2_econ], nuvec_best; family = :cm_meanzc, K_mean = K_mean_actual, K_pair = K_mean_actual,
            mean_target_layout = "scalar_nu_per_k", contrasts = CONTRASTS, cm_L = L,
            Delta_dual = verify_cold.Delta_dual, delta_budget = DELTA)
        lp("P1 candidate saved (Delta_dual in [0.5,2.05]): ", seed_path)
    end
end
lp()
lp("OUTER A/B DONE (", VARIANT, "/", LABEL, ") -- ", Dates.now())
