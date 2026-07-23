# Short outer shakedown for the origin-specific-ZC restriction (task brief
# Section 11.3): direct joint constrained search over gp, zfree, and all
# eta_{o,k} (run_originzc_upper_checkpointed's own KNITRO outer loop -- NOT a
# fixed-gp profile), from the calibrated benchmark A*, for a bounded wall
# budget. Performance/integration shakedown, not a final bound. Cold-verifies
# the best feasible incumbent at the end (independent fresh inner re-solve).
#
# Usage: julia --project=. -t 20 d20_originzc_shakedown.jl <K> <delta> <budget_s> <ckpt_dir> <label>
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
include(joinpath(@__DIR__, "direction_bounds.jl"))
using Printf, LinearAlgebra, Statistics, Dates

lp(xs...) = (println(xs...); flush(stdout))

const K = parse(Int, ARGS[1])
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const CKPT_DIR = ARGS[4]
const LABEL = ARGS[5]
mkpath(CKPT_DIR)

lp("="^100)
lp("Origin-ZC shakedown: K_mean=K_pair=", K, " delta=", DELTA, " budget=", BUDGET, "s label=", LABEL, "  ", Dates.now())
lp("="^100)

ctx = d20_real_setup_design(W = 80000, δ = DELTA, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
D = ctx.D

gp0 = frechet_benchmark_gp(ctx)
z0 = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
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

res = run_originzc_upper_checkpointed(w0; W = 80000, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "originzc_shakedown_$(LABEL)", label = LABEL,
    checkpoint_interval_s = 60.0, cm_gradient_backend = :cplus,
    distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = K, K_pair = K,
    power_target_layout = :origin_by_power)

lp()
lp("shakedown finished: knitro_status=", res.knitro_status, " wall=", round(res.wall, digits = 1),
   "s n_eval=", res.n_eval, " n_grad=", res.n_grad, " kappa=", res.kappa)
if res.best === nothing
    lp("NO FEASIBLE INCUMBENT FOUND in this budget.")
else
    lp("best_feasible: gp=", res.best.gp, " Delta=", res.best.Delta, " n_eval=", res.best.n_eval)
    # cold-verify: fresh inner re-solve at the incumbent's own w, independent of the checkpointed base/verify
    D2_econ = length(w0) - n_eta(layout)
    wbest = res.best.w
    xf_best = x_free_from_w(wbest[1:D2_econ], pe)
    νfull_best = exp.(wbest[D2_econ+1:end])
    pcx = build_originzc_production_context(ctx, CS, layout)
    _, base_cold, verify_cold = cm_originzc_production_value_verified(xf_best, νfull_best, pcx)
    lp("COLD-VERIFY: Delta_dual=", verify_cold.Delta_dual, " (checkpoint recorded ", res.best.Delta, ") |diff|=",
       abs(verify_cold.Delta_dual - res.best.Delta), " verified_success=", is_verified_success(verify_cold),
       " class=", classify_inner_result(verify_cold))
end
lp()
lp("SHAKEDOWN DONE (", LABEL, ") -- ", Dates.now())
