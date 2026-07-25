# ============================================================================
# Real D=20/W=80,000/L=50/:cdf_power outer shakedown launcher (task brief
# §12). Starts from the calibration point (gp*, zfree*), targets
# kappa*+Delta_kappa (matching the pre-omit-ROW archive's own convention --
# starting exactly AT gp* gives the degenerate Delta*=0 corner, not a useful
# outer-search starting point), runs `run_frechet_upper` for the given
# wall-clock budget.
#
# Usage: julia launch_frechet_shakedown.jl <maxtime_real_seconds> [W] [L]
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "run_frechet_upper.jl"))
using Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

const MAXTIME = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 120.0
const W = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 80_000
const L = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
const DELTA = 1.0

t0 = time()
lp("MAXTIME=$MAXTIME W=$W L=$L DELTA=$DELTA")
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
lp("[$(round(time()-t0,digits=1))s] ctx built  D=$(ctx.D) D_dest=$(ctx.D_dest)")
pe = build_pivot_elimination(ctx)

cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
lp("[$(round(time()-t0,digits=1))s] fpcx built  ncore=$(fpcx.aug.ncore) ncm=$(fpcx.aug.ncm)")

x_free_calib = ctx.θ0_up[ctx.free_idx]
z_star = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
zfree_star = pivot_reduce(z_star, pe)
σ = ctx.σ
κ_star = 1 - x_free_calib[1]^(σ / (σ - 1))
Δκ = 1e-4
κ_target = κ_star + Δκ
gp_target = (1 - κ_target)^((σ - 1) / σ)
w0 = vcat(gp_target, zfree_star)
lp("[$(round(time()-t0,digits=1))s] kappa*=$κ_star  target kappa=$κ_target  gp*=$(x_free_calib[1])  gp_target=$gp_target")

lp("[$(round(time()-t0,digits=1))s] warm-up: verifying starting point (gp_target, zfree*) is cold-solvable...")
base0, verify0 = cm_frechet_verified_state(vcat(gp_target, vec(exp.(pivot_expand(zfree_star, pe)))), fpcx)
lp("[$(round(time()-t0,digits=1))s] warm-up outcome=$(frechet_solve_outcome(base0.inner_status))  verified=$(is_verified_success(verify0))  Delta=$(verify0.Delta_dual)")

lp("[$(round(time()-t0,digits=1))s] launching run_frechet_upper, maxtime_real=$MAXTIME ...")
result = run_frechet_upper(fpcx, ctx, pe, w0; delta = DELTA, maxtime_real = MAXTIME,
    opt_file = "csw_outer_wallclock_sr1.opt", verbose = true)

lp("="^100)
lp("SHAKEDOWN RESULT")
lp("="^100)
lp("knitro_status=$(result.knitro_status)  wall=$(round(result.wall,digits=1))s")
lp("n_eval=$(result.n_eval)  n_grad=$(result.n_grad)  n_new_point_solve=$(result.n_new_point_solve)")
lp("n_base_reused_at_gradient=$(result.n_base_reused_at_gradient)")
lp("n_time_limit_no_certificate=$(result.n_time_limit_no_certificate)  n_infeasible_certificate=$(result.n_infeasible_certificate)")
if result.best !== nothing
    lp("BEST: gp=$(result.best.gp)  Delta=$(result.best.Delta)  kappa=$(result.kappa)  n_eval=$(result.best.n_eval)  t=$(round(result.best.t,digits=1))s")
    lp("[$(round(time()-t0,digits=1))s] cold-verifying best incumbent...")
    xf_best = vcat(result.best.gp, vec(exp.(pivot_expand(result.best.w[2:end], pe))))
    base_cv, verify_cv = cm_frechet_verified_state(xf_best, fpcx)
    lp("cold-verify: outcome=$(frechet_solve_outcome(base_cv.inner_status))  verified=$(is_verified_success(verify_cv))  Delta=$(verify_cv.Delta_dual)")
else
    lp("BEST: none found")
end
lp("total wall (incl. setup): $(round(time()-t0,digits=1))s")
