# ============================================================================
# Fixed-Fréchet-marginals D=20/W=80,000/L=50 outer shakedown (task brief §10.3).
# marginal_mode=:frechet_reference, cm_extension=:cm_only, backend=:cplus,
# delta=1.0, start=A*, wall budget=10 minutes, direct joint constrained
# search (no fixed-gp profile). Writes a CMCheckpointV5 checkpoint, cold-
# verifies the best feasible incumbent, reports whether the divergence
# constraint is binding or slack.
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
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_frechet_outer_driver.jl"))
include(joinpath(@__DIR__, "knitro_version_check.jl"))
using Printf, LinearAlgebra, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
t0 = time()
elapsed() = round(time() - t0, digits = 1)

lp("=== test_frechet_d20_shakedown === ", Dates.now())

const L = 50
const DELTA = 1.0
const MAXTIME = 600.0   # 10 minutes, task brief section 10.3

ctx = d20_real_setup_design(W = 80000, δ = DELTA, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
D = ctx.D
lp(@sprintf("[%.1fs] ctx built. D=%d W=%d", elapsed(), D, size(ctx.U, 1)))

cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
                       marginal_mode = :frechet_reference)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
lp(@sprintf("[%.1fs] fixed-Frechet production context built. ncm=%d d_total=%d", elapsed(), fpcx.aug.ncm, fpcx.ctx_cm.obj.d))

x_free_calib = ctx.θ0_up[ctx.free_idx]
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0 = vcat(x_free_calib[1], pivot_reduce(z0, pe))
lp(@sprintf("[%.1fs] starting from A* (w0[1]=%.10f)", elapsed(), w0[1]))

lp("="^100); lp("Direct joint constrained search: minimize w[1] s.t. Delta_dual(w) <= ", DELTA, ", maxtime_real=", MAXTIME); lp("="^100)
result = run_frechet_upper_cplus(fpcx, ctx, pe, w0; delta = DELTA, maxtime_real = MAXTIME,
                                  opt_file = "csw_outer_wallclock_sr1.opt", z_halfwidth = 30.0, verbose = true)

lp("="^100); lp("SHAKEDOWN RESULT"); lp("="^100)
lp("knitro_status=", result.knitro_status, "  wall=", round(result.wall, digits = 1), "s  n_eval=", result.n_eval, "  n_grad=", result.n_grad)
lp("kappa=", result.kappa)

if result.best === nothing
    lp("NO FEASIBLE INCUMBENT FOUND within the wall budget -- reporting honestly, not fabricating a result.")
else
    b = result.best
    lp(@sprintf("best incumbent: gp=%.10f  Delta=%.10f  n_eval=%d  t=%.1fs", b.gp, b.Delta, b.n_eval, b.t))

    lp("="^100); lp("COLD VERIFICATION of the best feasible incumbent"); lp("="^100)
    xf_best = x_free_from_w(b.w, pe)
    base_cv, verify_cv = archC_frechet_verified_state(xf_best, fpcx.ctx_cm, fpcx.fctx)
    lp(@sprintf("cold-verified Delta_dual=%.10f  (search reported %.10f, |diff|=%.3e)", verify_cv.Delta_dual, b.Delta, abs(verify_cv.Delta_dual - b.Delta)))
    lp(@sprintf("inner_status=%s  primal_dual_gap=%.3e  weight_norm_resid=%.3e  max_abs_moment_kkt_resid=%.3e",
                string(verify_cv.inner_status), verify_cv.primal_dual_gap, verify_cv.weight_norm_resid, verify_cv.max_abs_moment_kkt_resid))
    verified_ok = is_verified_success(verify_cv)
    lp("is_verified_success=", verified_ok)

    slack = DELTA - verify_cv.Delta_dual
    binding = slack < 1e-3
    lp(@sprintf("divergence constraint: Delta=%.10f vs delta=%.10f -> slack=%.3e -> %s",
                verify_cv.Delta_dual, DELTA, slack, binding ? "BINDING" : "SLACK"))

    # Task brief section 11: benchmark target discrepancy diagnostics at the incumbent
    m = base_cv.m_star; p = m ./ sum(m)
    resid = Matrix{Float64}(undef, D, L)
    for l in 1:L, o in 1:D
        resid[o, l] = sum(p[s] * (ctx.U[s, o] <= fpcx.targets.thresholds[l]) for s in 1:size(ctx.U,1)) - fpcx.targets.targets[l]
    end
    lp(@sprintf("max |weighted-CDF residual vs F* target| at incumbent = %.3e", maximum(abs.(resid))))

    # Write checkpoint (task brief section 9/10.3: "write a checkpoint")
    ckpt_ctx = cm_frechet_checkpoint_context(cfg, fpcx.targets)
    ckpt = CMCheckpointV5(CM_FRECHET_CHECKPOINT_SCHEMA, "frechet_shakedown_" * string(Dates.now()), "frechet_d20_shakedown",
        :frechet_shakedown, true, DELTA, size(ctx.U,1), ctx.draw_seed, ctx.draw_design,
        ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
        L, fpcx.targets.probs, :orthonormal, :equal, :cumulative, :structured, :cplus,
        :cm_only, 0, 0, :direct, FRECHET_FEATURE_LAYOUT_VERSION,
        b.gp, b.w[2:end], Float64[], log.(reshape(xf_best[2:end], D, D)), Float64[], Dict{Int,Float64}(),
        (w = b.w, Delta = verify_cv.Delta_dual, feasible = verify_cv.Delta_dual <= DELTA + 1e-6, gp = b.gp),
        result.n_eval, result.n_grad, result.wall, MAXTIME - result.wall, :shakedown_complete,
        KNITRO_PRODUCTION_VERSION,
        ckpt_ctx.marginal_mode, ckpt_ctx.theta_star, ckpt_ctx.scale, ckpt_ctx.sigma, ckpt_ctx.probs,
        ckpt_ctx.thresholds_checksum, ckpt_ctx.target_checksum, ckpt_ctx.feature_layout_version)

    ckpt_dir = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "fixed_frechet_d20_shakedown")
    mkpath(ckpt_dir)
    ckpt_path = joinpath(ckpt_dir, "shakedown_checkpoint.jls")
    save_cm_frechet_checkpoint(ckpt_path, ckpt)
    lp("checkpoint written to ", ckpt_path)

    # round-trip sanity
    reloaded = load_cm_frechet_checkpoint(ckpt_path)
    lp("checkpoint round-trip OK: ", reloaded.marginal_mode === :frechet_reference && reloaded.best_feasible.Delta == verify_cv.Delta_dual)
end

lp()
lp(@sprintf("Peak RSS: %.2f GB", Sys.maxrss() / 1e9))
lp("Total wall: ", elapsed(), "s")
