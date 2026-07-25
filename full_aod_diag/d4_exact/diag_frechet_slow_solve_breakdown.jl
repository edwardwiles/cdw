# ============================================================================
# Diagnostic (user-requested, live 2026-07-24): the outer shakedown showed a
# single inner solve at a point near (but not exactly at) the calibration
# point taking ~700-1000s wall time. This script isolates and measures WHERE
# that time goes, using the codebase's OWN existing profiling infrastructure
# (`@prof`/`prof_summary()`, instrumentation.jl) rather than ad hoc timers --
# `inner_loop_KNITRO_archgeneric` already wraps the entire `KN_solve` call in
# `@prof "inner_knitro_dual_solve_arch"`, and the moment build / structured
# Hessian callback are separately profiled, so comparing their times directly
# answers "is it moment/Hessian construction, or KNITRO's own internal
# per-iteration linear algebra?" without guessing.
#
# Target point: (gp_target, zfree*) -- the SAME point already measured at
# ~558-790s across three prior runs (the outer shakedown's own warm-up step,
# and the standalone gradient-diagnostic's base solve) -- fully reproducible
# from ctx.θ0_up, unlike the outer loop's own internally-chosen eval-2 point
# (whose exact zfree step direction was never logged and is not recoverable
# now that the process was killed).
#
# Also captures the RAW KNITRO iteration table (outlev=4, normally outlev=0
# in production) to a dedicated log file.
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
using Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
t0 = time()
const W = 80_000; const L = 50; const DELTA = 1.0

ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
lp(@sprintf("[%.1fs] ctx built  D=%d D_dest=%d", time()-t0, ctx.D, ctx.D_dest))
pe = build_pivot_elimination(ctx)

cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
t_fpcx = @elapsed fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
lp(@sprintf("[%.1fs] fpcx built in %.1fs  ncore=%d ncm=%d  (n_inner_vars = ncore+ncm+1 = %d)",
    time()-t0, t_fpcx, fpcx.aug.ncore, fpcx.aug.ncm, fpcx.aug.ncore+fpcx.aug.ncm+1))

x_free_calib = ctx.θ0_up[ctx.free_idx]
z_star = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
zfree_star = pivot_reduce(z_star, pe)
σ = ctx.σ
κ_star = 1 - x_free_calib[1]^(σ / (σ - 1))
gp_target = (1 - (κ_star + 1e-4))^((σ - 1) / σ)
x_free0 = vcat(gp_target, vec(exp.(pivot_expand(zfree_star, pe))))
lp(@sprintf("[%.1fs] target point: gp=%.10f (gp*=%.10f)", time()-t0, gp_target, x_free_calib[1]))

# ---- point over full theta for solve ----
θ_full0 = CS.reconstruct_full(x_free0, fpcx.ctx_cm.m)

# Point KNITRO at the verbose-outlev copy so we capture the raw per-iteration table.
verbose_opt = joinpath(@__DIR__, "ek_inner_diag_verbose.opt")
fpcx.ctx_cm.obj.inner_loop_opt = verbose_opt
lp(@sprintf("[%.1fs] inner_loop_opt set to verbose copy (outlev=4): %s", time()-t0, verbose_opt))

prof_reset!()
CS.INNER_ITERS_TOTAL[] = 0
lp(@sprintf("[%.1fs] === STARTING TIMED SOLVE (this is the ~700-1000s step being diagnosed) ===", time()-t0))
t_solve = @elapsed begin
    K, x_sol, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(fpcx.ctx_cm.obj, θ_full0;
        hess_cb_builder = _o -> archC_frechet_cdf_power_hess_cb_builder(fpcx.fctx))
end
lp(@sprintf("[%.1fs] === SOLVE DONE: wall=%.1fs  nStatus=%d  n_fg=%d  n_hess=%d  knitro_iters=%d ===",
    time()-t0, t_solve, nStatus, n_fg, n_hess, CS.INNER_ITERS_TOTAL[]))
lp("category=$(decode_knitro_status(nStatus).category)")

lp("")
lp("="^100)
lp("PROFILING BREAKDOWN (this codebase's own @prof instrumentation, instrumentation.jl)")
lp("="^100)
rows = prof_summary()
for r in rows
    @printf("  %-55s n=%-4d median=%-8.4fs  min=%-8.4fs  max=%-8.4fs\n",
        r.label, r.n, r.median_s, r.min_s, r.max_s)
end
total_prof_labeled = 0.0
for label in sort(collect(keys(PROF_TIMES)))
    tot = sum(PROF_TIMES[label])
    global total_prof_labeled
    println("  LABEL TOTAL: ", label, " = ", round(tot, digits = 3), "s  (n=", length(PROF_TIMES[label]), ")")
    if label != "inner_knitro_dual_solve_arch"
        total_prof_labeled += tot
    end
end
knitro_total = haskey(PROF_TIMES, "inner_knitro_dual_solve_arch") ? sum(PROF_TIMES["inner_knitro_dual_solve_arch"]) : NaN
lp("")
lp(@sprintf("Total wall time for the solve call:              %.3fs", t_solve))
lp(@sprintf("Time inside KN_solve (KNITRO's own C library):   %.3fs  (%.1f%% of total)", knitro_total, 100*knitro_total/t_solve))
lp(@sprintf("Time in ALL OTHER @prof-labeled Julia callbacks:  %.3fs  (moment build + Hessian callback + FG callback overhead, %.1f%% of total)",
    total_prof_labeled, 100*total_prof_labeled/t_solve))
lp(@sprintf("Unaccounted (setup/KN_new/KN_free/option load):   %.3fs", t_solve - knitro_total))
lp("")
lp(@sprintf("n_fg (function+gradient callback count) = %d", n_fg))
lp(@sprintf("n_hess (Hessian callback count) = %d", n_hess))
lp(@sprintf("KNITRO-reported total iterations (CS.INNER_ITERS_TOTAL[]) = %d", CS.INNER_ITERS_TOTAL[]))
lp(@sprintf("=> average wall time per KNITRO iteration: %.3fs", knitro_total / max(CS.INNER_ITERS_TOTAL[], 1)))
lp("")
lp("Raw KNITRO per-iteration table (outlev=4) is interleaved above in this same log (search for 'Iter' / the objective/FeasError/OptError table).")
