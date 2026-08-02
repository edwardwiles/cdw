# Performance closeout task (2026-08-02), Section 13, FULL arm, flexible_CM: real outer search via
# the actual production checkpointed driver (run_cm_upper_checkpointed, cm_checkpoint.jl) --
# unchanged, called as designed, with EVERY kwarg that matters left at its own documented default
# (marginal_restriction=:common_flexible, cm_extension=:cm_only -- plain flexible CM, no nu/eta axis
# at all, matching the REDUCED arm exactly with no pinning needed).
#
# ctx-construction kwargs match run_outer_flexcm_reduced_constrained_2026-08-02.jl's own explicit
# d20_real_setup_design call exactly (same determinism argument as the origin-ZC scripts).
#
# Usage: julia run_outer_flexcm_full_2026-08-02.jl <W> <maxtime_real_s>
const D4X = @__DIR__
t0_total = time()
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl",
          "cm_screen_bridge.jl","gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl",
          "cm_aspace_coordinate.jl","cm_checkpoint.jl",
          "cm_originzc_target_layout.jl","cm_originzc_moments.jl","cm_originzc_production.jl","cm_originzc_cplus.jl",
          "cm_originzc_config.jl","cm_originzc_checkpoint.jl","direction_bounds.jl"]
    include(joinpath(D4X, f))
end
using Printf, Dates
lp(xs...) = (println(xs...); flush(stdout))
lp("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s")

const W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const MAXT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 600.0
const DELTA = 1.0
const L_VAL = 50
probs = cm_equal_grid_probs(L_VAL)

t_ctx = @elapsed ctx1 = d20_real_setup_design(W = W_VAL, δ = DELTA, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
D = ctx1.D; Ddest = ctx1.D_dest
@printf("MATCHED-CTX DIAGNOSTIC: context build=%.2fs  D=%d Ddest=%d  kappa_upper=%.10f  gp0=%.10f  gp_bounds=(%.6f,%.6f)\n",
    t_ctx, D, Ddest, ctx1.bounds.κ_max, frechet_benchmark_gp(ctx1), ctx1.bounds.γp_lo, ctx1.bounds.γp_hi)
flush(stdout)

pe1 = build_pivot_elimination(ctx1)
w0 = cm_w0_from_calibration(ctx1, pe1, :legacy_z)   # NO eta appended -- plain flexible CM has no nu/eta axis at all
@printf("w0: gp0=%.10f  length(zfree)=%d\n", w0[1], length(w0) - 1)
flush(stdout)

ckpt_dir = mktempdir()
outdir = joinpath(D4X, "..", "..", "docs")
mkpath(outdir)

lp("="^100); lp("FULL constrained outer search: flexible_CM  W=$(W_VAL)  maxtime_real=$(MAXT)s  delta=$(DELTA)")
lp("="^100)
result = run_cm_upper_checkpointed(w0; W = W_VAL, delta = DELTA, draw_design = :sobol_randomized, draw_seed = 20260719,
    L = L_VAL, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_flexible, cm_extension = :cm_only,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    A_coordinate_mode = :legacy_z, outer_direct_hessopt = :sr1,
    maxtime_real = MAXT, ckpt_dir = ckpt_dir, run_id = "flexcm_full_w$(W_VAL)",
    label = "flexcm_full_w$(W_VAL)", checkpoint_interval_s = 60.0,
    cm_gradient_backend = :cplus, verbose = true)

lp("="^100)
@printf("FULL RESULT: knitro_status=%d wall=%.1fs n_eval=%d n_grad=%d kappa=%.10f\n",
    result.knitro_status, result.wall, result.n_eval, result.n_grad, result.kappa)
if result.best !== nothing
    @printf("  best feasible: gp=%.10f Delta=%.10f found_at_eval=%d t=%.1fs\n",
        result.best.gp, result.best.Delta, result.best.n_eval, result.best.t)
else
    lp("  NO feasible incumbent found")
end
@printf("TOTAL WALL: %.2fs\n", time() - t0_total)
lp("DONE")
