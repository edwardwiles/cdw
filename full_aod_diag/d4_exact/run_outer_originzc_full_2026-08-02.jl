# Performance closeout task (2026-08-02), Section 13, FULL arm: real outer search for origin-ZC
# via the actual production checkpointed driver (run_originzc_upper_checkpointed,
# cm_originzc_checkpoint.jl) -- unchanged, called as designed.
#
# ctx-construction kwargs (W, delta, find_smallest, draw_design, draw_seed, destination_sample,
# exclude_diagonal_gravity, gravity_exclude_cells, sigmaHat) are IDENTICAL to
# run_outer_originzc_reduced_constrained_2026-08-02.jl's own explicit d20_real_setup_design call --
# this driver rebuilds its own ctx internally from these SAME arguments
# (d20_real_setup_design(W=W, delta=delta, ..., sigmaHat=sigmaHat), confirmed by direct source read
# of cm_originzc_checkpoint.jl -- a deterministic function of these arguments given the fixed
# draw_seed), so the two processes solve against the SAME real data/calibration without needing to
# share a ctx object across processes. Prints the same kappa/gp0/bounds diagnostic the REDUCED
# script prints, for an empirical cross-check (belt-and-suspenders on top of the determinism
# argument).
#
# distribution_restriction=:origin_specific_moments, K_mean=1, K_pair=0 matches the REDUCED script's
# own OriginByPowerLayout(D,1,0) exactly (mean-only restriction, K_pair forced to 0 -- confirmed via
# OriginZCConfig's own docstring).
#
# nu (eta, in log-space) is PINNED to log(1.0)=0 for every origin via nu_bounds -- a zero-width box
# -- to match the REDUCED arm, which never treats nu as a free KNITRO variable at all (see that
# script's own header). Without this, the FULL arm would be solving a materially easier/different
# problem (free nu) than the REDUCED arm (fixed nu), which would confound the comparison.
#
# A_coordinate_mode=:legacy_z (not the newer :powered_aspace default) -- explicit, documented
# "byte-identical to every pre-existing production run" mode, avoiding the aspace-transform's own
# separate correctness surface for this comparison; validated at real D=20/W=80,000 for origin-ZC
# in test_transformed_a_restricted_families_d20.jl (both :legacy_z and :powered_aspace pass there).
#
# Usage: julia run_outer_originzc_full_2026-08-02.jl <W> <maxtime_real_s>
const D4X = @__DIR__
t0_total = time()
# Verbatim include list from test_transformed_a_restricted_families_d20.jl (the known-working real
# D20/W80,000 gate that already calls run_originzc_upper_checkpointed successfully with
# destination_sample=:exclude_row and both A_coordinate_mode values), plus direction_bounds.jl for
# frechet_benchmark_gp.
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

t_ctx = @elapsed ctx1 = d20_real_setup_design(W = W_VAL, δ = DELTA, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
D = ctx1.D; Ddest = ctx1.D_dest
@printf("MATCHED-CTX DIAGNOSTIC: context build=%.2fs  D=%d Ddest=%d  kappa_upper=%.10f  gp0=%.10f  gp_bounds=(%.6f,%.6f)\n",
    t_ctx, D, Ddest, ctx1.bounds.κ_max, frechet_benchmark_gp(ctx1), ctx1.bounds.γp_lo, ctx1.bounds.γp_hi)
flush(stdout)

pe1 = build_pivot_elimination(ctx1)
w0_econ = cm_w0_from_calibration(ctx1, pe1, :legacy_z)
eta0 = zeros(D)   # log(nu)=0 <=> nu=1.0 for every origin -- matches REDUCED's own fixed nuvec0=fill(1.0,D)
w0 = vcat(w0_econ, eta0)
nu_bounds = [(-1e-8, 1e-8) for _ in 1:D]   # PIN nu at ~1.0 -- OriginZCConfig requires lo<hi strictly
# (a literal zero-width box errors: "not a valid (lo<hi) interval"), so use a negligibly narrow one
# instead (relative nu range ~2e-8, i.e. effectively fixed for any practical purpose) -- matches
# REDUCED's never treating nu as free at all, to the precision KNITRO's own box-bound mechanism allows.
@printf("w0: gp0=%.10f  length(zfree)=%d  length(eta0)=%d  (nu PINNED at 1.0 for all origins)\n",
    w0[1], length(w0_econ) - 1, length(eta0))
flush(stdout)

ckpt_dir = mktempdir()
outdir = joinpath(D4X, "..", "..", "docs")
mkpath(outdir)

lp("="^100); lp("FULL constrained outer search: origin-ZC  W=$(W_VAL)  maxtime_real=$(MAXT)s  delta=$(DELTA)")
lp("="^100)
result = run_originzc_upper_checkpointed(w0; W = W_VAL, delta = DELTA, draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    distribution_restriction = :origin_specific_moments, K_mean = 1, K_pair = 0,
    power_target_layout = :origin_by_power, nu_bounds = nu_bounds,
    A_coordinate_mode = :legacy_z, outer_direct_hessopt = :sr1,
    maxtime_real = MAXT, ckpt_dir = ckpt_dir, run_id = "originzc_full_w$(W_VAL)",
    label = "originzc_full_w$(W_VAL)", checkpoint_interval_s = 60.0,
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
