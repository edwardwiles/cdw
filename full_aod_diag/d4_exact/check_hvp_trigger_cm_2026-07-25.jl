# ============================================================================
# Section 7 trigger check: "If Hessian callbacks still consume more than 50% of inner-solve
# wall time [after threading], perform a bounded HVP comparison." Measures the REAL post-
# threading Hessian share directly via a complete real inner CC dual solve + @prof instrumentation,
# rather than estimating from the isolated per-callback numbers in test_cm_threaded_hessian.jl.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/check_hvp_trigger_cm_2026-07-25.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf

lp(xs...) = (println(xs...); flush(stdout))
lp(">>> Threads.nthreads() = ", Threads.nthreads())

using Serialization
# Reuse the already-known-hard point saved by c14_find_hard_cm_point.jl (checked in at
# results/fullA_d4/c14_parallel_prod/hard_cm_point.jls, base production/fullA-exact): wall=30.4s,
# n_fg=10, n_hess=9 at delta=1 -- a genuinely hard, many-Hessian-callback inner solve, unlike the
# calibration point (which converges too fast to invoke the Hessian at all, see this script's
# first run). Built under destination_sample=:all_legacy (length(xf)=401=D^2+1 for D=20), so this
# check uses :all_legacy to match, rather than this task's usual :exclude_row default.
hard_fixture = deserialize(joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod", "hard_cm_point.jls"))
lp(">>> loaded hard point: label=", hard_fixture.hard_point.label, " original_wall=", hard_fixture.hard_point.wall,
   " n_fg=", hard_fixture.hard_point.n_fg, " n_hess=", hard_fixture.hard_point.n_hess,
   " Delta_dual=", hard_fixture.hard_point.Delta_dual)

ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :all_legacy)
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[50]
pcx = build_cm_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, probs = probs)
lp(">>> cctx.use_threaded_bins = ", pcx.cctx.use_threaded_bins)
xf_hard = hard_fixture.hard_point.xf

# Warm-up (untimed) at a DIFFERENT point (calibration): pays JIT cost without leaving obj.x
# converged at xf_hard, which would trivially warm-start the "measured" call below and defeat the
# whole point of testing a genuinely hard COLD solve (obj.x is retained state across calls --
# archC_base_state/archC_verified_state take no warm= kwarg and always use whatever obj.x already
# holds via CS.inner_loop_initial_values(obj)).
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
pcx.ctx_cm.obj.x .= NaN   # force the measured call below to be a genuine COLD solve at xf_hard
prof_reset!()

t0 = time()
base, verify = archC_verified_state(xf_hard, pcx.ctx_cm, pcx.cctx)
t_total = time() - t0
lp(">>> real inner solve: wall=", round(t_total, digits = 3), "s inner_status=", base.inner_status)

rows = prof_summary()
mkpath(joinpath(@__DIR__, "..", "..", "docs", "key_results"))
write_csv_rows(joinpath(@__DIR__, "..", "..", "docs", "key_results", "hvp_trigger_check_cm_prof_2026-07-25.csv"), rows)
for r in rows
    total_s = r.mean_s * r.n
    lp("  [prof] ", r.label, " n=", r.n, " total_s=", round(total_s, digits = 4),
       " mean_s=", round(r.mean_s, digits = 4), " share_of_measured_wall=", round(100 * total_s / t_total, digits = 2), "%")
end

hess_row = findfirst(r -> r.label == "inner_dual_hessian_callback_archC", rows)
if hess_row !== nothing
    hess_share = 100 * (rows[hess_row].mean_s * rows[hess_row].n) / t_total
    lp(">>> HESSIAN CALLBACK SHARE OF REAL INNER-SOLVE WALL TIME (post-threading) = ", round(hess_share, digits = 2), "%")
    lp(">>> section 7 trigger (>50%): ", hess_share > 50.0 ? "FIRES -- proceed to bounded HVP comparison" : "DOES NOT FIRE -- HVP not required")
else
    lp(">>> WARNING: 'inner_dual_hessian_callback_archC' label not found in prof_summary() -- check label name / PROF_ENABLED[]")
end
lp(">>> DONE")
