# Continuation 14 (integration/fullA-cm-parallel-production), Task 1: find/construct a genuinely
# hard CM L=50 point (target: ~7-10 Hessian callbacks, ~20-35s inner solve -- NOT an easy
# near-calibration case) and a near-infeasible CM point, for use as the benchmark fixture in Task 2
# (combined CM-Hessian threaded/BLAS benchmark).
#
# Method, and why: results/fullA_d4/c13_perf_comparison/CM_L50/prof_summary.csv (a real full outer
# CM L=50 run, DRAW_SEED=20260719, delta=1, committed on this branch already) shows
# inner_dual_hessian_callback_archC fired 394 times over 44 inner_knitro_dual_solve_arch calls
# (mean 8.95/solve) with per-solve wall time median 30.3s (min 11.2s, max 165.6s) -- i.e. the
# brief's target difficulty is exactly the TYPICAL point on that real trajectory, not an outlier.
# That run's own per-iterate x vectors were never saved to disk (only the final best-feasible
# incumbent was checkpointed), so this script reconstructs a comparably-hard point directly:
# same CM production context (L=50, cumulative/structured, :anchored, nested-grid cutpoints,
# CMConfig production entry point per docs/fullA_common_marginals_production_integration.md sec 3),
# same calibration start (gp0*1.01 cold-start convention, c10_canonical_benchmark.jl /
# timing_harness.jl), reduced-coordinate (zfree) perturbations along ONE fixed random direction
# (seed 777, same normalization convention timing_harness.jl uses for its "nearby"/"distant"
# points) at increasing magnitude, instrumented via cm_hessian_architectures.jl's own
# _INNER_CALL_COUNTERS (n_hess_calls) -- the same counter cm_base_state_v2/
# inner_loop_KNITRO_archgeneric already populate, not a new counter. The magnitude whose
# (n_hess_calls, wall) lands closest to (7-10, 20-35s) is kept as "hard_cm_point"; the smallest
# magnitude at which the inner solve fails outright (nStatus not in {0,-100,-101,-103}) is the
# infeasible witness, and the largest CONVERGED magnitude just below it is kept as
# "near_infeasible_cm_point" (matches docs/fullA_D20_infeasibility_screening_report.md sec 4.1's
# own finding that |step| in [2,10] on this reduced coordinate scale reliably produces genuine
# infeasible D=20 points -- reused convention, not invented here).
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod")
mkpath(OUTDIR)

lp("=== c14_find_hard_cm_point === ", Dates.now())
W = 80000; DELTA = 1.0
Random.seed!(20260719)   # SAME seed as c13_perf_CM_L50.jl / c13_d20_cm_upper_continuation.jl
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d", time()-t0, D, W))

x_free_calib = ctx.θ0_up[ctx.free_idx]
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

snaps = nested_grid_sequence([10, 20, 50])
cfg = CMConfig(common_marginals = true, cm_grid_rule = :nested_family, cm_grid_sizes = [10, 20, 50],
               cm_basis = :cumulative, cm_hessian_backend = :structured, contrasts = :anchored)
t0 = time()
pcx = build_cm_production_context_v2(ctx, CS, cfg; L = 50)
lp(@sprintf(">>> CM production context (L=50, CMConfig v2) built in %.1fs. ncm=%d d_total=%d",
    time()-t0, pcx.aug.ncm, pcx.ctx_cm.obj.d))
# Sanity: nested-grid cutpoints match what c13_perf_CM_L50.jl's L=50 stage used.
@assert cm_resolve_probs_for_L(cfg, 50) == snaps[50]

function eval_point(label, w; catch_errors = true)
    xf = x_free_from_w(w)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    t0 = time()
    ok = true
    K = base = nothing
    errmsg = ""
    try
        K, base = cm_production_value_v2(xf, pcx; use_cache = false)
    catch e
        ok = false
        errmsg = sprint(showerror, e)
    end
    wall = time() - t0
    counters = _INNER_CALL_COUNTERS[]
    if ok
        Delta_dual = -base.ζstar
        lp(@sprintf("[%s] wall=%.3fs  n_fg=%d  n_hess=%d  Delta_dual=%.10f  nStatus=%d",
            label, wall, counters.n_fg_calls, counters.n_hess_calls, Delta_dual, base.inner_status))
    else
        lp(@sprintf("[%s] wall=%.3fs  n_fg=%d  n_hess=%d  FAILED: %s",
            label, wall, counters.n_fg_calls, counters.n_hess_calls, first(errmsg, 200)))
    end
    return (label = label, w = copy(w), xf = xf, wall = wall, n_fg = counters.n_fg_calls,
            n_hess = counters.n_hess_calls, ok = ok, K = K, base = base,
            Delta_dual = ok ? -base.ζstar : NaN, errmsg = errmsg)
end

results = NamedTuple[]

lp("-"^100); lp("STAGE 1: calibration + magnitude scan along a fixed reduced-coordinate direction")
lp("-"^100)
push!(results, eval_point("calibration (mag=0)", vcat(gp0 * 1.01, zfree0)))

Random.seed!(777)   # SAME seed timing_harness.jl uses for its "nearby" direction
dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))

for mag in (0.05, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
    w = vcat(gp0 * 1.01, zfree0 .+ mag .* dir)
    push!(results, eval_point("mag=$mag", w))
    # stop scanning further out once we've seen a genuine failure -- no need to push deeper
    if !results[end].ok
        lp(">>> stopping magnitude scan: mag=$mag failed, have both a converged boundary neighbor and a failure")
        break
    end
end

lp(""); lp("="^100); lp("SUMMARY"); lp("="^100)
for r in results
    lp(@sprintf("  %-16s ok=%-5s wall=%8.3fs n_fg=%3d n_hess=%3d Delta_dual=%s",
        r.label, string(r.ok), r.wall, r.n_fg, r.n_hess, r.ok ? @sprintf("%.6f", r.Delta_dual) : "n/a"))
end

# ---- pick the "hard" point: closest to (n_hess in 7:10, wall in 20:35), among converged points ----
converged = filter(r -> r.ok, results)
function hardness_score(r)
    hess_pen = r.n_hess < 7 ? (7 - r.n_hess) : (r.n_hess > 10 ? (r.n_hess - 10) : 0)
    wall_pen = r.wall < 20 ? (20 - r.wall) / 5 : (r.wall > 35 ? (r.wall - 35) / 5 : 0)
    return hess_pen + wall_pen
end
best_idx = argmin([hardness_score(r) for r in converged])
hard_pt = converged[best_idx]
lp(""); lp(">>> SELECTED hard CM point: ", hard_pt.label, "  n_hess=", hard_pt.n_hess, "  wall=", round(hard_pt.wall, digits=2), "s")

# ---- near-infeasible: last converged point immediately before the first failure (if any); else
#      the point with largest Delta_dual (closest to violating delta) among converged ----
first_fail = findfirst(r -> !r.ok, results)
if first_fail !== nothing && first_fail > 1
    near_infeasible_pt = results[first_fail - 1]
    infeasible_witness = results[first_fail]
    lp(">>> near-infeasible witness: first genuine failure at ", infeasible_witness.label,
       "; near-infeasible point = last converged neighbor, ", near_infeasible_pt.label,
       "  Delta_dual=", near_infeasible_pt.Delta_dual)
else
    # no outright failure seen in the scanned range -- fall back to the converged point with the
    # largest Delta_dual (closest to the delta=1 boundary) as the "near-infeasible" proxy, and
    # note this explicitly rather than fabricating a failure.
    near_infeasible_pt = converged[argmax([r.Delta_dual for r in converged])]
    infeasible_witness = nothing
    lp(">>> NO outright inner-solve failure encountered in the scanned magnitude range (0.05 to 5.0). ",
       "Using the converged point with largest Delta_dual as a near-infeasible PROXY: ",
       near_infeasible_pt.label, "  Delta_dual=", near_infeasible_pt.Delta_dual,
       " (delta=", DELTA, "). This is reported honestly as a proxy, not a genuine near-infeasible witness.")
end

payload = (
    created = string(now()), draw_seed = 20260719, W = W, delta = DELTA, L = 50,
    cfg = cfg, direction_seed = 777,
    hard_point = (label = hard_pt.label, w = hard_pt.w, xf = hard_pt.xf, wall = hard_pt.wall,
                  n_fg = hard_pt.n_fg, n_hess = hard_pt.n_hess, Delta_dual = hard_pt.Delta_dual),
    near_infeasible_point = (label = near_infeasible_pt.label, w = near_infeasible_pt.w,
                              xf = near_infeasible_pt.xf, wall = near_infeasible_pt.wall,
                              n_fg = near_infeasible_pt.n_fg, n_hess = near_infeasible_pt.n_hess,
                              Delta_dual = near_infeasible_pt.Delta_dual),
    infeasible_witness = infeasible_witness === nothing ? nothing :
        (label = infeasible_witness.label, w = infeasible_witness.w, xf = infeasible_witness.xf,
         wall = infeasible_witness.wall, n_fg = infeasible_witness.n_fg, n_hess = infeasible_witness.n_hess,
         errmsg = infeasible_witness.errmsg),
    all_scan_results = [(label=r.label, wall=r.wall, n_fg=r.n_fg, n_hess=r.n_hess, ok=r.ok,
                          Delta_dual=r.Delta_dual) for r in results],
)
outpath = joinpath(OUTDIR, "hard_cm_point.jls")
serialize(outpath, payload)
lp(""); lp(">>> saved fixture to ", outpath)
lp("DONE_C14_FIND_HARD_CM_POINT")
