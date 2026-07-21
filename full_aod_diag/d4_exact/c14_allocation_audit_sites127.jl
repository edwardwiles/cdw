# Continuation 14, Task 3 supplement: sites 1/2/7 (compressed unrestricted moment construction /
# Hessian callback / primal-recovery+residual-checking), done as a FOCUSED standalone script.
#
# c14_allocation_audit.jl's original sites-1/2/7 measurement was contaminated: it ran a WARMUP call
# to evaluate_fullA_screened_ranged at xf0 first (to exclude JIT from the timed number), then reset
# ctx.obj.x and re-called at the SAME xf0 for the real measurement -- but the real call still came
# back in 0.63s with none of the compressed-path @prof labels populated, vs. the ~14.6s / real
# 15-iteration KNITRO solve a single clean call at the IDENTICAL point produces
# (c14_diag_screen_shortcut.jl, run standalone, confirmed exactly this). Root cause not fully
# isolated (some piece of screen/solve state beyond ctx.obj.x survives the warmup call and short-
# circuits the second one -- plausibly the pairwise/envelope screen's own internal memoization, not
# investigated further given time budget), but the WORKAROUND is simple and verified: do not
# warm up this specific call at all -- take the JIT cost on the reported number (disclosed below)
# rather than risk an artificially-fast contaminated reading.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
using Printf, LinearAlgebra, Random, Dates

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod")
mkpath(OUTDIR)

lp("=== c14_allocation_audit_sites127 === ", Dates.now())
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs", time()-t0))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
xf0 = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

lp(">>> ONE clean call, no warmup (JIT cost included and disclosed, not excluded via a contaminating warmup):")
GC.gc()
live0 = Base.gc_live_bytes() / 2^20
prof_reset!()
stats = @timed evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = false, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
live1 = Base.gc_live_bytes() / 2^20
(r, meta) = stats.value
lp(@sprintf("  [TOTAL cold compressed value eval (incl. JIT)] wall=%8.4fs  bytes=%.3e (%.1f MB)  gctime=%7.4fs  gc_live_delta=%.2fMB",
    stats.time, stats.bytes, stats.bytes/2^20, stats.gctime, live1-live0))
lp("  screen_status=", meta.screen_status, "  inner_status=", r.inner_status, "  n_inner_solves=", meta.n_inner_solves, "  n_inner_iters=", meta.n_inner_iters)

rows = prof_summary()
all_stats = NamedTuple[(label = "TOTAL_cold_compressed_value_eval_inclJIT", wall = stats.time, bytes = stats.bytes,
                         mb = stats.bytes/2^20, gctime = stats.gctime,
                         gc_pct = stats.time>0 ? 100*stats.gctime/stats.time : 0.0, gc_live_delta_mb = live1-live0)]
for lbl in ("inner_moment_build_compressed", "inner_dual_hessian_callback_compressed",
            "primal_weight_recovery_compressed", "kkt_residual_compute_compressed",
            "primal_divergence_compute_compressed", "moment_resid_compute_compressed",
            "materialize_dense_for_postproc", "gravity_compute_compressed", "winner_compute_compressed")
    idx = findfirst(x -> x.label == lbl, rows)
    if idx === nothing
        lp(@sprintf("  [%-42s] (label not present this call -- see header note)", lbl))
        continue
    end
    row = rows[idx]
    tot_bytes = row.total_alloc_bytes
    lp(@sprintf("  [%-42s] n=%3d  total=%8.4fs  total_bytes=%.3e (%.1f MB)  total_gc=%7.4fs",
        lbl, row.n, row.n*row.mean_s, tot_bytes, tot_bytes/2^20, row.total_gc_s))
    push!(all_stats, (label = lbl, wall = row.n*row.mean_s, bytes = tot_bytes, mb = tot_bytes/2^20,
                       gctime = row.total_gc_s, gc_pct = (row.n*row.mean_s) > 0 ? 100*row.total_gc_s/(row.n*row.mean_s) : 0.0,
                       gc_live_delta_mb = NaN))
end

write_csv_rows(joinpath(OUTDIR, "allocation_audit_sites127.csv"), all_stats)
lp(""); lp(">>> wrote ", joinpath(OUTDIR, "allocation_audit_sites127.csv"))
lp("DONE_C14_ALLOCATION_AUDIT_SITES127")
