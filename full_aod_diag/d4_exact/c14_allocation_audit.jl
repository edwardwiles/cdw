# Continuation 14, Task 3: allocation audit across the 7 call sites the brief lists:
#   1. unrestricted moment construction (compressed_moments.jl / structured_moment_build.jl)
#   2. unrestricted Hessian callback (oracle_fast.jl dense path + compressed_live.jl compressed path)
#   3. CM structured Hessian callback (cm_hessian_architectures.jl) -- reuses c14_cm_hessian_benchmark's
#      OWN already-collected numbers (results/fullA_d4/c14_parallel_prod/cm_hessian_benchmark.csv,
#      hess_cb_alloc_bytes column) rather than re-running expensive D20/L50 KNITRO solves a second
#      time -- same physical measurement, no need to duplicate the compute.
#   4. build_lfix_base_cache / L_fix gradient base cache (lfix_incremental.jl / lfix_buffer_reuse.jl)
#   5. one coordinate probe (a_block_fd_component!, the function composite_gradient_fast.jl's
#      do_coord! closure actually calls per coordinate -- do_coord! itself is a local closure, not
#      separately callable)
#   6. a complete gradient call (composite_gradient_at_fast_buffered)
#   7. primal recovery / residual checking -- reuses compressed_live.jl's OWN existing @prof labels
#      (primal_weight_recovery_compressed, kkt_residual_compute_compressed,
#      primal_divergence_compute_compressed, moment_resid_compute_compressed), obtained "for free"
#      by running one real compressed value evaluation (sites 1/2/7 are all labeled inside the SAME
#      already-instrumented compressed callback -- no new instrumentation needed for those).
#
# Method: @timed (bytes allocated, gctime) as the primary measurement (matches instrumentation.jl's
# own @prof mechanism, which ALSO already records mean_alloc_bytes/total_gc_s per labeled stage --
# reused directly for sites 1/2/7 rather than re-instrumented), PLUS Base.gc_live_bytes() before/
# after each site as a peak-live-memory proxy per the brief's own suggested methods. One warmup call
# per site excluded from the reported numbers (JIT compilation cost is not an allocation-pattern
# finding).
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
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod")
mkpath(OUTDIR)

gc_live_mb() = Base.gc_live_bytes() / 2^20

"""
    measure(f, label; warmup=true)

Run `f()` once as a (discarded) warmup if `warmup`, then once for real, reporting
(wall, bytes, gctime, gc_live_delta_mb). Returns (result, stats_namedtuple).
"""
function measure(f, label; warmup = true)
    warmup && f()
    GC.gc(); live0 = gc_live_mb()
    stats = @timed f()
    live1 = gc_live_mb()
    lp(@sprintf("  [%-42s] wall=%8.4fs  bytes=%.3e (%.1f MB)  gctime=%7.4fs (%.1f%%)  gc_live_delta=%.2fMB",
        label, stats.time, stats.bytes, stats.bytes/2^20, stats.gctime,
        stats.time > 0 ? 100*stats.gctime/stats.time : 0.0, live1 - live0))
    return stats.value, (label = label, wall = stats.time, bytes = stats.bytes, mb = stats.bytes/2^20,
                          gctime = stats.gctime, gc_pct = stats.time > 0 ? 100*stats.gctime/stats.time : 0.0,
                          gc_live_delta_mb = live1 - live0)
end

lp("=== c14_allocation_audit === ", Dates.now())
W = 80000; DELTA = 1.0
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d", time()-t0, D, W))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0 * 1.01, zfree0)
xf0 = x_free_from_w2(w0)

all_stats = NamedTuple[]

lp(""); lp("-"^100); lp("SITES 1/2/7: one real compressed unrestricted value evaluation"); lp("-"^100)
lp("(build/moment-construction, Hessian callback, primal recovery/residual checking all already")
lp(" @prof-labeled inside evaluate_fullA_screened_ranged's compressed path -- reused, not reinstrumented)")
# warmup (JIT) -- discard prof state from it
evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed, cache = nothing,
    use_cache = false, warm = false, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
# IMPORTANT (bug found live in c14_cm_hessian_benchmark.jl, same root cause here): ctx.obj.x is
# production's own warm-start cache (PsiObjectiveBundleImplicit.x), overwritten with the converged
# dual on every successful solve and reused as the NEXT solve's start whenever use_cached_x=true
# and norm(obj.x)<1e6. Without resetting it, the warmup call above would leave the REAL timed call
# below trivially warm-started from ITS OWN prior solution at the identical point (1 FG call, ~0
# further work) -- not a genuine cold-solve allocation measurement. Reset to force a fresh solve.
ctx.obj.x .= NaN
GC.gc(); live0 = gc_live_mb()
prof_reset!()
stats127 = @timed evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed, cache = nothing,
    use_cache = false, warm = false, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
live1 = gc_live_mb()
lp(@sprintf("  [%-42s] wall=%8.4fs  bytes=%.3e (%.1f MB)  gctime=%7.4fs  gc_live_delta=%.2fMB",
    "TOTAL cold compressed value eval", stats127.time, stats127.bytes, stats127.bytes/2^20, stats127.gctime, live1-live0))
rows127 = prof_summary()
for lbl in ("inner_moment_build_compressed", "inner_dual_hessian_callback_compressed",
            "primal_weight_recovery_compressed", "kkt_residual_compute_compressed",
            "primal_divergence_compute_compressed", "moment_resid_compute_compressed",
            "materialize_dense_for_postproc", "gravity_compute_compressed", "winner_compute_compressed")
    r = findfirst(x -> x.label == lbl, rows127)
    if r === nothing
        lp(@sprintf("  [%-42s] (label not present this call)", lbl))
        continue
    end
    row = rows127[r]
    tot_bytes = row.total_alloc_bytes
    lp(@sprintf("  [%-42s] n=%3d  total=%8.4fs  total_bytes=%.3e (%.1f MB)  total_gc=%7.4fs",
        lbl, row.n, row.n*row.mean_s, tot_bytes, tot_bytes/2^20, row.total_gc_s))
    push!(all_stats, (label = lbl, wall = row.n*row.mean_s, bytes = tot_bytes, mb = tot_bytes/2^20,
                       gctime = row.total_gc_s, gc_pct = (row.n*row.mean_s) > 0 ? 100*row.total_gc_s/(row.n*row.mean_s) : 0.0,
                       gc_live_delta_mb = NaN))
end
push!(all_stats, (label = "TOTAL_cold_compressed_value_eval", wall = stats127.time, bytes = stats127.bytes,
                   mb = stats127.bytes/2^20, gctime = stats127.gctime,
                   gc_pct = stats127.time>0 ? 100*stats127.gctime/stats127.time : 0.0, gc_live_delta_mb = live1-live0))

lp(""); lp("-"^100); lp("SITE 2 (also): dense/unrestricted Hessian callback via oracle_fast.jl (hessopt=1 dense path)"); lp("-"^100)
θ_full0 = CS.reconstruct_full(xf0, ctx.m)
ctx.obj.x .= NaN   # force cold start (see c14_cm_hessian_benchmark.jl for why this matters)
inner_loop_internal_profiled(ctx.obj, θ_full0)   # warmup
ctx.obj.x .= NaN
prof_reset!()
_, stat_dense = measure(() -> inner_loop_internal_profiled(ctx.obj, θ_full0), "dense unrestricted cold value (oracle_fast)"; warmup = false)
push!(all_stats, stat_dense)
rows_dense = prof_summary()
for lbl in ("inner_moment_build", "inner_dual_hessian_callback")
    r = findfirst(x -> x.label == lbl, rows_dense)
    r === nothing && continue
    row = rows_dense[r]
    tot_bytes = row.total_alloc_bytes
    lp(@sprintf("  [%-42s] n=%3d  total=%8.4fs  total_bytes=%.3e (%.1f MB)  total_gc=%7.4fs",
        lbl, row.n, row.n*row.mean_s, tot_bytes, tot_bytes/2^20, row.total_gc_s))
    push!(all_stats, (label = lbl*"_dense", wall = row.n*row.mean_s, bytes = tot_bytes, mb = tot_bytes/2^20,
                       gctime = row.total_gc_s, gc_pct = (row.n*row.mean_s) > 0 ? 100*row.total_gc_s/(row.n*row.mean_s) : 0.0,
                       gc_live_delta_mb = NaN))
end

lp(""); lp("-"^100); lp("SITE 4: build_lfix_base_cache"); lp("-"^100)
base = solve_base_state(xf0, ctx)
_, s4 = measure(() -> build_lfix_base_cache(xf0, ctx, base; validate_dense = false), "build_lfix_base_cache")
push!(all_stats, s4)
cache = build_lfix_base_cache(xf0, ctx, base; validate_dense = false)

lp(""); lp("-"^100); lp("SITE 5: one coordinate probe (a_block_fd_component!)"); lp("-"^100)
q_buf = Vector{Float64}(undef, cache.W); psi_buf = Vector{Float64}(undef, cache.W)
coord_k = 2   # first non-gravity coordinate
_, s5 = measure(() -> a_block_fd_component!(q_buf, psi_buf, cache, ctx, pe, w0, coord_k, 0.01), "a_block_fd_component! (one coord, buffers preallocated)")
push!(all_stats, s5)

lp(""); lp("-"^100); lp("SITE 6: complete gradient call (composite_gradient_at_fast_buffered)"); lp("-"^100)
bwc = Dict{Int,Float64}()
_, s6 = measure(() -> composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc), "composite_gradient_at_fast_buffered (threaded=true, full D2=$(D2) coords)")
push!(all_stats, s6)

lp(""); lp("-"^100); lp("SITE 3: CM structured Hessian callback -- see results/fullA_d4/c14_parallel_prod/cm_hessian_benchmark.csv"); lp("-"^100)
cm_csv = joinpath(OUTDIR, "cm_hessian_benchmark.csv")
if isfile(cm_csv)
    lp("  (reusing c14_cm_hessian_benchmark.jl's own hess_cb_alloc_bytes/hess_cb_gc_s columns -- see that CSV; not re-run here to avoid a duplicate D20/L50 KNITRO cost)")
else
    lp("  cm_hessian_benchmark.csv not found yet -- run c14_cm_hessian_benchmark.jl first for site 3's numbers")
end

lp(""); lp("="^100); lp("SUMMARY (sites 1,2,4,5,6,7 -- site 3 in cm_hessian_benchmark.csv)"); lp("="^100)
lp(@sprintf("%-45s %10s %12s %10s %8s", "site", "wall_s", "bytes", "MB", "gc_pct"))
for s in all_stats
    lp(@sprintf("%-45s %10.4f %12.3e %10.2f %8.2f", s.label, s.wall, s.bytes, s.mb, s.gc_pct))
end

write_csv_rows(joinpath(OUTDIR, "allocation_audit.csv"), all_stats)
lp(""); lp(">>> wrote ", joinpath(OUTDIR, "allocation_audit.csv"))
lp("DONE_C14_ALLOCATION_AUDIT")
