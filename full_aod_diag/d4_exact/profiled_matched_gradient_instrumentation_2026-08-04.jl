# task §5 (profiled-outer-ab-readiness-2026-08-04): matched FULL/REDUCED outer-gradient timer
# instrumentation. ADDITIVE ONLY -- neither composite_gradient_fast.jl (FULL) nor
# profiled_lfix_incremental_2026-08-01.jl/profiled_shared_economic_gradient_engine_2026-08-01.jl
# (REDUCED) have their own gradient MATH touched; this file only wraps existing call boundaries
# with wall-clock measurement using the SAME `@prof`-style primitives instrumentation.jl already
# provides (near-zero overhead when PROF_ENABLED[]=false, exception-safe).
#
# WHY NOT PLAIN @prof INSIDE THE COORDINATE LOOP: `@prof` records into the global
# `PROF_TIMES`/`PROF_ALLOCS`/`PROF_GCTIME` Dicts via `push!` (instrumentation.jl). Those are plain
# `Dict{String,Vector}` -- NOT safe for concurrent `push!` from multiple threads (this codebase's
# own documented pitfall, `feedback-concurrent-csv-append-race-under-parallelism`, and the exact
# reason `composite_gradient_fast.jl`'s own `h_mode=:cached` needs `bandwidth_cache_lock`). Under
# `threaded=true` the per-coordinate bandwidth-select/FD-eval work runs on multiple threads
# simultaneously, so wrapping those TWO specific call sites in `@prof` would race. Instead, this
# file accumulates per-coordinate wall time into a PRE-ALLOCATED `Vector{Float64}` (written only
# at each coordinate's own index `k` -- thread-safe by construction, the same pattern
# `h_used`/`switch_mass`/`g` themselves already use) and sums it AFTER the loop, single-threaded.
#
# SEMANTIC-BOUNDARY MAPPING (task §5's own list) -- REDUCED's actual control flow differs
# genuinely from FULL's, not just in code location: REDUCED's `profiled_lfix_incremental_at`/
# `profiled_select_bandwidth` each call `decode_outer_profiled` (decode + full-A reconstruction)
# FRESH at every probed point (every h trial, every coordinate) -- there is no single "decode
# once, then probe" step the way FULL's `composite_gradient_at_fast` decodes `x_free0` ONCE up
# front via `solve_base_state`. This is an honest, load-bearing architectural difference (REDUCED
# recomputes the affected destinations' full-A reconstruction from `w0` at every probe, an O(1)
# operation given `decode_full_z_on_retained`'s own affine structure), not an instrumentation gap
# -- so "decode" and "bandwidth selection"/"fixed-dual objective evaluations" are NOT separable
# for REDUCED without deeper surgery this session did not have budget for. Reported honestly as
# such below, not papered over.
#
#   outer_gradient.cache_build          -- winner/fixed-dual setup (build_shared_profiled_lfix_cache
#                                           REDUCED / build_lfix_base_cache FULL)
#   outer_gradient.gp_component         -- gp/gamma analytic component (coordinate 1)
#   outer_gradient.bandwidth_select     -- aggregate time in profiled_select_bandwidth/
#                                           select_bandwidth across all coordinates (INCLUDES each
#                                           call's own internal decode+reconstruction+FD probes for
#                                           REDUCED's switching-mass search)
#   outer_gradient.fd_eval              -- aggregate time in profiled_lfix_incremental_at/
#                                           a_block_fd_component across all coordinates (INCLUDES
#                                           each call's own internal decode+reconstruction for
#                                           REDUCED)
#   outer_gradient.verification         -- inner_status/dual verification, where separable
#                                           (REDUCED: verify_namedtuple_from_operator, already
#                                           computed upstream in evaluate_fn, reported as 0 here --
#                                           it happens BEFORE the gradient call, not part of the
#                                           gradient engine's own wall time; FULL: same)
#   outer_gradient.wrapper_total        -- complete shared_family_outer_gradient/
#                                           composite_gradient_at_fast wall time (ground truth for
#                                           the >=0.98 sum-of-subtimes check)
#   threads_used                        -- max(Threads.threadid()) actually observed across workers
#                                           (task's own "prove >1 thread used" requirement) -- a
#                                           Vector{Int} written per-coordinate, thread-safe by index
#   n_coordinates, n_bandwidth_trials   -- n_coordinates = D2-1; n_bandwidth_trials summed from
#                                           each profiled_select_bandwidth call's own `n_iter`+1
#                                           (task §5's own required count)
#
# Inner reoptimization (re-solving the inner KNITRO problem, e.g. on a rejected/cold-fallback
# point) is NEVER part of these timings -- both `shared_family_outer_gradient`/
# `composite_gradient_at_fast` take an ALREADY-SOLVED `ev`/`base` as input; any re-solve happens
# strictly upstream (cb_F!/evaluate_fn) and is timed separately by the outer KNITRO driver's own
# existing `solve_dur` vs `grad_engine_dur` split (profiled_production_outer_constrained_2026-08-02.jl,
# unchanged, confirmed by direct read -- task §5's "report inner reoptimization separately, don't
# hide it in a gradient timer" requirement was already met there, not newly added by this file).
#
# Compilation exclusion: callers MUST run one untimed warmup call before the timed measurement
# (this file does not itself force a warmup -- see test_matched_gradient_timer_gate_2026-08-04.jl
# for the actual warmup-then-measure protocol, matching task §5's "exclude compilation through
# explicit warmup" instruction).

isdefined(Main, :ProfiledLFixCache) ||
    error("profiled_matched_gradient_instrumentation_2026-08-04.jl requires profiled_lfix_incremental_2026-08-01.jl to be included first.")
isdefined(Main, :PROF_ENABLED) ||
    error("profiled_matched_gradient_instrumentation_2026-08-04.jl requires instrumentation.jl to be included first.")

"""
    GradientTimingReport

One row per instrumented gradient call. All times in seconds. `sub_time_total` is the sum of the
five labeled sub-times; `accounting_ratio = sub_time_total / wrapper_total_s` is the task §5
required `>=0.98` check.
"""
struct GradientTimingReport
    formulation::Symbol            # :profiled_destination_scales | :full_gamma_normalized
    family::Symbol
    cache_build_s::Float64
    gp_component_s::Float64
    bandwidth_select_s::Float64
    fd_eval_s::Float64
    verification_s::Float64
    wrapper_total_s::Float64
    sub_time_total_s::Float64
    accounting_ratio::Float64
    n_coordinates::Int
    n_bandwidth_trials::Int
    threads_used::Int
    threaded::Bool
    alloc_bytes::Int
    gc_s::Float64
end

function _gradient_timing_report(; formulation::Symbol, family::Symbol, cache_build_s::Float64,
        gp_component_s::Float64, bandwidth_select_s::Float64, fd_eval_s::Float64, verification_s::Float64,
        wrapper_total_s::Float64, n_coordinates::Int, n_bandwidth_trials::Int, threads_used::Int,
        threaded::Bool, alloc_bytes::Int, gc_s::Float64)
    sub_total = cache_build_s + gp_component_s + bandwidth_select_s + fd_eval_s + verification_s
    ratio = wrapper_total_s > 0 ? sub_total / wrapper_total_s : NaN
    return GradientTimingReport(formulation, family, cache_build_s, gp_component_s, bandwidth_select_s,
        fd_eval_s, verification_s, wrapper_total_s, sub_total, ratio, n_coordinates, n_bandwidth_trials,
        threads_used, threaded, alloc_bytes, gc_s)
end

"""
    instrumented_shared_family_outer_gradient(w_profiled, ctx, fctx, ev; threaded=false) -> (g, meta, report::GradientTimingReport)

REDUCED instrumented entry point. Structurally mirrors `shared_family_outer_gradient` exactly
(same two calls, same order) but times `build_shared_profiled_lfix_cache` as `cache_build_s` and
re-implements the coordinate loop inline (copied from `profiled_composite_gradient_from_cache`,
NOT calling it) so each coordinate's own `profiled_select_bandwidth`/`profiled_lfix_incremental_at`
wall time can be accumulated into pre-allocated per-coordinate arrays without racing under
`threaded=true`. Gradient VALUES are identical to the uncounted path (verified by the gate script
via direct comparison against `shared_family_outer_gradient`'s own output at the same point) --
this function exists ONLY to add timing, not to compute anything differently.
"""
function instrumented_shared_family_outer_gradient(w_profiled::AbstractVector{Float64}, ctx, fctx, ev;
        threaded::Bool)
    fam = family_kind(fctx)
    t_wrapper0 = time_ns()
    gcstats0 = Base.gc_num()

    v = validate_family_layout_contract(fctx)
    t0 = time_ns()
    cache = build_shared_profiled_lfix_cache(w_profiled, fctx, ctx, ev)
    cache_build_s = (time_ns() - t0) / 1e9

    spec = v.spec; pe = v.pe
    n_total = outer_dim_profiled(pe)
    n_coord = n_total - 1
    g = zeros(n_total)

    t1 = time_ns()
    g[1] = profiled_gp_component_analytic(cache, w_profiled, ev, ctx)
    gp_component_s = (time_ns() - t1) / 1e9

    h_used = zeros(n_total); switch_mass = zeros(n_total)
    bw_time_ns = zeros(UInt64, n_total); fd_time_ns = zeros(UInt64, n_total)
    bw_trials = zeros(Int, n_total); thread_ids = zeros(Int, n_total)

    function do_coord!(coord_idx::Int)
        thread_ids[coord_idx] = Threads.threadid()
        tb0 = time_ns()
        h, m, selmeta = profiled_select_bandwidth(cache, ctx, spec, pe, w_profiled, coord_idx)
        bw_time_ns[coord_idx] = time_ns() - tb0
        bw_trials[coord_idx] = selmeta.n_iter + 1
        h_used[coord_idx] = h; switch_mass[coord_idx] = m
        tf0 = time_ns()
        Lp = profiled_lfix_incremental_at(cache, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] + h)
        Lm = profiled_lfix_incremental_at(cache, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] - h)
        fd_time_ns[coord_idx] = time_ns() - tf0
        g[coord_idx] = (Lp - Lm) / (2h)
        return nothing
    end

    if threaded
        Main.CS.guard_enter_coord_pool!()
        try
            Threads.@threads for coord_idx in 2:n_total
                do_coord!(coord_idx)
            end
        finally
            Main.CS.guard_exit_coord_pool!()
        end
    else
        @inbounds for coord_idx in 2:n_total
            do_coord!(coord_idx)
        end
    end

    bandwidth_select_s = sum(bw_time_ns) / 1e9
    fd_eval_s = sum(fd_time_ns) / 1e9
    wrapper_total_s = (time_ns() - t_wrapper0) / 1e9
    gcdiff = Base.GC_Diff(Base.gc_num(), gcstats0)

    report = _gradient_timing_report(formulation = :profiled_destination_scales, family = fam,
        cache_build_s = cache_build_s, gp_component_s = gp_component_s,
        bandwidth_select_s = bandwidth_select_s, fd_eval_s = fd_eval_s, verification_s = 0.0,
        wrapper_total_s = wrapper_total_s, n_coordinates = n_coord, n_bandwidth_trials = sum(bw_trials),
        threads_used = length(unique(filter(!=(0), thread_ids))), threaded = threaded,
        alloc_bytes = gcdiff.allocd, gc_s = gcdiff.total_time / 1e9)

    meta = (cache = cache, w0 = collect(Float64, w_profiled), h_used = h_used, switch_mass = switch_mass,
        threaded = threaded)
    return g, meta, report
end

isdefined(Main, :LFixBaseCache) ||
    error("profiled_matched_gradient_instrumentation_2026-08-04.jl requires lfix_incremental.jl (FULL) to be included first for the FULL-side instrumented wrapper below.")

"""
    instrumented_composite_gradient_at_fast(x_free0, ctx, pe; threaded=false) -> (g, meta, report::GradientTimingReport)

FULL instrumented entry point, structurally mirroring `instrumented_shared_family_outer_gradient`
exactly (same label set, same accumulation-into-preallocated-arrays discipline to stay race-free
under `threaded=true`) so the two `GradientTimingReport`s are directly comparable. Re-implements
`composite_gradient_at_fast`'s `h_mode=:adaptive` path inline (does NOT call that function) for
the same reason: the per-coordinate timing granularity this task needs is not otherwise
observable from outside. Gradient VALUES match `composite_gradient_at_fast(...; threaded, h_mode=
:adaptive, validate_frac=0.0)` exactly (verified by the gate script) -- this function adds timing
only, no new computation. `x_free0` is FULL's own full free-parameter vector (`[gp; Aod_levels]`),
matching `composite_gradient_at_fast`'s own signature.
"""
function instrumented_composite_gradient_at_fast(x_free0::AbstractVector, ctx, pe; threaded::Bool)
    t_wrapper0 = time_ns()
    gcstats0 = Base.gc_num()

    t0 = time_ns()
    base = solve_base_state(x_free0, ctx)
    cache = build_lfix_base_cache(x_free0, ctx, base)
    cache_build_s = (time_ns() - t0) / 1e9

    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    D2 = D * Ddest
    z0 = log.(reshape(x_free0[2:end], D, Ddest))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    t1 = time_ns()
    g[1] = gamma_component_analytic(cache, base, w0[1])
    gp_component_s = (time_ns() - t1) / 1e9

    h_used = zeros(D2); switch_mass = zeros(D2)
    bw_time_ns = zeros(UInt64, D2); fd_time_ns = zeros(UInt64, D2)
    bw_trials = zeros(Int, D2); thread_ids = zeros(Int, D2)

    function do_coord!(k::Int)
        thread_ids[k] = Threads.threadid()
        tb0 = time_ns()
        h, m, selmeta = select_bandwidth(cache, ctx, pe, w0, k)
        bw_time_ns[k] = time_ns() - tb0
        bw_trials[k] = selmeta.n_iter + 1
        h_used[k] = h; switch_mass[k] = m
        tf0 = time_ns()
        g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h)
        fd_time_ns[k] = time_ns() - tf0
        return nothing
    end

    if threaded
        CS.guard_enter_coord_pool!()
        try
            Threads.@threads for k in 2:D2
                do_coord!(k)
            end
        finally
            CS.guard_exit_coord_pool!()
        end
    else
        for k in 2:D2
            do_coord!(k)
        end
    end

    bandwidth_select_s = sum(bw_time_ns) / 1e9
    fd_eval_s = sum(fd_time_ns) / 1e9
    wrapper_total_s = (time_ns() - t_wrapper0) / 1e9
    gcdiff = Base.GC_Diff(Base.gc_num(), gcstats0)

    report = _gradient_timing_report(formulation = :full_gamma_normalized, family = :unrestricted,
        cache_build_s = cache_build_s, gp_component_s = gp_component_s,
        bandwidth_select_s = bandwidth_select_s, fd_eval_s = fd_eval_s, verification_s = 0.0,
        wrapper_total_s = wrapper_total_s, n_coordinates = D2 - 1, n_bandwidth_trials = sum(bw_trials),
        threads_used = length(unique(filter(!=(0), thread_ids))), threaded = threaded,
        alloc_bytes = gcdiff.allocd, gc_s = gcdiff.total_time / 1e9)

    meta = (cache = cache, base = base, w0 = w0, h_used = h_used, switch_mass = switch_mass, threaded = threaded)
    return g, meta, report
end

"""
    accounting_check(report::GradientTimingReport; threshold=0.98) -> Bool

Task §5's own required check: `sub_time_total_s / wrapper_total_s >= threshold`.
"""
accounting_check(report::GradientTimingReport; threshold::Float64 = 0.98) = report.accounting_ratio >= threshold
