# ============================================================================
# Staged delta>=2 workflow (task brief section 8 / plan Phase E): finer
# divergence continuation (delta: start -> 3 -> 4 -> 5, checkpointing the best
# feasible point at each stage as the next stage's start) via repeated calls
# to the EXISTING, unmodified run_polish_checkpointed (joint (gamma',A) solve).
# Exposed as the recommended delta>=2 driver mode; the ordinary direct mode
# (run_polish_checkpointed straight at the target delta) remains available
# and is what this file's own comparison benchmarks against.
#
# Does NOT implement the fixed-g profile continuation (task 8.2, min_A
# Delta*(g,A) at increasing g) or full KNITRO-algorithm staging (task 8.3,
# Interior/CG exploration -> Interior-Direct polish) in this pass -- see the
# handoff doc's Phase E section for what was and wasn't attempted given this
# session's time budget, and why staged DELTA continuation alone (without
# those two refinements) was tested first as the simplest version of the
# idea.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))

"""
    run_staged_delta5_continuation(label, g_start, zfree_start; find_smallest,
        delta_stages=[2.0,3.0,4.0,5.0], stage_maxtime_real=90.0, W_in=80000,
        draw_seed_in=20260719, ckpt_root, kwargs...)

Runs `run_polish_checkpointed` once per stage in `delta_stages` (assumed already sorted
ascending), feeding the previous stage's terminal `(g, zfree)` forward as the next stage's
start. Returns the final stage's result plus a per-stage summary Vector.

`find_smallest` is REQUIRED (no default) -- addendum fix: the pre-fix version of this
function hardcoded `false` unconditionally regardless of which direction the caller
wanted, silently always running the real LOWER-kappa branch (maximize gp) while being
used/labeled as an "upper" continuation throughout this repo's own history. Per the
EVIDENCED direction convention (see direction_bounds.jl: `find_smallest=true` is the
real upper/larger-kappa branch, confirmed by this repo's own explicitly-calibrated
`run_d4_optimized_fd.jl`/`c_canon_run_one.jl` conventions, real established multi-
session numbers, and the algebraic kappa(gp) relationship), callers wanting the upper
continuation must now pass `find_smallest=true` explicitly.
"""
function run_staged_delta5_continuation(label::String, g_start::Float64, zfree_start::Vector{Float64};
        find_smallest::Bool,
        delta_stages::Vector{Float64} = [2.0, 3.0, 4.0, 5.0],
        stage_maxtime_real::Float64 = 90.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, draw_seed_in::Int = 20260719, ckpt_root::AbstractString,
        use_dual_bank::Bool = true, use_exact_cache::Bool = true,
        reuse_context::Bool = true,   # task §5/§6: build the real-data context ONCE and thread it
        # through every stage via set_context_delta! instead of paying the ~65-83s
        # d20_real_setup_design/pe/rsc rebuild at every stage (the diagnosed root cause of the
        # earlier staged-vs-direct comparison's staged arm losing on wall time -- see
        # docs/fullA_driver_delta5_diagnostics_handoff.md §5-6). Set false to reproduce the old
        # per-stage-rebuild behavior exactly (e.g. for an A/B wall-time comparison).
        cross_delta::Bool = false,   # allocation/cache-cleanup task §12: when true (and
        # use_exact_cache=true), construct ONE CrossDeltaExactCache before the stage loop and
        # thread it through every stage's run_polish_checkpointed call via exact_cache_override=,
        # instead of each stage building its own throwaway SafeExactCache(). An exact hit from a
        # DIFFERENT delta-stage at the SAME x_free/find_smallest/context is served without a
        # re-solve, with Delta_minus_delta patched to the current stage's own delta. Requires
        # reuse_context=true (the cache is only valid across stages that share ONE ctx -- see
        # CrossDeltaExactCache's own docstring); errors if reuse_context=false. Default false:
        # zero behavior change unless explicitly requested, per AUD-08's own "only after
        # cache-state and fingerprint work passes" gate (already satisfied on this branch).
        draw_design_in::Symbol = :pseudorandom, inner_opt_override::Union{Nothing,AbstractString} = nothing,
        allow_direction_box_migration::Bool = false)
    cross_delta && !reuse_context && error("run_staged_delta5_continuation($label): cross_delta=true requires reuse_context=true -- a CrossDeltaExactCache is only valid across stages that share one ctx.")
    g = g_start; zfree = copy(zfree_start)
    stage_summaries = NamedTuple[]
    res = nothing
    reuse = nothing
    cross_delta_cache_obj = (cross_delta && use_exact_cache) ? CrossDeltaExactCache() : nothing
    if reuse_context
        lp("=== building reusable context ONCE (task §5), delta_stages[1]=", delta_stages[1],
           " find_smallest=", find_smallest, " (", find_smallest ? "upper" : "lower", ") === ", Dates.now())
        t0 = time()
        reuse = build_fullA_context(W = W_in, δ = delta_stages[1], find_smallest = find_smallest,
            draw_design = draw_design_in, draw_seed = draw_seed_in, inner_loop_opt = inner_opt_override)
        lp("  context build: ", round(time() - t0, digits = 1), "s")
    end
    for (i, delta) in enumerate(delta_stages)
        stage_label = "$(label)_stage$(i)_d$(delta)"
        ckpt_dir = joinpath(ckpt_root, "stage$(i)_d$(delta)")
        mkpath(ckpt_dir)
        lp("=== STAGED CONTINUATION stage ", i, "/", length(delta_stages), ": delta=", delta,
           " (start g=", g, ") ===", "  ", Dates.now())
        t0 = time()
        cache_size_before = cross_delta_cache_obj === nothing ? 0 : length(cross_delta_cache_obj)
        res = run_polish_checkpointed(stage_label, find_smallest, g, zfree;
            maxtime_real = stage_maxtime_real, hessopt_tag = hessopt_tag,
            W_in = W_in, delta_in = delta, draw_seed_in = draw_seed_in,
            ckpt_dir = ckpt_dir, checkpoint_interval_s = 30.0,
            use_dual_bank = use_dual_bank, use_exact_cache = use_exact_cache,
            exact_cache_override = cross_delta_cache_obj,
            reuse = reuse, allow_direction_box_migration = allow_direction_box_migration)
        t_stage = time() - t0
        cache_size_after = cross_delta_cache_obj === nothing ? 0 : length(cross_delta_cache_obj)
        b = res.best_feasible
        if b !== nothing
            g = b.gp; zfree = copy(b.w[2:end])
        end
        if cross_delta_cache_obj !== nothing
            lp("  stage ", i, " cross-delta cache: ", cache_size_before, " -> ", cache_size_after,
               " entries (", res.n_eval, " evals this stage; entries unchanged from before this stage ",
               "were served WITHOUT a re-solve -- an exact size-growth-vs-n_eval gap is the inter-stage ",
               "hit signal, see docs/fullA_postmerge_allocation_productionization.md for the full benchmark)")
        end
        push!(stage_summaries, (stage = i, delta = delta, wall = t_stage, n_eval = res.n_eval,
            kappa = res.kappa, n_rejected = res.n_rejected, knitro_status = res.knitro_status,
            best_gp = b === nothing ? NaN : b.gp,
            cross_delta_cache_size_before = cache_size_before, cross_delta_cache_size_after = cache_size_after))
        lp("  stage ", i, " done: wall=", round(t_stage, digits=1), "s kappa=", res.kappa,
           " n_eval=", res.n_eval, " n_rejected=", res.n_rejected, " knitro_status=", res.knitro_status)
    end
    return (final = res, stages = stage_summaries, final_g = g, final_zfree = zfree)
end
