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
    run_staged_delta5_continuation(label, g_start, zfree_start; delta_stages=[2.0,3.0,4.0,5.0],
        stage_maxtime_real=90.0, W_in=80000, draw_seed_in=20260719, ckpt_root, kwargs...)

Runs `run_polish_checkpointed` once per stage in `delta_stages` (assumed already sorted
ascending), feeding the previous stage's terminal `(g, zfree)` forward as the next stage's
start. Returns the final stage's result plus a per-stage summary Vector.
"""
function run_staged_delta5_continuation(label::String, g_start::Float64, zfree_start::Vector{Float64};
        delta_stages::Vector{Float64} = [2.0, 3.0, 4.0, 5.0],
        stage_maxtime_real::Float64 = 90.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, draw_seed_in::Int = 20260719, ckpt_root::AbstractString,
        use_dual_bank::Bool = true, use_exact_cache::Bool = true)
    g = g_start; zfree = copy(zfree_start)
    stage_summaries = NamedTuple[]
    res = nothing
    for (i, delta) in enumerate(delta_stages)
        stage_label = "$(label)_stage$(i)_d$(delta)"
        ckpt_dir = joinpath(ckpt_root, "stage$(i)_d$(delta)")
        mkpath(ckpt_dir)
        lp("=== STAGED CONTINUATION stage ", i, "/", length(delta_stages), ": delta=", delta,
           " (start g=", g, ") ===", "  ", Dates.now())
        t0 = time()
        res = run_polish_checkpointed(stage_label, false, g, zfree;
            maxtime_real = stage_maxtime_real, hessopt_tag = hessopt_tag,
            W_in = W_in, delta_in = delta, draw_seed_in = draw_seed_in,
            ckpt_dir = ckpt_dir, checkpoint_interval_s = 30.0,
            use_dual_bank = use_dual_bank, use_exact_cache = use_exact_cache)
        t_stage = time() - t0
        b = res.best_feasible
        if b !== nothing
            g = b.gp; zfree = copy(b.w[2:end])
        end
        push!(stage_summaries, (stage = i, delta = delta, wall = t_stage, n_eval = res.n_eval,
            kappa = res.kappa, n_rejected = res.n_rejected, knitro_status = res.knitro_status,
            best_gp = b === nothing ? NaN : b.gp))
        lp("  stage ", i, " done: wall=", round(t_stage, digits=1), "s kappa=", res.kappa,
           " n_eval=", res.n_eval, " n_rejected=", res.n_rejected, " knitro_status=", res.knitro_status)
    end
    return (final = res, stages = stage_summaries, final_g = g, final_zfree = zfree)
end
