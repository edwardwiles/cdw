# ============================================================================
# unify-random-draw-production-pipeline, 2026-07-30, task §20: performance/memory gate.
# Measures draw generation, context construction, and screen construction wall-clock + peak RSS
# for each draw design at a moderate W (80,000, matching the existing production default) and, if
# time/memory budget allows, at the campaign's target W=500,000 (construction only -- no outer-loop
# KNITRO optimization, which is out of scope for "do not launch a production campaign").
#
# Usage: julia --project=. full_aod_diag/d4_exact/measure_draw_pipeline_resource_gate_2026-07-30.jl [W]
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80_000

function peak_rss_gb()
    # Linux /proc self status VmHWM = peak resident set size ("high water mark")
    for line in eachline("/proc/self/status")
        if startswith(line, "VmHWM:")
            kb = parse(Int, split(line)[2])
            return kb / (1024^2)
        end
    end
    return NaN
end

println("="^90)
println("DRAW_PIPELINE_RESOURCE_GATE  W=", W, "  (", Threads.nthreads(), " threads)")
println("="^90)

rows = NamedTuple[]
for design in (:pseudorandom, :sobol_randomized, :halton_scrambled)
    GC.gc()
    t0 = time()
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = design, draw_seed = 20260719)
    t_total = time() - t0
    rss = peak_rss_gb()
    u_bytes = sizeof(ctx.U)
    row = (design = design, W = W,
           t_uniform_and_transform_s = ctx.draw_meta.timing.uniform_and_transform,
           t_ctx_build_s = ctx.draw_meta.timing.ctx_build,
           t_pairwise_s = ctx.draw_meta.timing.pairwise,
           t_witness_s = ctx.draw_meta.timing.witness,
           t_total_s = t_total,
           draw_matrix_bytes = u_bytes,
           draw_matrix_gb = u_bytes / 1024^3,
           peak_rss_gb = rss)
    push!(rows, row)
    @printf("%-18s  total=%7.2fs  draw_gen=%7.2fs  ctx=%7.2fs  pairwise=%6.2fs  witness=%6.2fs  U=%.4fGB  peak_RSS=%.3fGB\n",
        design, t_total, coalesce(row.t_uniform_and_transform_s, NaN), row.t_ctx_build_s,
        row.t_pairwise_s, row.t_witness_s, row.draw_matrix_gb, rss)
    ctx = nothing
end

open(joinpath(@__DIR__, "..", "..", "docs", "key_results", "draw_pipeline_resource_gate_W$(W)_2026-07-30.csv"), "w") do io
    println(io, "design,W,t_uniform_and_transform_s,t_ctx_build_s,t_pairwise_s,t_witness_s,t_total_s,draw_matrix_bytes,draw_matrix_gb,peak_rss_gb")
    for r in rows
        println(io, r.design, ",", r.W, ",", r.t_uniform_and_transform_s, ",", r.t_ctx_build_s, ",",
                r.t_pairwise_s, ",", r.t_witness_s, ",", r.t_total_s, ",", r.draw_matrix_bytes, ",",
                r.draw_matrix_gb, ",", r.peak_rss_gb)
    end
end
println("Wrote docs/key_results/draw_pipeline_resource_gate_W$(W)_2026-07-30.csv")
