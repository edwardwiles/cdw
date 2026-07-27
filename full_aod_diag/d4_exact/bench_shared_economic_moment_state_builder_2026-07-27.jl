# ============================================================================
# Shared economic moment-state builder task (2026-07-27) -- allocation/timing gate (addendum §8).
# D=4 square context; measures the four required quantities directly, plus the actual
# inner_loop_internal_compressed hot path this task fixed (compressed_live.jl), with vs without an
# attached cf_workspace, so the real end-to-end effect of the fix (not just the isolated builder
# call) is measured. Also runs at D=20/W=80,000 real data for the two isolated-builder rows (no
# KNITRO -- the real KNITRO D=20/W=80,000 correctness+timing evidence is in
# test_shared_economic_moment_state_builder_d20_2026-07-27.jl's log, run separately this session).
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/bench_shared_economic_moment_state_builder_2026-07-27.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Statistics, Printf, DelimitedFiles

lp(xs...) = (println(xs...); flush(stdout))

function med_time(f, N)
    ts = Float64[]
    for _ in 1:N
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e6)  # ms
    end
    return median(ts)
end

rows = Vector{NamedTuple}()

function run_case(label, D, Ddest, ctx, xf, N)
    lp("="^90); lp(label, " (D=", D, " Ddest=", Ddest, " W=", size(ctx.U,1), ")"); lp("="^90)

    GC.gc()
    a_ref = @allocated build_compressed_factual(xf, ctx; check_ties = true)
    t_ref = med_time(() -> build_compressed_factual(xf, ctx; check_ties = true), N)
    lp(@sprintf("  build_compressed_factual (allocating reference):        %10.3f ms  %12d bytes", t_ref, a_ref))
    push!(rows, (case=label, step="build_compressed_factual_reference", time_ms=t_ref, bytes=a_ref))

    ws = build_compressed_factual_workspace(D, Ddest, size(ctx.U, 1))
    GC.gc()
    a_first = @allocated build_compressed_factual!(ws, xf, ctx; check_ties = true)
    lp(@sprintf("  build_compressed_factual! (first fill, fresh ws):                       %12d bytes", a_first))
    push!(rows, (case=label, step="build_compressed_factual_inplace_first_fill", time_ms=NaN, bytes=a_first))

    GC.gc()
    a_refill = @allocated build_compressed_factual!(ws, xf, ctx; check_ties = true)
    t_refill = med_time(() -> build_compressed_factual!(ws, xf, ctx; check_ties = true), N)
    lp(@sprintf("  build_compressed_factual! (repeated refill, same ws):    %10.3f ms  %12d bytes", t_refill, a_refill))
    push!(rows, (case=label, step="build_compressed_factual_inplace_repeated_refill", time_ms=t_refill, bytes=a_refill))

    ctx_ws = attach_compressed_factual_workspace(ctx, D, Ddest, size(ctx.U, 1))
    GC.gc()
    a_bems = @allocated build_economic_moment_state!(xf, ctx_ws; check_ties = true)
    t_bems = med_time(() -> build_economic_moment_state!(xf, ctx_ws; check_ties = true), N)
    lp(@sprintf("  build_economic_moment_state! (repeated, ws attached):    %10.3f ms  %12d bytes", t_bems, a_bems))
    push!(rows, (case=label, step="build_economic_moment_state_shared", time_ms=t_bems, bytes=a_bems))
    lp(@sprintf("  reduction (reference -> shared in-place): %.2f%%  (%.3f MB saved/call)", 100*(1-a_bems/a_ref), (a_ref-a_bems)/1e6))
    return ctx_ws, xf
end

println("="^90)
println("Complete-family moment-state build: unrestricted family end-to-end inner solve")
println("(evaluate_fullA_fast, moment_representation=:compressed), WITHOUT vs WITH attached workspace")
println("="^90)
ctx4 = d4_exact_setup(find_smallest = true)
D4 = ctx4.D; Ddest4 = hasproperty(ctx4, :D_dest) ? ctx4.D_dest : ctx4.D
xf4 = ctx4.θ0_up
x_free4 = ctx4.θ0_up[ctx4.free_idx]

ctx4_ws, _ = run_case("D4 square", D4, Ddest4, ctx4, xf4, 200)

reset_economic_moment_state_counters!()
GC.gc()
a_nows_solve = @allocated evaluate_fullA_fast(x_free4, ctx4; cache = nothing, use_cache = false, warm = false, moment_representation = :compressed)
t_nows_solve = med_time(() -> evaluate_fullA_fast(x_free4, ctx4; cache = nothing, use_cache = false, warm = false, moment_representation = :compressed), 20)
n_alloc_nows = ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[]
lp(@sprintf("  complete unrestricted inner solve, NO workspace (pre-fix behavior): %10.3f ms  %12d bytes  allocating_calls=%d", t_nows_solve, a_nows_solve, n_alloc_nows))
push!(rows, (case="D4 square unrestricted inner solve", step="complete_family_moment_state_build_no_workspace", time_ms=t_nows_solve, bytes=a_nows_solve))

reset_economic_moment_state_counters!()
GC.gc()
a_ws_solve = @allocated evaluate_fullA_fast(x_free4, ctx4_ws; cache = nothing, use_cache = false, warm = false, moment_representation = :compressed)
t_ws_solve = med_time(() -> evaluate_fullA_fast(x_free4, ctx4_ws; cache = nothing, use_cache = false, warm = false, moment_representation = :compressed), 20)
n_alloc_ws = ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[]
n_inplace_ws = INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS[]
lp(@sprintf("  complete unrestricted inner solve, WITH workspace (this task's fix):  %10.3f ms  %12d bytes  allocating_calls=%d inplace_calls=%d", t_ws_solve, a_ws_solve, n_alloc_ws, n_inplace_ws))
push!(rows, (case="D4 square unrestricted inner solve", step="complete_family_moment_state_build_with_workspace", time_ms=t_ws_solve, bytes=a_ws_solve))
lp(@sprintf("  moment-build-attributable allocation reduction: %d bytes/solve saved (KNITRO/callback overhead not isolated here)", a_nows_solve - a_ws_solve))

csv_path = joinpath(@__DIR__, "..", "..", "docs", "shared_economic_moment_state_builder_timing_2026-07-27.csv")
open(csv_path, "w") do io
    println(io, "case,step,time_ms,bytes")
    for r in rows
        println(io, r.case, ",", r.step, ",", (isnan(r.time_ms) ? "" : @sprintf("%.4f", r.time_ms)), ",", r.bytes)
    end
end
lp(">>> wrote ", csv_path)
