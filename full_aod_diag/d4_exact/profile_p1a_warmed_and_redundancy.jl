# ============================================================================
# Continuation 5, Priority 1A: (1) fresh warmed component breakdown of the
# exact-hard evaluation at the CURRENT canonical candidate (reuses
# oracle_fast.jl's already-validated nested @prof instrumentation verbatim --
# no new profiling methodology, just a fresh run at the new point/commit),
# and (2) the NEW audit this continuation's task explicitly asks for: does a
# KNITRO outer iterate's cb_F! (objective/constraint callback) followed by
# cb_G! (gradient callback) at the SAME x redundantly build the moment matrix
# and re-solve the inner CC dual TWICE? Measured directly, not assumed.
# ============================================================================
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
using Statistics, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_p1a_warmed_and_redundancy")
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

function x_free_from_w(w::AbstractVector)
    z = pivot_expand(w[2:end], pe)
    return vcat(w[1], vec(exp.(z)))
end

"Minimal stats_row matching prof_summary()'s exact field set (mirrors profile_components_v2.jl's own helper, not re-imported to avoid a naming clash on N_REPS)."
function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted)/n
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            p90_s = sorted[clamp(ceil(Int, 0.9*n), 1, n)], p95_s = sorted[clamp(ceil(Int, 0.95*n), 1, n)],
            mean_s = μ, std_s = (n > 1 ? sqrt(sum((t-μ)^2 for t in sorted)/(n-1)) : 0.0),
            mean_alloc_bytes = NaN, total_alloc_bytes = NaN, mean_gc_s = NaN, total_gc_s = NaN)
end

const W_CAND = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf = x_free_from_w(W_CAND)

# ============================================================================
# PART 1: fresh warmed breakdown at the canonical candidate (was calibration
# point in the original performance_profile_v2.md; here the actual point KNITRO
# spends its time near). Reuses evaluate_fullA_fast verbatim.
# ============================================================================
println("="^78); println("PART 1: warmed evaluate_fullA_fast breakdown at upper_lfixcomposite_sr1_60s"); println("="^78)
evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true)   # JIT warm-up, untimed
prof_reset!()
const N_REPS = 50
total_times = Float64[]
for _ in 1:N_REPS
    t0 = time_ns(); evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true)
    push!(total_times, (time_ns() - t0) / 1e9)
end
summ = prof_summary()
push!(summ, stats_row("TOTAL_evaluate_fullA_fast", total_times))
write_csv_rows(joinpath(OUTDIR, "part1_warmed_breakdown.csv"), summ)
total_row = only(filter(r -> r.label == "TOTAL_evaluate_fullA_fast", summ))
for r in sort(summ, by = r -> -r.mean_s)
    r.label == "TOTAL_evaluate_fullA_fast" && continue
    @printf("  %-30s median=%.4fms  pct_of_total=%.1f%%\n", r.label, r.median_s*1000, 100*r.mean_s/total_row.mean_s)
end
@printf("  TOTAL (median) = %.4fms\n", total_row.median_s*1000)

# ============================================================================
# PART 2: the F+G-at-same-x redundancy audit
# ============================================================================
println("\n" * "="^78); println("PART 2: does cb_F! + cb_G! at the SAME x pay the inner solve/moment build TWICE?"); println("="^78)

# ---- (a) eval_F alone (mirrors run_d4_optimized_fd.jl's cb_F! -> eval_F -> evaluate_fullA) ----
evaluate_fullA(xf, ctx; cache = nothing, warm = true)   # warm-up
n0 = CS.INNER_SOLVE_COUNT[]
t_F = median([@elapsed evaluate_fullA(xf, ctx; cache = nothing, warm = true) for _ in 1:30])
n_solves_F = CS.INNER_SOLVE_COUNT[] - n0   # over 30 reps; divide by 30 for per-call

# ---- (b) eval_grad_dispatch(:lfix_composite) alone, AS CURRENTLY WIRED in run_d4_optimized_fd.jl
#      (composite_gradient_at called WITHOUT a `base` argument -- always re-solves) ----
composite_gradient_at(xf, ctx, pe)   # warm-up
n0 = CS.INNER_SOLVE_COUNT[]
t_G_unshared = median([@elapsed composite_gradient_at(xf, ctx, pe) for _ in 1:30])
n_solves_G_unshared = CS.INNER_SOLVE_COUNT[] - n0

# ---- (c) the ACTUAL KNITRO callback sequence at one outer iterate: cb_F! then cb_G!, AS CURRENTLY
#      WIRED (no sharing) ----
n0 = CS.INNER_SOLVE_COUNT[]
t_sequence_unshared = median([@elapsed begin
    evaluate_fullA(xf, ctx; cache = nothing, warm = true)
    composite_gradient_at(xf, ctx, pe)
end for _ in 1:30])
n_solves_sequence_unshared = (CS.INNER_SOLVE_COUNT[] - n0)

# ---- (d) the FIX: share the base state solved by eval_F with composite_gradient_at, at the SAME x
#      (composite_gradient_at already accepts an optional `base` kwarg for exactly this -- just never
#      wired up in run_d4_optimized_fd.jl's eval_grad_dispatch). Equivalence-checked below, not just timed. ----
function eval_F_and_shared_base(xf)
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
    # the base state eval_F's inner solve already produced -- rebuild the BaseDualState struct from
    # what evaluate_fullA computed, at ZERO additional inner solves (same theta_full, same converged
    # obj.arg1/inner_x that evaluate_fullA just left inside `obj`)
    base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
    return r, base
end
r0, base0 = eval_F_and_shared_base(xf)   # warm-up
n0 = CS.INNER_SOLVE_COUNT[]
t_sequence_shared = median([@elapsed begin
    r, base = eval_F_and_shared_base(xf)
    composite_gradient_at(xf, ctx, pe; base = base)
end for _ in 1:30])
n_solves_sequence_shared = (CS.INNER_SOLVE_COUNT[] - n0)

@printf("\n  eval_F alone (warmed median):                       %.4fms   (%.2f inner solves/call)\n", t_F*1000, n_solves_F/30)
@printf("  composite_gradient_at alone, UNSHARED base (current): %.4fms   (%.2f inner solves/call)\n", t_G_unshared*1000, n_solves_G_unshared/30)
@printf("  cb_F!+cb_G! SEQUENCE, UNSHARED (as currently wired):  %.4fms   (%.2f inner solves/call)\n", t_sequence_unshared*1000, n_solves_sequence_unshared/30)
@printf("  cb_F!+cb_G! SEQUENCE, SHARED base (the fix):          %.4fms   (%.2f inner solves/call)\n", t_sequence_shared*1000, n_solves_sequence_shared/30)
@printf("  Savings from sharing: %.4fms (%.1f%%), %.2f fewer inner solves/call\n",
    (t_sequence_unshared - t_sequence_shared)*1000,
    100*(t_sequence_unshared - t_sequence_shared)/t_sequence_unshared,
    n_solves_sequence_unshared/30 - n_solves_sequence_shared/30)

# ---- equivalence check: shared-base gradient MUST equal unshared-base gradient at the same x ----
g_unshared, _ = composite_gradient_at(xf, ctx, pe)
g_shared, _ = composite_gradient_at(xf, ctx, pe; base = base0)
maxdiff = maximum(abs.(g_unshared .- g_shared))
println("\n  EQUIVALENCE: max|g_unshared - g_shared| = ", maxdiff, "  (", maxdiff < 1e-10 ? "PASS -- sharing the base state changes NOTHING mathematically" : "FAIL", ")")

open(joinpath(OUTDIR, "part2_redundancy_audit.txt"), "w") do io
    println(io, "eval_F alone (warmed median ms): ", t_F*1000, "  inner_solves/call: ", n_solves_F/30)
    println(io, "composite_gradient_at alone, UNSHARED (ms): ", t_G_unshared*1000, "  inner_solves/call: ", n_solves_G_unshared/30)
    println(io, "F+G sequence UNSHARED (ms): ", t_sequence_unshared*1000, "  inner_solves/call: ", n_solves_sequence_unshared/30)
    println(io, "F+G sequence SHARED (ms): ", t_sequence_shared*1000, "  inner_solves/call: ", n_solves_sequence_shared/30)
    println(io, "savings_ms: ", (t_sequence_unshared - t_sequence_shared)*1000)
    println(io, "savings_pct: ", 100*(t_sequence_unshared - t_sequence_shared)/t_sequence_unshared)
    println(io, "max_abs_gradient_diff_shared_vs_unshared: ", maxdiff)
end
println("\nWrote all Priority 1A artifacts to ", OUTDIR)
