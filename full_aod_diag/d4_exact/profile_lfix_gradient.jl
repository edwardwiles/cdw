# ============================================================================
# Phase 1 (continuation 3): L_fix gradient component profile, per explicit
# user request. Profiles the EXISTING, already-validated three_way_derivatives.jl
# machinery (solve_base_state, fixed_dual_L) via an additive, @prof-wrapped
# MIRROR (fixed_dual_L_profiled) -- equivalence-checked against the original
# before being trusted for timing, same discipline as oracle_fast.jl. This is
# the FULL-REBUILD baseline (one full obj.moments! call per perturbation) --
# Phase 2's block-local/incremental evaluators will be benchmarked against
# this file's own numbers, not re-measured from scratch.
#
# Verifies directly (CS.INNER_SOLVE_COUNT[] diffs, not inferred): a full
# 16-dim central-FD L_fix gradient uses EXACTLY ONE real inner CC dual solve
# (the base_state solve) and ZERO at every one of the 32 perturbation evals.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
using LinearAlgebra: dot
using Printf
using Random

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_lfix_gradient")
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

"""
    fixed_dual_L_profiled(x_free, ctx, base) -> Float64

Additive @prof-wrapped mirror of three_way_derivatives.jl::fixed_dual_L.
"""
function fixed_dual_L_profiled(x_free::AbstractVector, ctx, base::BaseDualState)
    obj = ctx.obj
    θ_full = @prof "lfix_reconstruct" CS.reconstruct_full(x_free, ctx.m)
    W = size(obj.U, 1); d = obj.d
    K = zeros(eltype(θ_full), W); G = zeros(eltype(θ_full), W, d)
    @prof "lfix_perturbation_moments" begin
        obj.moments!(K, G, θ_full, obj.U, obj)
    end
    oci = obj.outer_constr_index
    q = @prof "lfix_q_assembly" begin
        [-base.ζstar - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:W]
    end
    Psi_q = similar(q)
    @prof "lfix_scalar_eval" begin
        CS.Psi!(Psi_q, q)
    end
    return -(sum(Psi_q) / W + base.ζstar)
end

# ---- equivalence check vs the trusted original, before trusting this for timing ----
w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
xf0 = x_free_from_w(w_up40)
base0 = solve_base_state(xf0, ctx)
println("="^78); println("EQUIVALENCE CHECK: fixed_dual_L_profiled vs fixed_dual_L"); println("="^78)
eq_ok = true
for trial in 1:10
    w = w_up40 .+ 0.005 .* randn(MersenneTwister(1000+trial), length(w_up40))
    xf = x_free_from_w(w)
    va = fixed_dual_L(xf, ctx, base0)
    vb = fixed_dual_L_profiled(xf, ctx, base0)
    d = abs(va - vb)
    global eq_ok &= d < 1e-12
    println("  trial $trial: fixed_dual_L=$va  profiled=$vb  diff=$d  ", d < 1e-12 ? "PASS" : "FAIL")
end
eq_ok || error("profile_lfix_gradient.jl: fixed_dual_L_profiled does not match fixed_dual_L")
println("Equivalence: ALL PASS\n")
prof_reset!()

# ---- warm-up (JIT) ----
_ = solve_base_state(xf0, ctx)
_ = fixed_dual_L_profiled(xf0 .+ 1e-6, ctx, base0)
prof_reset!()

# ---- N gradient evaluations: central FD, 16-dim reduced coordinate, h=0.01 ----
const N_GRAD = 20
const HFD = 0.01
solve_counts_per_grad = Int[]
total_times = Float64[]
for g in 1:N_GRAD
    t0 = time_ns()
    solves_before = CS.INNER_SOLVE_COUNT[]

    xb = x_free_from_w(w_up40)
    base = @prof "lfix_base_solve" solve_base_state(xb, ctx)
    @prof "lfix_base_state_setup" begin
        # base already extracted (zeta*, lambda*, m*) inside solve_base_state; this label times the
        # NEGLIGIBLE additional copy/wrap cost a live gradient driver would add on top (there is none
        # here beyond what solve_base_state itself does -- reported as ~0, not silently omitted).
        base
    end

    grad = zeros(length(w_up40))
    for i in 1:length(w_up40)
        wp = copy(w_up40); wp[i] += HFD
        wm = copy(w_up40); wm[i] -= HFD
        Lp = fixed_dual_L_profiled(x_free_from_w(wp), ctx, base)
        Lm = fixed_dual_L_profiled(x_free_from_w(wm), ctx, base)
        @prof "lfix_fd_assembly" begin
            grad[i] = (Lp - Lm) / (2 * HFD)
        end
    end
    push!(total_times, (time_ns() - t0) / 1e9)
    push!(solve_counts_per_grad, CS.INNER_SOLVE_COUNT[] - solves_before)
end

rows = prof_summary()
function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted)/n
    σ = n > 1 ? sqrt(sum((t-μ)^2 for t in sorted)/(n-1)) : 0.0
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            p90_s = sorted[clamp(ceil(Int, 0.9*n), 1, n)], p95_s = sorted[clamp(ceil(Int, 0.95*n), 1, n)],
            mean_s = μ, std_s = σ, mean_alloc_bytes = NaN, total_alloc_bytes = NaN, mean_gc_s = NaN, total_gc_s = NaN)
end
push!(rows, stats_row("TOTAL_lfix_gradient_16dim", total_times))
write_csv_rows(joinpath(OUTDIR, "profile_lfix_gradient.csv"), rows)

total_median = (filter(r -> r.label == "TOTAL_lfix_gradient_16dim", rows)[1]).median_s
println("="^78); println("L_fix 16-DIM GRADIENT COMPONENT BREAKDOWN (median per gradient call = $(round(total_median,digits=4))s, N=$N_GRAD)"); println("="^78)
for lbl in ("lfix_base_solve", "lfix_base_state_setup", "lfix_reconstruct", "lfix_perturbation_moments", "lfix_q_assembly", "lfix_scalar_eval", "lfix_fd_assembly")
    r = filter(r -> r.label == lbl, rows)
    isempty(r) && continue
    tot = r[1].mean_s * r[1].n   # total across all N_GRAD*calls-per-grad occurrences
    n_per_grad = r[1].n / N_GRAD
    @printf("  %-28s total=%.4fs  n_calls_total=%d (%.1f/gradient)  mean_per_call=%.6fs  pct_of_grad_wall=%.1f%%\n",
            lbl, tot, r[1].n, n_per_grad, r[1].mean_s, 100*tot/(total_median*N_GRAD))
end

println("\nCS.INNER_SOLVE_COUNT[] diff per gradient call (expect EXACTLY 1: the base solve, ZERO at every perturbation):")
println("  values across $N_GRAD gradients: ", solve_counts_per_grad)
all_exactly_one = all(==(1), solve_counts_per_grad)
println("  ALL EXACTLY 1: ", all_exactly_one ? "PASS -- confirms L_fix is genuinely inner-solve-free at every perturbation" : "FAIL")

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "N_GRAD=", N_GRAD, " HFD=", HFD, " reduced_dim=", length(w_up40))
    println(io, "TOTAL_lfix_gradient_16dim median: ", total_median, "s")
    println(io, "inner_solve_count diff per gradient: ", solve_counts_per_grad)
    println(io, "all exactly 1: ", all_exactly_one)
    for lbl in ("lfix_base_solve", "lfix_base_state_setup", "lfix_reconstruct", "lfix_perturbation_moments", "lfix_q_assembly", "lfix_scalar_eval", "lfix_fd_assembly")
        r = filter(r -> r.label == lbl, rows)
        isempty(r) && continue
        println(io, lbl, ": n=", r[1].n, " mean_s=", r[1].mean_s, " total_s=", r[1].mean_s*r[1].n)
    end
end
println("\nWrote ", OUTDIR)
