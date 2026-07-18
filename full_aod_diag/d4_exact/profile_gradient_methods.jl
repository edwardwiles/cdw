# ============================================================================
# Phase 1A/B: resolve the Phase D "n_inner_solves=34 even for methods that
# don't re-solve the inner problem" question with AUTHORITATIVE counters
# (CS.INNER_SOLVE_COUNT[] before/after diffs), not FD-probe counting.
#
# FINDING (confirmed below, not assumed): Phase D's `central_fd_grad` counted
# `ncalls += 2` per coordinate regardless of what the wrapped function does --
# for Q_adj_FD/L_fix_FD (which call `frozen_adjoint_Q`/`fixed_dual_L`, both
# operating on FROZEN duals, never touching CS.inner_loop_internal) this
# counted FUNCTION EVALUATIONS, not inner solves. The true n_inner_solves for
# those two methods is 0 by construction -- they only ever call obj.moments!.
# Only optimized_Delta (used by "Delta_FD") and method_A_pathwise_ad's
# ForwardDiff pass (which still calls frozen_adjoint_Q, so also 0 real inner
# solves -- ForwardDiff differentiates through the FROZEN-dual construction,
# never re-solving) actually touch the inner solver, and Method A does so
# ZERO times (it's exactly 1 ForwardDiff.gradient call, no inner solve at
# all). This script measures wall time AND true inner-solve/moments-eval
# counts side by side so the CORRECTED metric replaces the mislabeled one.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_profiled.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "derivative_methods.jl"))
using Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_gradient_methods")
mkpath(OUTDIR)
const H = 0.01
const N_REPS = 10

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
w_to_xfree(w) = (z = pivot_expand(w[2:end], pe); vcat(w[1], vec(exp.(z))))

x0_calib = CS.pack_free(ctx.θ0_up, ctx.m)
w_maxit40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
    0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
    1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
    0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]

points = [("calibration", x0_calib), ("upper_maxit40", w_to_xfree(w_maxit40))]

"Count real inner solves + moments! calls (via a counting wrapper on ctx.obj.moments!) for a full gradient computation."
function count_and_time(f_gradient::Function, ctx; n_reps::Int = N_REPS)
    times = Float64[]
    solves_list = Int[]; moments_calls_list = Int[]
    # count obj.moments! calls by wrapping it (restored after)
    orig_moments = ctx.obj.moments!
    n_moments_calls = Ref(0)
    ctx.obj.moments! = (K, G, θ, U, o) -> (n_moments_calls[] += 1; orig_moments(K, G, θ, U, o))
    try
        for rep in 1:n_reps
            solves0 = CS.INNER_SOLVE_COUNT[]
            n_moments_calls[] = 0
            t0 = time_ns()
            f_gradient()
            elapsed = (time_ns() - t0) / 1e9
            push!(times, elapsed)
            push!(solves_list, CS.INNER_SOLVE_COUNT[] - solves0)
            push!(moments_calls_list, n_moments_calls[])
        end
    finally
        ctx.obj.moments! = orig_moments
    end
    return times, solves_list, moments_calls_list
end

function summarize(times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted)/n
    return (median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end], mean_s = μ)
end

rows = NamedTuple[]
for (label, x_free) in points
    println("="^78); println("POINT: $label"); println("="^78); flush(stdout)
    base = solve_base_state(x_free, ctx)

    methods = [
        ("A_pathwise_AD", () -> method_A_pathwise_ad(x_free, ctx, base)),
        ("Q_adj_FD", () -> begin
            n = length(x_free); g = zeros(n)
            for i in 1:n
                xp = copy(x_free); xp[i] += H; xm = copy(x_free); xm[i] -= H
                g[i] = (frozen_adjoint_Q(xp, ctx, base) - frozen_adjoint_Q(xm, ctx, base)) / (2H)
            end
            g
        end),
        ("L_fix_FD", () -> begin
            n = length(x_free); g = zeros(n)
            for i in 1:n
                xp = copy(x_free); xp[i] += H; xm = copy(x_free); xm[i] -= H
                g[i] = (fixed_dual_L(xp, ctx, base) - fixed_dual_L(xm, ctx, base)) / (2H)
            end
            g
        end),
        ("Delta_FD_optimized_value", () -> begin
            n = length(x_free); g = zeros(n)
            for i in 1:n
                xp = copy(x_free); xp[i] += H; xm = copy(x_free); xm[i] -= H
                g[i] = (optimized_Delta(xp, ctx; warm=true) - optimized_Delta(xm, ctx; warm=true)) / (2H)
            end
            g
        end),
    ]

    for (mname, f) in methods
        times, solves, mcalls = count_and_time(f, ctx)
        s = summarize(times)
        push!(rows, (point = label, method = mname, median_s = s.median_s, min_s = s.min_s, mean_s = s.mean_s,
            n_inner_solves_per_call = solves[1], n_moments_evals_per_call = mcalls[1],
            n_reps = length(times)))
        @printf("  [%-25s] median=%.4fs  TRUE n_inner_solves=%d  n_moments_evals=%d  (n=%d coords: %d)\n",
            mname, s.median_s, solves[1], mcalls[1], length(times), length(x_free))
    end
    flush(stdout)
end

write_csv_rows(joinpath(OUTDIR, "profile_gradient_methods.csv"), rows)

println("\n" * "="^78); println("CORRECTED n_inner_solves METRIC (vs Phase D's mislabeled n_inner_solves=34)"); println("="^78)
for r in rows
    println("  $(r.point) / $(r.method): TRUE n_inner_solves=$(r.n_inner_solves_per_call) (Phase D's table called this 'n_inner_solves' but it was counting probe evaluations, $(r.n_moments_evals_per_call) of which were obj.moments! calls)")
end
println("\nWrote ", joinpath(OUTDIR, "profile_gradient_methods.csv"))
