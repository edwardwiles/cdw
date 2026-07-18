# ============================================================================
# Continuation-session Phase D: can a cheap full-A gradient replace
# optimized-value finite differences?
#
# Benchmarks, at multiple outer points, four gradient methods (all operating
# in the RAW x_free (17-dim: gamma_focal_prime + all 16 A_od entries) space
# that three_way_derivatives.jl/derivative_methods.jl already use -- gravity
# elimination is NOT needed here since Delta/Q_adj/L_fix are all well-defined
# regardless of whether gravity happens to hold at a probe point):
#   1. Method A: hard pathwise AD (ForwardDiff.gradient on frozen_adjoint_Q)
#      -- ONE cheap call, known-biased (misses the winner-boundary term).
#   2. Q_adj FD: central FD (h=0.01) on frozen_adjoint_Q (dual FROZEN, G(x)
#      fully recomputed, Psi never re-evaluated -- linear in G).
#   3. L_fix FD: central FD (h=0.01) on fixed_dual_L (dual FROZEN, G(x) fully
#      recomputed, Psi FULLY re-evaluated -- matches production sequential
#      method's own fixed_dual_fd_full correction, sequential_methodology.tex
#      sec 10.1, h=0.1 there vs h=0.01 here -- see the h-choice discussion below).
#   4. Delta FD (GROUND TRUTH): central FD (h=0.01) on optimized_Delta -- the
#      fully re-solved inner CC dual at every probe (what Phase A/the existing
#      run_d4_optimized_fd.jl driver already uses).
# Method E (temperature-grid smoothed scalar) is DELIBERATELY EXCLUDED here:
# this session found (see docs/fullA_d4_resume_audit.md sec 6.4 and the
# smoothing_check.csv commit) that the smoothed-moments path
# (smoothed_factual_G/smoothed_frozen_adjoint_Q) has an unresolved
# call-history-dependent non-determinism bug -- including it in a benchmark
# whose whole point is trustworthy comparison would be actively misleading
# until that bug is fixed. Flagged as a gap, not silently dropped.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "derivative_methods.jl"))
using Random, LinearAlgebra, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phaseD_gradient_benchmark")
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
w_to_xfree(w) = (z = pivot_expand(w[2:end], pe); vcat(w[1], vec(exp.(z))))

# ---- required points ----
x0_calib = CS.pack_free(ctx.θ0_up, ctx.m)
w_maxit15 = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069,
    0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011,
    1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663,
    0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]
w_maxit40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
    0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
    1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
    0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_lower_stalled = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916,
    -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819,
    0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236,
    0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

points = [
    ("calibration", x0_calib),
    ("upper_maxit15", w_to_xfree(w_maxit15)),
    ("upper_maxit40", w_to_xfree(w_maxit40)),
    ("lower_stalled", w_to_xfree(w_lower_stalled)),
]

const H = 0.01
rng_global = MersenneTwister(20260718)

function central_fd_grad(f, x0::Vector{Float64}, h::Float64)
    n = length(x0); g = zeros(n); ncalls = 0
    for i in 1:n
        xp = copy(x0); xp[i] += h
        xm = copy(x0); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h); ncalls += 2
    end
    return g, ncalls
end

rows = NamedTuple[]
for (label, x_free) in points
    println("="^78); println("POINT: $label"); println("="^78); flush(stdout)
    base = solve_base_state(x_free, ctx)

    t0 = time(); g_A = method_A_pathwise_ad(x_free, ctx, base); t_A = time() - t0; n_A = 1

    t0 = time(); g_Qadj, n_Qadj = central_fd_grad(x -> frozen_adjoint_Q(x, ctx, base), x_free, H); t_Qadj = time() - t0

    t0 = time(); g_Lfix, n_Lfix = central_fd_grad(x -> fixed_dual_L(x, ctx, base), x_free, H); t_Lfix = time() - t0

    t0 = time(); g_Delta, n_Delta = central_fd_grad(x -> optimized_Delta(x, ctx; warm = true), x_free, H); t_Delta = time() - t0

    for (mname, g, ncall, tsec) in (("A_pathwise_AD", g_A, n_A, t_A), ("Q_adj_FD", g_Qadj, n_Qadj, t_Qadj), ("L_fix_FD", g_Lfix, n_Lfix, t_Lfix))
        cosim = dot(g, g_Delta) / max(norm(g) * norm(g_Delta), 1e-300)
        normratio = norm(g) / max(norm(g_Delta), 1e-300)
        sign_agree = count(sign.(g) .== sign.(g_Delta)) / length(g)
        # random-directional prediction error (5 fixed directions, shared across methods per point)
        rng = MersenneTwister(hash((label,)))
        dirs = [normalize(randn(rng, length(x_free))) for _ in 1:5]
        pred_err = [dot(g, d) - dot(g_Delta, d) for d in dirs]
        mean_abs_pred_err = sum(abs.(pred_err)) / length(pred_err)
        push!(rows, (point = label, method = mname, cosine_vs_Delta = cosim, norm_ratio_vs_Delta = normratio,
            sign_agreement_vs_Delta = sign_agree, mean_abs_directional_pred_err = mean_abs_pred_err,
            n_inner_solves = ncall, wall_seconds = tsec, norm_grad = norm(g)))
        @printf("  [%-15s] cos=%.4f normratio=%.3f signagree=%.3f meanprederr=%.4f n_evals=%d wall=%.3fs\n",
            mname, cosim, normratio, sign_agree, mean_abs_pred_err, ncall, tsec)
    end
    push!(rows, (point = label, method = "Delta_FD_GROUNDTRUTH", cosine_vs_Delta = 1.0, norm_ratio_vs_Delta = 1.0,
        sign_agreement_vs_Delta = 1.0, mean_abs_directional_pred_err = 0.0,
        n_inner_solves = n_Delta, wall_seconds = t_Delta, norm_grad = norm(g_Delta)))
    @printf("  [%-15s] (reference) n_evals=%d wall=%.3fs\n", "Delta_FD", n_Delta, t_Delta)
    flush(stdout)
end

open(joinpath(OUTDIR, "phaseD_comparison_table.csv"), "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join((r[c] for c in cols), ","))
    end
end

println("\n" * "="^78); println("SUMMARY"); println("="^78)
for m in ("A_pathwise_AD", "Q_adj_FD", "L_fix_FD")
    mrows = filter(r -> r.method == m, rows)
    mean_cos = sum(r.cosine_vs_Delta for r in mrows) / length(mrows)
    mean_cost_ratio = sum(r.n_inner_solves for r in mrows) / sum(r.n_inner_solves for r in filter(r -> r.method == "Delta_FD_GROUNDTRUTH", rows))
    println("$m: mean cosine-vs-Delta across $(length(mrows)) points = $(round(mean_cos,digits=4)), total evals = $(sum(r.n_inner_solves for r in mrows)) vs Delta_FD's $(sum(r.n_inner_solves for r in filter(r->r.method=="Delta_FD_GROUNDTRUTH",rows)))")
end
println("\nWrote ", joinpath(OUTDIR, "phaseD_comparison_table.csv"))
println("\nNOTE: Method E (temperature-grid smoothed scalar) excluded -- see header comment and")
println("docs/fullA_d4_resume_audit.md sec 6.4 (unresolved non-determinism bug in that path).")
