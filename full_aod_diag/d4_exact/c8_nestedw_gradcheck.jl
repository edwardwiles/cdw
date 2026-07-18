# ============================================================================
# Continuation 8, Section 9 (nested-W). Fast-vs-slow gradient cross-check at
# a SAMPLE of (W, candidate) points -- NOT all 6, per the task's explicit
# "not all of them, that would be expensive" instruction.
#
# "Fast" = the SAME reduced-w-space gradient the reopt driver actually uses
# (composite_gradient_at for upper/lfix_composite, composite_gradient_at_fast
# for lower/lfix_composite_fast -- run_d4_optimized_fd.jl's own dispatch,
# reproduced here).
# "Slow" = central finite-difference of Delta_of_w (the optimized-value
# object, fully re-solving the inner CC dual at every probe) -- reusing
# run_d4_optimized_fd.jl's eval_grad_central_fd pattern (one-sided fallback
# on a failed probe), evaluated at TWO bandwidths per point: the established
# FIXED_H=0.01 default, and the W-dependent adaptive h from h_sweep.jl's
# adaptive_h_candidate (the SAME per-(W,point) value already computed and
# logged in c8_nestedw_run_grid.jl's `h_diagnostic` column -- recomputed here
# identically, not re-derived, for exact traceability). Reporting both lets a
# reader see whether the cosine verdict is bandwidth-sensitive, directly
# addressing the task's "don't reuse a W=8000-tuned h at W=80000 unchecked".
#
# Blockwise decomposition (gamma-component vs the D^2-1 z_free "A-block")
# reuses phase6_blockwise_gradient_check.jl's own finding/convention: a
# full-vector cosine can look good purely because gamma dominates the norm,
# concealing a bad A-block -- report both.
# ============================================================================
include(joinpath(@__DIR__, "c8_nestedw_context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "h_sweep.jl"))
using LinearAlgebra, Printf, Dates

const COMMIT = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c8_nestedw_gradcheck_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "c8_nestedw_gradcheck.csv")
println(">>> c8_nestedw_gradcheck.jl  commit=$COMMIT  outdir=$OUTDIR"); flush(stdout)

const w_upper = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
const w_lower = [0.9973649883022927, 0.4191333995096165, 0.3278704261228879, 0.34822377242086583, 0.3266848028818515, 1.1414170966875377, 1.299758470774316, 1.005176411943463, 1.0769193031619044, 0.8872003122885062, 0.7853545670122154, 0.8376830845864721, 0.7594387919364166, 1.7457822520932191, 1.4007037442409944, 1.5130549509169442]

# sample points: 3 of the possible 6 (W,candidate) combos -- both directions covered, and the largest
# W (where trustworthiness matters most / where composite_gradient_at_fast's O(1)-per-probe internals
# have never been exercised at this scale before) covered for BOTH candidates.
const SAMPLE_POINTS = [
    (label = "upper", W = 8000, w0 = w_upper, find_smallest = true, gradient_method = :lfix_composite),
    (label = "upper", W = 80000, w0 = w_upper, find_smallest = true, gradient_method = :lfix_composite),
    (label = "lower", W = 80000, w0 = w_lower, find_smallest = false, gradient_method = :lfix_composite_fast),
]

const FIXED_H = 0.01
cossim(a, b) = dot(a, b) / max(norm(a) * norm(b), 1e-300)

function eval_grad_central_fd(Delta_of_w::Function, w::Vector{Float64}, h::Float64)
    n = length(w); g = zeros(n)
    for i in 1:n
        wp = copy(w); wp[i] += h; wm = copy(w); wm[i] -= h
        Δp = Delta_of_w(wp); Δm = Delta_of_w(wm)
        if isfinite(Δp) && isfinite(Δm)
            g[i] = (Δp - Δm) / (2h)
        elseif isfinite(Δp)
            Δ0 = Delta_of_w(w); g[i] = (Δp - Δ0) / h
        elseif isfinite(Δm)
            Δ0 = Delta_of_w(w); g[i] = (Δ0 - Δm) / h
        else
            g[i] = 0.0
        end
    end
    return g
end

CSV_COLS = ["candidate", "W", "h_fixed", "h_adaptive", "full_cos_h001", "full_cos_hadaptive",
            "gamma_relerr_h001", "gamma_relerr_hadaptive", "Ablock_cos_h001", "Ablock_cos_hadaptive",
            "Ablock_sign_agree_h001", "Ablock_sign_agree_hadaptive", "wall_fast_s", "wall_slow_h001_s", "wall_slow_hadaptive_s"]
open(CSV_PATH, "w") do io
    println(io, join(CSV_COLS, ","))
end

for pt in SAMPLE_POINTS
    println("="^78); println("POINT: candidate=$(pt.label) W=$(pt.W)"); println("="^78); flush(stdout)
    ctx = build_nested_ctx(pt.W; find_smallest = pt.find_smallest)
    pe = build_pivot_elimination(ctx)
    D2 = length(pt.w0)
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    Delta_of_w(w) = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = true).Delta_dual

    # "fast" gradient -- same dispatch as the reopt driver, no shared base state (standalone probe)
    xf0 = x_free_from_w(pt.w0)
    t0 = time()
    g_fast = if pt.gradient_method == :lfix_composite
        g, _ = composite_gradient_at(xf0, ctx, pe); g
    else
        g, _ = composite_gradient_at_fast(xf0, ctx, pe; base = nothing, threaded = true, h_mode = :adaptive); g
    end
    wall_fast = time() - t0

    # v_free here must be length(ctx.free_idx) == D2+1 (index 1 = gamma'_focal, skipped internally by
    # v_free_to_Amat) -- NOT the D2-length reduced-w vector.
    n_free = length(ctx.free_idx)
    v = zeros(n_free); v[2] = 1.0
    h_adapt = clamp(adaptive_h_candidate(xf0, v, ctx), 1e-4, 0.05)

    t0 = time(); g_slow_001 = eval_grad_central_fd(Delta_of_w, pt.w0, FIXED_H); wall_001 = time() - t0
    t0 = time(); g_slow_adapt = eval_grad_central_fd(Delta_of_w, pt.w0, h_adapt); wall_adapt = time() - t0

    GAMMA_IDX = 1; A_IDX = 2:D2
    function summarize(g, g_ref)
        full_cos = cossim(g, g_ref)
        gamma_relerr = abs(g[GAMMA_IDX] - g_ref[GAMMA_IDX]) / max(abs(g_ref[GAMMA_IDX]), 1e-300)
        A_cos = cossim(g[A_IDX], g_ref[A_IDX])
        A_sign = count(sign.(g[A_IDX]) .== sign.(g_ref[A_IDX])) / (D2 - 1)
        return (full_cos = full_cos, gamma_relerr = gamma_relerr, A_cos = A_cos, A_sign = A_sign)
    end
    s001 = summarize(g_fast, g_slow_001)
    sadapt = summarize(g_fast, g_slow_adapt)

    @printf("  h_fixed=%.4f h_adaptive=%.5f\n", FIXED_H, h_adapt)
    @printf("  [vs h=0.01     ] full_cos=%.4f gamma_relerr=%.4f A_cos=%.4f A_sign_agree=%.3f\n", s001.full_cos, s001.gamma_relerr, s001.A_cos, s001.A_sign)
    @printf("  [vs h=adaptive ] full_cos=%.4f gamma_relerr=%.4f A_cos=%.4f A_sign_agree=%.3f\n", sadapt.full_cos, sadapt.gamma_relerr, sadapt.A_cos, sadapt.A_sign)
    @printf("  wall: fast=%.2fs slow(h=0.01)=%.2fs slow(h=adaptive)=%.2fs\n", wall_fast, wall_001, wall_adapt)
    flush(stdout)

    open(CSV_PATH, "a") do io
        println(io, join([pt.label, pt.W, FIXED_H, h_adapt, s001.full_cos, sadapt.full_cos,
            s001.gamma_relerr, sadapt.gamma_relerr, s001.A_cos, sadapt.A_cos, s001.A_sign, sadapt.A_sign,
            wall_fast, wall_001, wall_adapt], ","))
    end
end

println("\nWrote ", CSV_PATH)
println("DONE")
