# ============================================================================
# Live-path equivalence for enable_autarky_cf! (autarky_cf.jl): compares a
# reference ctx (generic hFunctionCounter! CF) against an autarky-CF-wired ctx,
# field-for-field through the FULL evaluate_fullA (optimized dual variables,
# primal & dual divergence, recovered weights, KKT residual, gravity, winners),
# AND a divergence-gradient trajectory check, at calibration + both candidates +
# random feasible perturbations. Because the specialized path only swaps the CF
# column for a BIT-IDENTICAL implementation, every downstream result must be
# bit-identical.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
include(joinpath(@__DIR__, "autarky_cf.jl"))
using Random, Printf

ctx_ref  = d4_exact_setup(find_smallest = true)
ctx_spec = d4_exact_setup(find_smallest = true)
pc = enable_autarky_cf!(ctx_spec)   # returns nothing; wiring done in place

pe = build_pivot_elimination(ctx_ref)
D = ctx_ref.D; D2 = D^2
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

points = Dict(
    "calibration" => vcat(ctx_ref.θ0_up[3+D], pivot_reduce(zeros(D, D), pe)),
    "upper_lfixcomposite_sr1_60s" => [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845],
    "lower_stalled" => [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385],
)
rng = MersenneTwister(20260718)
for i in 1:5
    points["random_$i"] = points["upper_lfixcomposite_sr1_60s"] .+ 0.05 .* randn(rng, D2)
end

fields_scalar = (:gamma_focal_prime, :K_hard, :Delta_dual, :Delta_primal, :Delta_minus_delta,
                 :gravity_value, :gravity_R_mean, :gravity_R_sum, :gravity_R_beta, :zeta,
                 :m_mean, :m_min, :m_max, :mean_m_resid, :max_abs_moment_kkt_resid,
                 :max_abs_moment_resid, :primal_dual_gap, :weight_norm_resid)
fields_int = (:inner_status, :winner_hash)

function compare(rr, rc, label; tol = 0.0)   # tol=0.0: require BIT-IDENTICAL
    ok = true; worst = 0.0; worst_f = :none
    for f in fields_scalar
        a = getfield(rr, f); b = getfield(rc, f)
        (a === nothing && b === nothing) && continue
        (isa(a,Float64) && isnan(a) && isnan(b)) && continue
        d = abs(a - b); (d > worst) && (worst = d; worst_f = f)
        (d > tol) && (ok = false)
    end
    for f in fields_int
        getfield(rr, f) == getfield(rc, f) || (ok = false)
    end
    for f in (:lambda, :moment_resid, :logA)
        a = vec(collect(getfield(rr, f))); b = vec(collect(getfield(rc, f)))
        if length(a) == length(b) && !isempty(a)
            d = maximum(abs.(a .- b)); (d > worst) && (worst = d; worst_f = f)
            (d > tol) && (ok = false)
        elseif length(a) != length(b); ok = false; end
    end
    @printf("  %-30s worst|Δ|=%.3e (%s)  %s\n", label, worst, worst_f, ok ? "PASS" : "FAIL")
    return ok
end

all_pass = true
println("="^78)
println("LIVE-PATH EQUIVALENCE: evaluate_fullA, generic-CF ctx vs autarky-CF-wired ctx")
println("(tol = 0.0, i.e. requiring BIT-IDENTICAL)")
println("="^78)
for (label, w) in points
    xf = x_free_from_w(w)
    rr = evaluate_fullA(xf, ctx_ref;  cache = nothing, warm = false)
    rc = evaluate_fullA(xf, ctx_spec; cache = nothing, warm = false)
    global all_pass &= compare(rr, rc, label)
end

# ---- divergence-gradient trajectory check (central FD of Delta_dual wrt x_free) ----
println("\n" * "="^78)
println("DIVERGENCE-GRADIENT equivalence (central FD, h=1e-6), a few directions")
println("="^78)
function grad_fd(xf, ctx; h = 1e-6)
    n = length(xf); g = zeros(n)
    for k in 1:n
        xp = copy(xf); xp[k] += h; xm = copy(xf); xm[k] -= h
        fp = evaluate_fullA(xp, ctx; cache = nothing, warm = false).Delta_dual
        fm = evaluate_fullA(xm, ctx; cache = nothing, warm = false).Delta_dual
        g[k] = (fp - fm) / (2h)
    end
    return g
end
for label in ("calibration", "upper_lfixcomposite_sr1_60s")
    xf = x_free_from_w(points[label])
    gr = grad_fd(xf, ctx_ref); gc = grad_fd(xf, ctx_spec)
    # NaN-aware: components where BOTH are NaN (FD stepped into infeasible region
    # identically for both ctx) count as equal; finite components must match bit-for-bit.
    diffs = [(isnan(a) && isnan(b)) ? 0.0 : abs(a - b) for (a,b) in zip(gr, gc)]
    n_nan = count(k -> isnan(gr[k]) && isnan(gc[k]), 1:length(gr))
    d = maximum(diffs)
    pass = d == 0.0
    @printf("      (%d/%d components NaN in BOTH -- FD stepped infeasible identically)\n", n_nan, length(gr))
    global all_pass &= pass
    @printf("  %-30s max|Δ∇|=%.3e  %s\n", label, d, pass ? "PASS" : "FAIL")
end

println("\n" * "="^78)
println(all_pass ? "ALL AUTARKY-CF LIVE-PATH EQUIVALENCE CHECKS PASSED (bit-identical downstream)" :
                   "SOME CHECKS FAILED -- do not adopt the specialized autarky CF path")
println("="^78)
all_pass || error("verify_autarky_cf_wiring.jl: equivalence checks failed")
