# ============================================================================
# Phase 1 equivalence test: evaluate_fullA_fast (oracle_fast.jl) must agree
# with evaluate_fullA (oracle.jl) field-for-field (moments-reuse, fast
# winners, reduced-allocation KKT residual are all REFORMULATIONS of the same
# computation, not approximations -- exact/near-machine-precision agreement
# expected, not "close enough"), across calibration, both upper candidates,
# the lower candidate, random feasible perturbations, and warm/cold/cache
# paths, before being trusted for either correctness or timing.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using Random

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

FIELDS_NUMERIC = (:gamma_focal_prime, :K_hard, :Delta_dual, :Delta_primal, :Delta_minus_delta,
    :gravity_raw, :gravity_value, :gravity_R_sum, :gravity_R_mean, :gravity_R_beta,
    :max_abs_moment_resid, :zeta, :m_mean, :m_min, :m_max, :weight_norm_resid,
    :mean_m_resid, :max_abs_moment_kkt_resid, :primal_dual_gap)
FIELDS_EXACT = (:inner_status, :winner_hash)

function compare(label, x_free; warm = true, cache_a = nothing, cache_b = nothing)
    ra = evaluate_fullA(x_free, ctx; cache = cache_a, warm = warm)
    rb, meta = evaluate_fullA_fast(x_free, ctx; cache = cache_b, warm = warm)
    ok = true
    maxdiff = 0.0
    for f in FIELDS_NUMERIC
        va = getfield(ra, f); vb = getfield(rb, f)
        if isnan(va) && isnan(vb)
            continue
        end
        d = abs(va - vb)
        maxdiff = max(maxdiff, d)
        if d > 1e-9
            println("  MISMATCH field=$f  oracle=$va  fast=$vb  diff=$d")
            ok = false
        end
    end
    for f in FIELDS_EXACT
        va = getfield(ra, f); vb = getfield(rb, f)
        if va != vb
            println("  MISMATCH (exact) field=$f  oracle=$va  fast=$vb")
            ok = false
        end
    end
    # vector fields
    if !isapprox(collect(ra.logA), collect(rb.logA); atol = 1e-9, nans = true)
        println("  MISMATCH field=logA"); ok = false
    end
    if !isapprox(ra.moment_resid, rb.moment_resid; atol = 1e-9)
        println("  MISMATCH field=moment_resid"); ok = false
    end
    if !isapprox(ra.lambda, rb.lambda; atol = 1e-9)
        println("  MISMATCH field=lambda"); ok = false
    end
    println(rpad(label, 34), " warm=", warm, "  maxdiff(numeric fields)=", maxdiff,
            "  n_fg_calls=", meta.n_fg_calls, " n_hess_calls=", meta.n_hess_calls,
            "  n_inner_solves(fast)=", meta.n_inner_solves, "  ", ok ? "PASS" : "FAIL")
    return ok
end

all_ok = true

println("="^78); println("TEST 1: candidate points, warm=true"); println("="^78)
θ_cal = copy(ctx.θ0_up)
global all_ok &= compare("calibration (full theta as x_free)", ctx.θ0_up[ctx.free_idx]; warm = true)

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
xf_up40 = x_free_from_w(w_up40)
global all_ok &= compare("upper_maxit40", xf_up40; warm = true)

w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]
xf_low = x_free_from_w(w_low)
global all_ok &= compare("lower_stalled", xf_low; warm = true)

println("\n" * "="^78); println("TEST 2: cold start (warm=false)"); println("="^78)
global all_ok &= compare("upper_maxit40 (cold)", xf_up40; warm = false)
global all_ok &= compare("lower_stalled (cold)", xf_low; warm = false)

println("\n" * "="^78); println("TEST 3: random feasible perturbations around upper_maxit40"); println("="^78)
rng = MersenneTwister(20260718)
for trial in 1:15
    w = w_up40 .+ 0.005 .* randn(rng, length(w_up40))
    xf = x_free_from_w(w)
    global all_ok &= compare("random perturbation $trial", xf; warm = true)
end

println("\n" * "="^78); println("TEST 4: cache semantics (cache hit still equivalent to oracle, no re-solve)"); println("="^78)
cache_b = oracle_cache_for(ctx)
r1, m1 = evaluate_fullA_fast(xf_up40, ctx; cache = cache_b, warm = true)
r2, m2 = evaluate_fullA_fast(xf_up40, ctx; cache = cache_b, warm = true)
cache_ok = r2.cache_hit && m2.n_inner_solves == 0 && r1.Delta_dual == r2.Delta_dual
println("first call cache_hit=", r1.cache_hit, " (expect false); second call cache_hit=", r2.cache_hit,
        " n_inner_solves=", m2.n_inner_solves, " (expect true/0)  ", cache_ok ? "PASS" : "FAIL")
global all_ok &= cache_ok

println("\n" * "="^78); println("TEST 5: n_fg_calls/n_hess_calls sanity (COLD start, needs real iterations)"); println("="^78)
# NOTE: a WARM-started call at an already-converged point can terminate in a single fg-callback
# with ZERO Hessian calls (KNITRO recognizes optimality immediately, never takes a Newton step) --
# observed directly, not a bug. Cold-starting forces genuine iteration so this sanity check is
# actually informative.
obj_reset = ctx.obj; obj_reset.x .= NaN
_, meta5 = evaluate_fullA_fast(xf_up40, ctx; cache = nothing, warm = false)
sane = meta5.n_fg_calls > 0 && meta5.n_hess_calls > 0 && meta5.n_hess_calls <= meta5.n_fg_calls
println("n_fg_calls=", meta5.n_fg_calls, " n_hess_calls=", meta5.n_hess_calls, "  ", sane ? "PASS" : "FAIL")
global all_ok &= sane

println("\n" * "="^78)
println(all_ok ? "ALL ORACLE_FAST EQUIVALENCE TESTS PASSED" : "SOME TESTS FAILED")
println("="^78)
all_ok || error("test_oracle_fast.jl: equivalence check failed")
