# Overnight task 2026-07-22, Section 2.3/2.4: D=4 correctness battery for the CM-aware C+
# gradient backend (lfix_cm_cplus.jl) against the TRUSTED CM reference path
# (lfix_cm_aware.jl::build_lfix_base_cache_cm / composite_gradient_at_fast_cm /
# cm_production_bundle.jl::cm_production_gradient), NOT merely against the unrestricted C+
# backend. See docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md for the decomposition this test
# verifies. Exercises the REAL production wiring: cm_production_gradient vs
# cm_production_gradient_cplus, both driven off the same pcx = build_cm_production_context(...)
# and the same archC_base_state (inner solve is orthogonal to the outer-gradient-backend
# question this test targets -- both backends receive the identical base).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
using Printf, LinearAlgebra, Random

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
const L = 10

x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(9013)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.03 .* randn(length(x_free_perturbed) - 1))

Random.seed!(90131)
x_free_perturbed2 = copy(x_free_calib)
x_free_perturbed2[2:end] .*= exp.(0.08 .* randn(length(x_free_perturbed2) - 1))   # larger perturbation, more likely to shift winners

test_points = [("calibration", x_free_calib), ("perturbed_0.03", x_free_perturbed), ("perturbed_0.08", x_free_perturbed2)]

for contrasts in (:anchored, :orthonormal)
    println()
    println("#"^100)
    println("# contrasts = :", contrasts, "  (L=$L, D=$D, W=$W)")
    println("#"^100)

    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts)
    ctx_cm = pcx.ctx_cm; aug = pcx.aug; bins = pcx.bins; cctx = pcx.cctx
    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, W)

    println("="^100)
    println("SECTION 1: base-point q0 agreement (CM-Reference builder vs CM-C+ builder), shared base")
    println("="^100)
    for (label, xf) in test_points
        local base = archC_base_state(xf, ctx_cm, cctx)
        cache_ref = build_lfix_base_cache_cm(xf, ctx_cm, base, ctx, aug, bins)
        # NOTE: validate_dense=true is deliberately NOT used here -- build_lfix_base_cache_C!'s
        # own internal self-check compares q0 against a full obj.moments! dot product over
        # 1:oci-1, which for a CM-augmented ctx_cm INCLUDES the CM tail columns; q0 at this point
        # intentionally omits them (that's exactly the gap cm_fixed_contribution below closes),
        # so validate_dense=true would spuriously "fail" on a correct cache -- same reason
        # build_lfix_base_cache_cm (lfix_cm_aware.jl) never passes validate_dense=true against
        # ctx_cm either (c13_validate_cm_aware_lfix.jl, the existing Reference CM test, never
        # does so). The REAL validation is the explicit q0-vs-Reference comparison below.
        cache_cp = build_lfix_base_cache_cm_C!(ws, xf, ctx_cm, base, ctx, aug, bins; validate_dense = false)
        maxerr_q0 = maximum(abs.(cache_ref.q0 .- cache_cp.q0))
        @printf "  [%s] max|q0_ref - q0_cplus| = %.3e\n" label maxerr_q0
        check("[$label/$contrasts] q0 agreement (CM-Reference vs CM-C+) < 1e-8", maxerr_q0 < 1e-8)

        lfix_ref0 = lfix_from_q(cache_ref.q0, cache_ref.ζstar)
        lfix_cp0 = lfix_from_q(cache_cp.q0, cache_cp.ζstar)
        check("[$label/$contrasts] lfix_from_q(q0) agreement (Reference vs C+) < 1e-10",
              abs(lfix_ref0 - lfix_cp0) < 1e-10)
    end

    println()
    println("="^100)
    println("SECTION 2: full production-wiring gradient agreement, cm_production_gradient vs cm_production_gradient_cplus (PRIMARY gate, matched h_mode=:fixed)")
    println("="^100)
    for (label, xf) in test_points
        local base = archC_base_state(xf, ctx_cm, cctx)
        g_ref, meta_ref = cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        g_cp, meta_cp = cm_production_gradient_cplus(xf, pcx, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)

        maxerr = maximum(abs.(g_ref .- g_cp))
        relerr = maxerr / max(maximum(abs.(g_ref)), 1e-12)
        cosang = dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
        nflip = count(sign.(g_ref[2:end]) .!= sign.(g_cp[2:end]))
        @printf "  [%s] gamma: ref=%.6e cplus=%.6e diff=%.3e | max|Δg|=%.3e relerr=%.3e cos=%.10f sign-mismatches=%d/%d\n" label g_ref[1] g_cp[1] abs(g_ref[1]-g_cp[1]) maxerr relerr cosang nflip D2-1
        check("[$label/$contrasts] gamma-component agreement < 1e-6", abs(g_ref[1] - g_cp[1]) < 1e-6)
        check("[$label/$contrasts] A-block max abs diff < 1e-6 (matched h=0.01)", maxerr < 1e-6)
        check("[$label/$contrasts] A-block cosine > 1 - 1e-8", cosang > 1 - 1e-8)
        check("[$label/$contrasts] zero sign mismatches", nflip == 0)
    end

    println()
    println("="^100)
    println("SECTION 3: bandwidth-selected (h_mode=:adaptive/:cached) agreement -- exercises select_bandwidth vs select_bandwidth_C independently, own bisections")
    println("="^100)
    for (label, xf) in test_points
        local base = archC_base_state(xf, ctx_cm, cctx)
        g_ref, _ = cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = false, h_mode = :adaptive)
        bwcache = Dict{Int,Float64}()
        g_cp, _ = cm_production_gradient_cplus(xf, pcx, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwcache)

        cosang = dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
        relnorm = norm(g_ref[2:end] .- g_cp[2:end]) / (norm(g_ref[2:end]) + 1e-300)
        @printf "  [%s] own-bandwidth cosine=%.8f  relative A-block norm diff=%.3e\n" label cosang relnorm
        check("[$label/$contrasts] own-bandwidth cosine > 1 - 1e-4 (looser: independent bisections may pick slightly different h)", cosang > 1 - 1e-4)
    end

    println()
    println("="^100)
    println("SECTION 4: independently reoptimized directional check -- fixed-dual secant (Reference AND C+) vs true reoptimized central-FD secant of Delta_dual, anchored to the SAME sign evalResult.jac uses")
    println("="^100)
    "True reoptimized Delta_dual at x_free (fresh archC solve+recompute every call -- the independent ground truth)."
    function delta_dual_true(x_free)
        base_here = archC_base_state(x_free, ctx_cm, cctx)
        obj = ctx_cm.obj
        ncon = obj.d - obj.outer_constr_index + 2
        cbuf = zeros(ncon)
        inner_x = vcat(base_here.ζstar, base_here.λstar)
        obj(inner_x, constr = @view(cbuf[1:ncon]))
        return cbuf[1] / 1e10
    end
    xf0 = x_free_calib
    local base0 = archC_base_state(xf0, ctx_cm, cctx)
    z0 = log.(reshape(xf0[2:end], D, D))
    w0 = vcat(xf0[1], pivot_reduce(z0, pe))
    cache_ref0 = build_lfix_base_cache_cm(xf0, ctx_cm, base0, ctx, aug, bins)
    cache_cp0 = build_lfix_base_cache_cm_C!(ws, xf0, ctx_cm, base0, ctx, aug, bins)
    for k in unique([2, 3, min(5, D2)])
        h = 0.02
        wp = copy(w0); wp[k] += h; wm = copy(w0); wm[k] -= h
        zp = pivot_expand(wp[2:end], pe); zm = pivot_expand(wm[2:end], pe)
        xfp = vcat(wp[1], vec(exp.(zp))); xfm = vcat(wm[1], vec(exp.(zm)))
        true_secant = -(delta_dual_true(xfp) - delta_dual_true(xfm)) / (2h)   # sign matches evalResult.jac / g[k] convention, see algebra trace Section 6

        Lp_ref = lfix_incremental_at(cache_ref0, ctx_cm, pe, w0, k, w0[k] + h)
        Lm_ref = lfix_incremental_at(cache_ref0, ctx_cm, pe, w0, k, w0[k] - h)
        secant_ref = (Lp_ref - Lm_ref) / (2h)

        Lp_cp = lfix_incremental_at_C(cache_cp0, ctx_cm, pe, w0, k, w0[k] + h)
        Lm_cp = lfix_incremental_at_C(cache_cp0, ctx_cm, pe, w0, k, w0[k] - h)
        secant_cp = (Lp_cp - Lm_cp) / (2h)

        @printf "  [k=%d] true_reoptimized=%.6e  fixed_dual(Reference)=%.6e  fixed_dual(C+)=%.6e\n" k true_secant secant_ref secant_cp
        check("[k=$k/$contrasts] C+ fixed-dual secant matches Reference's fixed-dual secant (not the true secant -- known AUD-05 approximation, both share it equally)",
              isapprox(secant_cp, secant_ref; atol = 1e-6, rtol = 1e-6))
    end
end

println()
println("="^100)
println("TOTAL: $n_pass passed, $n_fail failed")
println("="^100)
n_fail == 0 || error("$n_fail check(s) failed")
println("DONE")
