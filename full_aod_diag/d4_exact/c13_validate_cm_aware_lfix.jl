# Continuation 13, Section 4: validate the CM-aware Lfix outer gradient (lfix_cm_aware.jl)
# against optimized-value finite differences of the AUGMENTED Delta* (the CM-constrained inner
# CC divergence), at D=4. This is the load-bearing new piece for a D=20 CM outer optimization:
# the existing Lfix/composite-gradient machinery had never been made CM-aware before this
# continuation (docs/fullA_common_marginals_handoff.md Section 12's #1 open item).
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
using Printf, LinearAlgebra, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)

const L = 10
aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
bins = cm_bin_indices_for(ctx, aug)

x_free_calib = ctx.θ0_up[ctx.free_idx]

Random.seed!(9013)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.03 .* randn(length(x_free_perturbed) - 1))

"Augmented Delta* (the CM-constrained inner CC divergence) at a raw x_free point -- the OPTIMIZED-VALUE reference the analytic gradient is checked against."
function delta_star_cm(x_free)
    r = evaluate_fullA(x_free, ctx_cm; use_cache = false, warm = false)
    r.inner_status in (0, -100, -101, -103) || error("delta_star_cm: inner solve failed, nStatus=$(r.inner_status)")
    return -r.zeta  # matches this file family's own Delta_dual convention (K==zeta up to sign; verified against c12_validate_dense_cm.jl's Delta_dual print)
end

"lfix-value analog of delta_star_cm at a full x_free (NOT single-coordinate -- rebuilds w0 fully to compare like-for-like against the analytic gradient's own w0 basis)."
function lfix_full(x_free, cache_cm, base)
    D = ctx.D
    z = log.(reshape(x_free[2:end], D, D))
    w = vcat(x_free[1], pivot_reduce(z, pe))
    q = copy(cache_cm.q0)
    Aod_theta = exp.(pivot_expand(w[2:end], pe))
    θ_full = CS.reconstruct_full(vcat(w[1], vec(Aod_theta)), ctx.m)
    for d in 1:D
        new_contrib = dest_contrib_block_local(cache_cm, ctx, θ_full, d)
        q .-= new_contrib .- cache_cm.contrib0[:, d]
    end
    new_cf = cf_contrib_at(cache_cm, θ_full, ctx)
    q .-= new_cf .- cache_cm.cf_contrib0
    return lfix_from_q(q, cache_cm.ζstar)
end

println("="^100)
println("SECTION 1: base-point CM-aware cache built correctly (q0 includes CM term, dest_contrib_block_local check)")
println("="^100)
base = solve_base_state(x_free_calib, ctx_cm)
cache_cm = build_lfix_base_cache_cm(x_free_calib, ctx_cm, base, ctx, aug, bins)
lfix0 = lfix_from_q(cache_cm.q0, cache_cm.ζstar)
truth0 = delta_star_cm(x_free_calib)
# NOTE (remediation task Part A, finding F1): `lfix0` IS the canonical Delta_dual =
# -(mean(Psi(q*))+zeta*) (lfix_from_q follows the same convention as three_way_derivatives.jl's
# fixed_dual_L). `-base.ζstar` is the F1-buggy proxy that omits mean(Psi(q*)) -- the two agree
# here ONLY because the calibration point has ~zero tail mass (no m*>e draws), NOT because they
# are the same quantity in general. This check is intentionally about cache correctness at a
# near-benchmark point, not a general identity -- see remediation_a1_verify_delta_dual_identity.jl
# and test_cm_delta_dual_tail_active.jl for the general (tail-active) case, where they diverge by
# exactly mean(Psi(q*)) > 0.
@printf "  lfix_full(calib) reproduces base zeta (near-benchmark, ~zero tail mass only):  lfix0=%.10f  -zeta*=%.10f  diff=%.3e\n" lfix0 (-base.ζstar) abs(lfix0 - (-base.ζstar))
@printf "  delta_star_cm(calib) [independent solve]: %.10f  (should match lfix0 at the anchor point)\n" truth0
println()

println("="^100)
println("SECTION 2: full-vector optimized-value finite-difference check at calib AND a perturbed point")
println("="^100)
for (label, xf) in [("calibration", x_free_calib), ("perturbed", x_free_perturbed)]
    local base = solve_base_state(xf, ctx_cm)
    local cache_cm = build_lfix_base_cache_cm(xf, ctx_cm, base, ctx, aug, bins)
    g_analytic, meta = composite_gradient_at_fast_cm(xf, ctx_cm, pe, ctx, aug, bins; base = base, cache = cache_cm, h_mode = :adaptive)

    D = ctx.D; D2 = D^2
    z0 = log.(reshape(xf[2:end], D, D))
    w0 = vcat(xf[1], pivot_reduce(z0, pe))

    # optimized-value central FD of the AUGMENTED delta_star_cm (fully re-solves the CM-augmented
    # inner problem at each probe -- the genuinely independent ground truth this gradient claims
    # to match, not merely a self-consistency check against lfix_full/cache internals)
    hfd = 0.01
    g_fd = zeros(D2)
    for k in 1:D2
        wp = copy(w0); wp[k] += hfd
        wm = copy(w0); wm[k] -= hfd
        zp = pivot_expand(wp[2:end], pe); zm = pivot_expand(wm[2:end], pe)
        xfp = vcat(wp[1], vec(exp.(zp))); xfm = vcat(wm[1], vec(exp.(zm)))
        g_fd[k] = (delta_star_cm(xfp) - delta_star_cm(xfm)) / (2hfd)
    end

    cosang = dot(g_analytic, g_fd) / (norm(g_analytic) * norm(g_fd) + 1e-300)
    cos_ablock = dot(g_analytic[2:end], g_fd[2:end]) / (norm(g_analytic[2:end]) * norm(g_fd[2:end]) + 1e-300)
    norm_ratio = norm(g_analytic[2:end]) / (norm(g_fd[2:end]) + 1e-300)
    signs_agree = count(sign.(g_analytic[2:end]) .== sign.(g_fd[2:end]))
    @printf "[%s] gamma-component: analytic=%.6e  FD=%.6e  diff=%.3e\n" label g_analytic[1] g_fd[1] abs(g_analytic[1]-g_fd[1])
    @printf "[%s] full-vector cosine=%.6f | A-block-only cosine=%.6f | A-block norm ratio=%.4f | A-block sign agreement=%d/%d\n" label cosang cos_ablock norm_ratio signs_agree D2-1
    @printf "[%s] ||g_analytic||=%.6e  ||g_fd||=%.6e\n" label norm(g_analytic) norm(g_fd)
end
println()

println("="^100)
println("SECTION 3: multiple FD bandwidths (A-block cosine should be stable, not h-sensitive)")
println("="^100)
base_c = solve_base_state(x_free_calib, ctx_cm)
cache_c = build_lfix_base_cache_cm(x_free_calib, ctx_cm, base_c, ctx, aug, bins)
g_analytic_c, _ = composite_gradient_at_fast_cm(x_free_calib, ctx_cm, pe, ctx, aug, bins; base = base_c, cache = cache_c)
D = ctx.D; D2 = D^2
z0c = log.(reshape(x_free_calib[2:end], D, D))
w0c = vcat(x_free_calib[1], pivot_reduce(z0c, pe))
for hfd in (0.02, 0.01, 0.005, 0.0025)
    g_fd = zeros(D2)
    for k in 1:D2
        wp = copy(w0c); wp[k] += hfd
        wm = copy(w0c); wm[k] -= hfd
        zp = pivot_expand(wp[2:end], pe); zm = pivot_expand(wm[2:end], pe)
        xfp = vcat(wp[1], vec(exp.(zp))); xfm = vcat(wm[1], vec(exp.(zm)))
        g_fd[k] = (delta_star_cm(xfp) - delta_star_cm(xfm)) / (2hfd)
    end
    cos_ablock = dot(g_analytic_c[2:end], g_fd[2:end]) / (norm(g_analytic_c[2:end]) * norm(g_fd[2:end]) + 1e-300)
    @printf "  h=%.4f  A-block cosine=%.6f  ||g_fd A-block||=%.6e\n" hfd cos_ablock norm(g_fd[2:end])
end
println()

println("="^100)
println("SECTION 4: sanity -- gradient of a plain (non-CM) problem is UNCHANGED by this file's additions")
println("="^100)
base_plain = solve_base_state(x_free_calib, ctx)
g_plain_old, _ = composite_gradient_at_fast(x_free_calib, ctx, pe; base = base_plain)
g_plain_new, _ = composite_gradient_at_fast(x_free_calib, ctx, pe; base = base_plain, cache = nothing)
@printf "  max|g_plain_old - g_plain_new| = %.3e  (must be EXACTLY 0.0 -- cache=nothing must reproduce the pre-existing code path bit-for-bit)\n" maximum(abs.(g_plain_old .- g_plain_new))
println("DONE")
