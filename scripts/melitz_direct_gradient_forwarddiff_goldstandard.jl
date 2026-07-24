# Follow-up investigation (user-raised concern, 2026-07-24): the two new "direct" backends
# disagreed with the existing analytic (jac_h-based) backend by up to ~17% relative at
# random perturbations. This script settles WHICH ONE is actually closer to the true
# envelope-theorem gradient by building a THIRD, genuinely independent construction with NO
# finite-difference truncation at all: ForwardDiff dual numbers propagated through the
# ENTIRE composite objective L(theta) = (1/W)*sum_w Psi(arg0(theta)[w]) at the FIXED dual x
# -- melitz_firm/melitz_moments_adapter_outer! are already generic in the number type (used
# elsewhere in this codebase for exactly this reason, e.g. melitz_moment_directional_derivative),
# so no source changes are needed to differentiate through them.
using Printf, Random, LinearAlgebra, ForwardDiff
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

"""
    dense_G(theta, ctx, obj) -> Matrix{T}

Builds the full (W, d) moment matrix directly from `melitz_expand_theta`/`melitz_firm`
(the SAME per-cell formula `_fill_compact_direct_columns!`/`_fill_compact_link!` already
use, validated bit-exact against the dense reference elsewhere in this codebase) --
bypassing `melitz_moments_adapter!`/`MelitzEquilibrium` entirely, since `MelitzEquilibrium`'s
constructor requires ALL fields to share one homogeneous type `T<:Real` and would reject a
mix of `Dual` (from theta-derived A/f) and `Float64` (ctx.expenditure/X_data) -- a
ForwardDiff-incompatibility in that ONE constructor, unrelated to melitz_firm/
melitz_expand_theta themselves (both already generic, used elsewhere in this codebase with
ForwardDiff, e.g. `melitz_moment_directional_derivative`).
"""
function dense_G(theta::AbstractVector{T}, ctx, obj) where {T}
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    layout = ctx.moment_layout
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    price_power_d = 1.0
    G = zeros(T, W, layout.num_moments)
    @inbounds for o in 1:D, d in 1:D
        trade_col = layout.trade_index[o, d]
        lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma,
                                ctx.expenditure[d], price_power_d, z)
            G[w, trade_col] = firm.realized_revenue / ctx.expenditure[d] - lambda_od
        end
    end
    expenditure_prime = ctx.w_prime * ctx.L[j]
    price_power_autarky = gamma_prime_j
    profit_j = zeros(T, W)
    @inbounds for d in 1:D
        lambda_od = ctx.X_data[j, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj.U[w, j]
            firm = melitz_firm(ctx.w[j], ctx.tau[j, d], A[j, d], f[j, d], sigma,
                                ctx.expenditure[d], price_power_d, z)
            profit_j[w] += firm.realized_operating_profit
        end
    end
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_auk = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, price_power_autarky, z_j)
        G[w, layout.focal_link_index] = profit_j[w] / ctx.w[j] - firm_auk.realized_operating_profit / ctx.w_prime
    end
    return G
end

function L_composite(theta::AbstractVector{T}, ctx, obj, x::AbstractVector{Float64}) where {T}
    W = size(obj.U, 1)
    d = ctx.moment_layout.num_moments
    G = dense_G(theta, ctx, obj)
    zeta = x[1]
    lambda = @view x[2:end]
    arg0 = zeros(T, W)
    @inbounds for w in 1:W
        acc = zero(T)
        for k in 1:d
            acc += G[w, k] * lambda[k]
        end
        arg0[w] = -zeta - acc
    end
    psi = zeros(T, W)
    obj.Psi!(psi, arg0)
    return sum(psi) / W
end

function investigate_one(ctx, obj_inner, theta::Vector{Float64}; label::String, h::Real=1e-4, coords=nothing)
    println("\n", "="^100)
    println(label)
    println("="^100)
    inner_opt = obj_inner.inner_loop_opt
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    obj_ref = build_melitz_implicit_bundle(ctx, obj_inner.U, theta; delta=1.0, find_smallest=true,
        gradient_backend=:B_argument_localized_parallel, h=h, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
    CS = CounterfactualSensitivity
    obj_ref.use_cached_x = false; obj_ref.x .= NaN
    _, x, nStatus = CS.inner_loop_internal(obj_ref, theta)
    @printf("  inner solve: nStatus=%d\n", nStatus)
    nStatus in (0, -100, -101, -103) || error("inner solve did not converge, nStatus=$nStatus")

    # Sanity check: dense_G's own reimplementation (bypassing MelitzEquilibrium for
    # ForwardDiff-compatibility) must reproduce obj_ref.H's ALREADY-COMPUTED moments exactly
    # at Float64 theta, or the gold standard below would be gold-plated garbage.
    G_check = dense_G(theta, ctx, obj_ref)
    G_from_H = @view obj_ref.H[:, 3:end]
    max_g_diff = maximum(abs.(G_check .- G_from_H))
    @printf("  dense_G self-check vs obj.H: max|diff|=%.3e (must be ~0)\n", max_g_diff)
    max_g_diff < 1e-9 || error("dense_G reimplementation does not match obj.H -- gold standard invalid")
    obj_ref.x .= x
    n = length(theta)

    # Method 1: analytic (jac_h-based)
    local_jac_ref = zeros(n)
    dummy_g = zeros(n)
    obj_ref(x, dummy_g, theta; jac=local_jac_ref)

    # Method 2: direct (mine)
    direct_serial = make_melitz_gradient_delta_direct_serial(h)
    local_jac_direct = zeros(n)
    direct_serial(local_jac_direct, theta, ctx, obj_ref, x)

    # Gold standard: ForwardDiff through the TRUE composite, no FD anywhere
    g_fd = ForwardDiff.gradient(th -> L_composite(th, ctx, obj_ref, x), theta)
    local_jac_gold = -1e10 .* g_fd

    diff1 = abs.(local_jac_ref .- local_jac_gold)
    diff2 = abs.(local_jac_direct .- local_jac_gold)
    scale = max.(1.0, abs.(local_jac_gold))

    coords_to_check = coords === nothing ? (1:n) : coords
    println("  coord   analytic(M1)       direct(M2)          ForwardDiff(gold)     rel|M1-gold|   rel|M2-gold|")
    for r in coords_to_check
        @printf("  %4d  % .6e   % .6e   % .6e   %.3e     %.3e\n",
            r, local_jac_ref[r], local_jac_direct[r], local_jac_gold[r], diff1[r]/scale[r], diff2[r]/scale[r])
    end
    rel1 = diff1 ./ scale
    rel2 = diff2 ./ scale
    @printf("  SUMMARY: max rel|M1-gold|=%.4e (coord %d)   max rel|M2-gold|=%.4e (coord %d)\n",
        maximum(rel1), argmax(rel1), maximum(rel2), argmax(rel2))
    @printf("  median rel|M1-gold|=%.4e   median rel|M2-gold|=%.4e\n", median(rel1), median(rel2))
    return (rel1=rel1, rel2=rel2)
end

using Statistics: median

if abspath(PROGRAM_FILE) == @__FILE__
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj_inner, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt, needs_outer_moment_jacobian=false)
    ctx = obj_inner.γ

    r1 = investigate_one(ctx, obj_inner, theta0; label="Base (Pareto) point, h=1e-4")

    rng = MersenneTwister(7)
    pert = theta0 .+ 0.005 .* randn(rng, length(theta0))
    r2 = investigate_one(ctx, obj_inner, pert; label="Random perturbation 1, h=1e-4")

    # same perturbation point, smaller h -- does M1/M2's own gap to gold SHRINK with h?
    r2b = investigate_one(ctx, obj_inner, pert; label="Random perturbation 1, h=1e-5", h=1e-5)
    r2c = investigate_one(ctx, obj_inner, pert; label="Random perturbation 1, h=1e-6", h=1e-6)
end
