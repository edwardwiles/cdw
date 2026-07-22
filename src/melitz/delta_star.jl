# Wiring into the existing Christensen-Connault inner minimum-divergence loop.
# See docs/melitz_delta_star.md Section 11 ("Reused Ricardian infrastructure").
#
# Reuses cc_algo/PsiObjectiveBundle.jl's PsiObjectiveBundleDelta and
# cc_algo/inner_loop_functions.jl's inner_loop UNMODIFIED -- the Melitz module supplies
# only a moments!(K, G, theta, U, obj) function matching the required contract.
#
# SCOPE (this milestone only validates the INNER loop, i.e. Delta(theta*) at the fixed F*
# parameters -- per the brief, finite-delta upper/lower bound programs and a fully general
# outer search over arbitrary theta are explicitly out of scope). Consequently the theta
# vector here packs only (A, f); w, f_entry, N, expenditure are held fixed in the closure
# (obj.gamma) at their F*-solved values. `melitz_moments_adapter!` is therefore only
# guaranteed correct when evaluated AT theta* (the inner KNITRO solve only ever calls
# moments! once, with theta held fixed throughout -- it optimizes over the dual variables
# (zeta, lambda), never theta -- so this is not a limitation for this milestone's ask).
# A genuinely general moments! (supporting arbitrary theta perturbations, needed for the
# eventual finite-delta outer search) would need to re-equilibrate expenditure/N from
# scratch at every candidate theta -- flagged explicitly as the next step, not attempted
# here.

"""
    melitz_theta_layout(D) -> NamedTuple

Named index ranges for the packed theta vector `vcat(vec(A), vec(f))`
(length `2*D^2`).
"""
function melitz_theta_layout(D::Int)
    return (D=D, A=1:D^2, f=(D^2+1):(2D^2), length=2D^2)
end

"""
    melitz_pack_theta(A, f) -> theta
"""
melitz_pack_theta(A::AbstractMatrix, f::AbstractMatrix) = vcat(vec(A), vec(f))

"""
    melitz_unpack_theta(theta, layout) -> (A, f)
"""
function melitz_unpack_theta(theta::AbstractVector, layout)
    D = layout.D
    A = reshape(theta[layout.A], D, D)
    f = reshape(theta[layout.f], D, D)
    return A, f
end

"""
    melitz_moments_adapter!(K, G, theta, U, obj)

Adapter matching the `(K, G, theta, U, obj) -> nothing` contract required by
`PsiObjectiveBundleDelta` (docs Section 11). `obj.gamma` is a `NamedTuple` holding
everything besides `(A, f)` needed to reconstruct the model: `theta_layout, sigma,
theta_star, target_country, tau, w, f_entry, N, expenditure, moment_layout, X_data,
entry_target, scale_trade`. See the module docstring above for the scope limitation
(only valid at theta == theta*, not a general re-equilibration).
"""
function melitz_moments_adapter!(K, G, theta, U, obj)
    ctx = obj.γ
    A, f = melitz_unpack_theta(theta, ctx.theta_layout)
    D = ctx.theta_layout.D

    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                   ctx.tau, ctx.w, A, f, ctx.f_entry)
    eq = MelitzEquilibrium(ctx.N, ctx.expenditure, ones(eltype(A), D), ctx.cutoff, ctx.X_data)
    # theta is held fixed at theta* throughout a single inner_loop call (only the dual
    # variables (zeta, lambda) are optimized -- see module docstring), so the
    # counterfactual is simply the one already computed at construction time; recomputing
    # it via solve_autarky_counterfactual here would spuriously fail its internal
    # consistency check against a Mode-1 SAMPLE-based X_data (which differs from the
    # analytical X_data used to build ctx.cutoff by ordinary Monte Carlo noise, not a
    # modeling error).
    cf = ctx.counterfactual

    melitz_moments!(K, G, primitives, eq, cf, U, ctx.moment_layout;
                     X_data=ctx.X_data, entry_target=ctx.entry_target, scale_trade=ctx.scale_trade)
    return nothing
end

"""
    build_melitz_psi_bundle(data::MelitzSyntheticData; X_data=nothing, entry_target=nothing,
                             scale_trade=false, inner_loop_opt=..., outer_loop_opt=...)
        -> (obj::PsiObjectiveBundleDelta, theta_star::Vector{Float64})

Builds a `PsiObjectiveBundleDelta` wired to the Melitz moments and the packed
`theta* = vcat(vec(A*), vec(f*))` from a `MelitzSyntheticData` fixture. `X_data`/
`entry_target` default to the EXACT-SAMPLE targets built from `data.z_draws` (docs
"Mode 1" -- i.e. Delta(theta*) should come out ~0, not just small).
"""
function build_melitz_psi_bundle(data::MelitzSyntheticData;
                                  X_data::Union{Nothing,Matrix{Float64}}=nothing,
                                  entry_target::Union{Nothing,Vector{Float64}}=nothing,
                                  scale_trade::Bool=false,
                                  inner_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_inner_loop_options.opt"),
                                  outer_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_outer_loop_options.opt"))
    p, eq = data.primitives, data.equilibrium
    D = p.D
    z_draws = data.z_draws
    W = size(z_draws, 1)

    if X_data === nothing || entry_target === nothing
        X_sample = zeros(D, D)
        entry_sample = zeros(D)
        for o in 1:D
            for w in 1:W
                z = z_draws[w, o]
                profit_sum = 0.0
                for d in 1:D
                    firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                        eq.expenditure[d], 1.0, z)
                    X_sample[o, d] += eq.entrant_mass[o] * firm.realized_revenue / W
                    profit_sum += firm.realized_operating_profit
                end
                entry_sample[o] += profit_sum / W
            end
        end
        X_data = something(X_data, X_sample)
        entry_target = something(entry_target, entry_sample)
    end

    theta_layout = melitz_theta_layout(D)
    moment_layout = MelitzMomentLayout(D)
    L = eq.expenditure ./ p.w

    ctx = (theta_layout=theta_layout, sigma=p.sigma, theta_star=p.theta_star,
           target_country=p.target_country, tau=p.tau, w=p.w, f_entry=p.f_entry,
           N=eq.entrant_mass, expenditure=eq.expenditure, cutoff=eq.cutoff,
           counterfactual=data.counterfactual, moment_layout=moment_layout,
           X_data=X_data, entry_target=entry_target, scale_trade=scale_trade, L=L)

    theta = melitz_pack_theta(p.A, p.f)

    obj = PsiObjectiveBundleDelta(
        γ=ctx,
        (moments!)=melitz_moments_adapter!,
        d=moment_layout.num_moments,
        l=length(theta),
        inequality_index=Int64[],
        U=z_draws,
        inner_loop_opt=inner_loop_opt,
        outer_loop_opt=outer_loop_opt,
    )

    return obj, theta
end

"""
    run_melitz_inner_delta(data::MelitzSyntheticData; kwargs...) -> (val, x, nStatus)

Runs the REAL CC inner minimum-divergence loop (`cc_algo/inner_loop_functions.jl`'s
`inner_loop`, calling KNITRO, completely unmodified) at the F* parameters. `val` is
`Delta(theta*)` -- should be ~0 for the exact-sample smoke test (docs "Mode 1").
"""
function run_melitz_inner_delta(data::MelitzSyntheticData; kwargs...)
    obj, theta = build_melitz_psi_bundle(data; kwargs...)
    val, x, nStatus = inner_loop(obj, theta)
    return val, x, nStatus, obj
end
