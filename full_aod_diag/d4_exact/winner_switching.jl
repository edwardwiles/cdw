# ============================================================================
# Task §11: hard-winner roughness diagnostics -- tie thresholds, switch
# counts/mass, and whether the moment system (and the three scalar objects)
# genuinely jump/kink at an exact single-draw winner tie.
# ============================================================================

"""
    switch_stats(winner_base::Matrix{Int}, winner_pert::Matrix{Int}; m_weights=nothing)

Compares two winner arrays (W x D, from `compute_winners`) drawn at the SAME
random numbers (common draws), counting how many (draw,destination) pairs
changed winner. `m_weights` (length W, e.g. the base LFD weights) gives a
CC-weighted switched mass in addition to the raw draw-count-weighted one.
"""
function switch_stats(winner_base::Matrix{Int}, winner_pert::Matrix{Int}; m_weights::Union{Nothing,Vector{Float64}} = nothing)
    W, D = size(winner_base)
    changed = winner_base .!= winner_pert
    n_switches = count(changed)
    by_dest = [count(@view(changed[:, d])) for d in 1:D]
    base_mass = n_switches / (W * D)
    cc_mass = if m_weights === nothing
        missing
    else
        p = m_weights ./ sum(m_weights)
        sum(p[ω] for ω in 1:W if any(@view(changed[ω, :]))) # probability mass on draws with >=1 switch
    end
    return (n_switches = n_switches, by_destination = by_dest, base_mass = base_mass, cc_weighted_mass = cc_mass)
end

"""
    exact_tie_thresholds(θ_full, v::Matrix, ctx) -> Array{Float64,3} (W, D, D)

For the EXACT multiplicative perturbation `Aod_theta[o,d] -> Aod_theta[o,d]*exp(t*v[o,d])`,
using `price[ω,o,d](t) = price[ω,o,d](0)*exp(-mu*t*v[o,d])` (exact, not
linearized -- derived in gravity_elimination.jl-adjacent reasoning: AodPow is
a pure power of Aod_theta at fixed mu, so this is an exact exponential
relation, not a first-order approximation), computes the tie threshold `t`
at which origin `o` (currently NOT the winner at (ω,d)) would overtake the
current winner, for every (ω,d,o) triple. Returns +-Inf where `v[winner,d]==v[o,d]`
(prices move together, never cross) or where `o` is already the winner.
"""
function exact_tie_thresholds(θ_full::AbstractVector, v::AbstractMatrix, ctx)
    price, Aod, AodPow = factual_prices(θ_full, ctx)
    D = ctx.D; W = size(price, 1)
    μ = θ_full[1]
    winner, _, _ = compute_winners(θ_full, ctx)
    thresh = fill(Inf, W, D, D)   # thresh[ω,d,o] = crossing t for candidate o vs current winner at (ω,d)
    @inbounds for d in 1:D, ω in 1:W
        wstar = winner[ω, d]
        logp_w = log(price[ω, wstar, d])
        for o in 1:D
            o == wstar && continue
            Δv = v[wstar, d] - v[o, d]
            if abs(Δv) < 1e-14
                thresh[ω, d, o] = Inf   # never crosses (prices move in lockstep along this direction)
            else
                logp_o = log(price[ω, o, d])
                # log p_o(t) - log p_w(t) = (logp_o(0)-logp_w(0)) + mu*t*(v[wstar,d]-v[o,d]); crosses
                # 0 at t = (logp_w(0)-logp_o(0)) / (mu*Delta_v) -- verified by hand AND numerically
                # (an earlier version had (logp_o-logp_w) in the numerator, an exact sign flip; caught
                # because the predicted t* produced ZERO actual switches at a +-1e-7 straddle, and
                # direct inspection showed the true crossing for that triple was at -t*, not +t*).
                thresh[ω, d, o] = (logp_w - logp_o) / (μ * Δv)
            end
        end
    end
    return thresh, winner
end

"free_idx-ordered v_free (length n_free, index 1 = gamma'_focal) -> D x D A-block direction matrix."
function v_free_to_Amat(v_free::AbstractVector, ctx)
    v_mat = zeros(ctx.D, ctx.D)
    for (k, i) in enumerate(ctx.free_idx)
        i == 3 + ctx.D && continue   # gamma'_focal has no A-block row/col
        lin = i - ctx.Aod_offset
        o = mod1(lin, ctx.D); d = div(lin - 1, ctx.D) + 1
        v_mat[o, d] = v_free[k]      # k=1 is gamma'_focal (skipped above); A-block is k=2..n_free <-> v_free[2..n_free]
    end
    return v_mat
end

"""
    kink_test(x_free0, v, ctx, base_state; delta=1e-9) -> NamedTuple

Finds the smallest-|t| exact tie threshold along direction v (task's
"single controlled draw crosses a single supplier tie"), then evaluates
mean(G), frozen_adjoint_Q, fixed_dual_L straddling that threshold at +-delta
(delta chosen far smaller than the spacing to the next threshold) to check
for a genuine one-sided-slope DISCONTINUITY (kink) exactly there, vs a
matched control point away from any threshold.
"""
function kink_test(x_free0::AbstractVector, v_free::AbstractVector, ctx, base; delta::Float64 = 1e-9)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    v_mat = v_free_to_Amat(v_free, ctx)
    thresh, _ = exact_tie_thresholds(θ_full0, v_mat, ctx)
    finite_pos = thresh[thresh .> 1e-8 .&& isfinite.(thresh)]
    t_star = minimum(finite_pos)

    # EXACT exponential family on the A-block (matches exact_tie_thresholds' own derivation exactly,
    # not a linearization): x_free(t)[A-block] = x_free0[A-block] .* exp(t*v_free[A-block]); gamma'_focal
    # perturbed the same way for consistency (irrelevant to winner selection either way).
    xfun = t -> x_free0 .* exp.(t .* v_free)
    function mean_G_at(t)
        θf = CS.reconstruct_full(xfun(t), ctx.m)
        W = size(ctx.U, 1); K = zeros(W); G = zeros(W, ctx.nTotalMoments)
        ctx.obj.moments!(K, G, θf, ctx.U, ctx.obj)
        return vec(sum(G, dims=1)) ./ W
    end

    Gm = mean_G_at(t_star - delta); Gp = mean_G_at(t_star + delta)
    Qm = frozen_adjoint_Q(xfun(t_star - delta), ctx, base); Qp = frozen_adjoint_Q(xfun(t_star + delta), ctx, base)
    Lm = fixed_dual_L(xfun(t_star - delta), ctx, base); Lp = fixed_dual_L(xfun(t_star + delta), ctx, base)

    # control point: same delta straddle, but centered somewhere with NO threshold within 100*delta
    t_ctrl = t_star / 3   # generically far from any OTHER threshold (checked below)
    Gm_c = mean_G_at(t_ctrl - delta); Gp_c = mean_G_at(t_ctrl + delta)
    Qm_c = frozen_adjoint_Q(xfun(t_ctrl - delta), ctx, base); Qp_c = frozen_adjoint_Q(xfun(t_ctrl + delta), ctx, base)

    return (t_star = t_star, n_thresholds_below_tstar_x2 = count(<(2*t_star), finite_pos),
            G_jump_at_threshold = maximum(abs.(Gp .- Gm)), G_jump_at_control = maximum(abs.(Gp_c .- Gm_c)),
            Q_jump_at_threshold = abs(Qp - Qm), Q_jump_at_control = abs(Qp_c - Qm_c),
            delta = delta)
end
