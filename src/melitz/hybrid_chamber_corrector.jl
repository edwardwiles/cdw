# 2026-07-30 hybrid chamber corrector session. Governing prompt:
# melitz_hybrid_chamber_lfd_corrector_2026-07-30, following up on
# docs/melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md (`lfd_preserving_state.jl`,
# committed 523c4af). EXPERIMENTAL -- does NOT touch production `(A,f)` search
# (`finite_delta_outer.jl`) or the default `:logf`/`:logcutoff` outer-search backends. Reuses
# `lfd_preserving_state.jl`'s Step 1B/1D/1E machinery verbatim (cellwise A recovery, f
# recovery, operator-at-Afg witness) -- this file adds a genuinely DIFFERENT Steps 1C/1C':
# a discrete chamber-selection layer (A-gravity) and a continuous within-chamber layer
# (focal free-entry link), replacing the prior session's single two-lever discrete-only
# corrector.
#
# ============================================================================
# KEY STRUCTURAL FACT NOT USED BY THE PRIOR SESSION (derived here, verified against
# `moments.jl`/`firm_quantities.jl`, not assumed):
#
# The focal link's only q-dependence is through the origin-j (target country) row's own
# fixed cost `f[j,d]`, via `melitz_moments!`'s own `profit_j[w] = sum_d
# firm(...).realized_operating_profit` (o==j only) and `melitz_firm`'s
# `realized_operating_profit = active ? (C_jd*z^(sigma-1)/sigma - w_j*f_jd) : 0`,
# `active = z > exp(q_jd)`. Fixing the ACTIVE SET at every row-j cell (i.e. staying inside
# the SAME chamber, no draw crossing at any (j,d) cell), the p*-weighted focal residual is
#
#     FocalResidual(q_row_j) = CONST(chamber) - sum_{d != j} N_jd(p*) * f_jd(q_jd)     (**)
#
# where `N_jd(p*,q_jd) = sum_w p*_w * 1{z[w,j] > exp(q_jd)}` (a PLAIN, non-z-power-weighted
# probability tail statistic -- distinct from `lfd_preserving_state.jl`'s own `T_od`, which
# IS z-power-weighted and used for the TRADE-moment/A-recovery block only) and `CONST(chamber)
# = sum_d C_jd*T_jd(p*)/(sigma*w_j) - autarky_term(gamma_prime_j, A[j,j], f[j,j])` is fixed
# throughout the chamber (every term on that side is either z-power-weighted-tail, which is
# fixed within chamber by the SAME step-function argument `lfd_preserving_state.jl` already
# established for `T_od`, or autarky, which has no q-dependence at all).
#
# `N_jd(p*,q_jd)` is ALSO an exact step function of `q_jd` (same mechanism as `T_od`: removing
# draws from a plain indicator sum), constant within a chamber -- so (**) restricted to a
# FIXED chamber is
#
#     FocalResidual(q_row_j) = CONST(chamber) - sum_{d!=j} N_jd(p*)*K'_jd*exp((sigma-1)*q_jd)
#
# an EXACT (not linearized), smooth, STRICTLY monotone-decreasing-in-each-coordinate function
# of the free row-j q cells within the chamber (every additive term
# `-N_jd(p*)*K'_jd*exp((sigma-1)*q_jd)` is strictly decreasing in `q_jd` whenever
# `N_jd(p*)>0`, `K'_jd>0`) -- genuinely different from the A-gravity/trade-moment block, which
# is EXACTLY constant (a pure step function) within the same chamber. This is why a smooth,
# exact Newton corrector (not a discrete local search) is the right tool for the focal link
# ALONE, conditional on a fixed chamber -- exactly the task's own Phase 1/3 request, and a
# genuine correction to the prior session's blanket claim that "every downstream object is a
# step function" (only the A-block/trade-moment block is; f/the focal link is smooth within a
# chamber, jumping only when N_jd/T_jd themselves jump at a crossing).
# ============================================================================

using LinearAlgebra: dot

# ----------------------------------------------------------------------------------------
# Plain-probability tail statistic N_od(p,q_od) -- the non-z-power-weighted analogue of
# `lfd_preserving_state.jl`'s `T_od`/`melitz_origin_suffix_tail`. Needed for the focal-link
# closed form (**) above: the `-w_o*f_od` term of `melitz_firm`'s profit has NO z-dependence,
# so its p*-weighted expectation is `w_o*f_od*Pr_p(active)`, a plain probability mass, not a
# z^(sigma-1)-weighted tail sum.
# ----------------------------------------------------------------------------------------

"""
    melitz_origin_suffix_prob(sorted_ctx, o, p) -> Vector{Float64} (length W+1)

`suffix[k] = sum_{m=k}^{W} p[permutation[m,o]]`, `suffix[W+1]=0.0`. Precomputed once per
origin per fixed `p`; every subsequent `melitz_N_od` lookup is `O(log W)`.
"""
function melitz_origin_suffix_prob(sorted_ctx::MelitzSortedTailContext, o::Int,
                                    p::AbstractVector{Float64})
    W = sorted_ctx.W
    perm_o = @view sorted_ctx.permutation[:, o]
    suffix = zeros(Float64, W + 1)
    @inbounds for k in W:-1:1
        suffix[k] = suffix[k+1] + p[perm_o[k]]
    end
    return suffix
end

"""
    melitz_N_od(q_od, o, sorted_ctx, suffix_o_prob) -> Float64

`N_od(p,q_od) = sum` over draws `s` of origin `o` with `log(z_s) > q_od` (strict) of `p_s`
(no `z^(sigma-1)` weight -- contrast `melitz_T_od`). `O(log W)`.
"""
function melitz_N_od(q_od::Real, o::Int, sorted_ctx::MelitzSortedTailContext,
                      suffix_o_prob::AbstractVector{Float64})
    k0 = melitz_active_tail_start(view(sorted_ctx.sorted_log_z, :, o), q_od)
    return suffix_o_prob[k0]
end

# ----------------------------------------------------------------------------------------
# Chamber bin identity: two q values for the SAME origin are "in the same chamber" (T_od AND
# N_od both exactly unchanged) iff `melitz_active_tail_start` returns the same sorted
# position. This is the ground-truth chamber-signature check used throughout Phases 2-4
# below (rather than a hand-derived affine-inequality bookkeeping system, which is derived
# analytically in the accompanying report for documentation but NOT relied on for
# correctness here -- every "did a bin change" check in this file recomputes the bin
# directly, exactly, via this function).
# ----------------------------------------------------------------------------------------

"chamber bin id (sorted active-tail-start position) for cell (o,q_od)."
melitz_chamber_bin(o::Int, sorted_ctx::MelitzSortedTailContext, q_od::Real) =
    melitz_active_tail_start(view(sorted_ctx.sorted_log_z, :, o), q_od)

"""
    melitz_chamber_signature(q, sorted_ctx) -> Matrix{Int} (D x D)

Chamber bin id for every `(o,d)` cell of the FULL `q` matrix. Two full states with an
IDENTICAL signature have IDENTICAL cellwise-recovered `A` (every `T_od` unchanged) and
IDENTICAL `N_od` for every cell -- the exact ground-truth "same chamber" test.
"""
function melitz_chamber_signature(q::AbstractMatrix{Float64}, sorted_ctx::MelitzSortedTailContext)
    D = size(q, 1)
    sig = Matrix{Int}(undef, D, D)
    @inbounds for o in 1:D, d in 1:D
        sig[o, d] = melitz_chamber_bin(o, sorted_ctx, q[o, d])
    end
    return sig
end

"Number of cells whose chamber bin differs between two signatures (0 means identical chamber)."
melitz_chamber_signature_diff_count(sig_a::AbstractMatrix{Int}, sig_b::AbstractMatrix{Int}) =
    count(sig_a .!= sig_b)

# ----------------------------------------------------------------------------------------
# Exact focal-link closed form and its exact Jacobian w.r.t. the free row-j q cells, at a
# FIXED chamber. Reuses `T_od`'s own suffix-tail machinery (via a caller-supplied
# `melitz_origin_suffix_tail`, `lfd_preserving_state.jl`) for the trade-revenue piece and the
# new `melitz_origin_suffix_prob`/`melitz_N_od` for the fixed-cost piece.
# ----------------------------------------------------------------------------------------

"""
    MelitzFocalLinkChamberModel

Precomputed, chamber-fixed closed-form pieces for (**) above: `Njd[d] = N_jd(p*,q_jd)` (the
CURRENT chamber's fixed probability mass, `d==j` unused), `Kjd[d]` (the multiplicative
constant in `f[j,d] = Kjd[d]*exp((sigma-1)*q_jd)`, from `melitz_log_f_from_q`), and
`const_term` (everything else: the trade-revenue tail contributions, fixed within chamber,
plus the (q-independent) autarky term, plus the current row-j `A`-dependent `C_jd` pieces).
Rebuilding this model is `O(D log W)`; evaluating (**) or its Jacobian at any `q_row_j` within
the SAME chamber is then `O(D)`, no operator rebuild, no `mul_Gt!` call.
"""
struct MelitzFocalLinkChamberModel
    j::Int
    sigma::Float64
    Njd::Vector{Float64}
    Kjd::Vector{Float64}
    const_term::Float64
end

"""
    melitz_build_focal_link_chamber_model(A, f, q, gamma_prime_j, p_star, ctx, sorted_ctx) -> MelitzFocalLinkChamberModel

Builds the chamber-fixed model (**) at the CURRENT explicit `(A,f,q,gamma_prime_j)` state
(via a direct, single, exact `mul_Gt!` evaluation to pin `const_term` so that (**) reproduces
the CURRENT exact focal residual identically at the current `q_row_j` -- avoids re-deriving
the trade-revenue/autarky pieces from raw primitives a second, error-prone way).
"""
function melitz_build_focal_link_chamber_model(A::AbstractMatrix{Float64}, f::AbstractMatrix{Float64},
        q::AbstractMatrix{Float64}, current_focal_residual::Real, p_star::Vector{Float64}, ctx,
        sorted_ctx::MelitzSortedTailContext)
    D = ctx.D
    j = ctx.target_country
    sigma = ctx.sigma
    suffix_j_prob = melitz_origin_suffix_prob(sorted_ctx, j, p_star)
    Njd = zeros(Float64, D)
    Kjd = zeros(Float64, D)
    s = 0.0
    @inbounds for d in 1:D
        d == j && continue
        Njd[d] = melitz_N_od(q[j, d], j, sorted_ctx, suffix_j_prob)
        Kjd[d] = f[j, d] * exp(-(sigma - 1) * q[j, d])   # f[j,d] = Kjd[d]*exp((sigma-1)*q_jd)
        s += Njd[d] * f[j, d]
    end
    const_term = current_focal_residual + s   # so that CONST - sum(Njd*f) == current_focal_residual exactly at q now
    return MelitzFocalLinkChamberModel(j, sigma, Njd, Kjd, const_term)
end

"Exact closed-form focal residual (**) at a trial row-j q vector (full D-length, index j unused), same chamber."
function melitz_focal_link_value(model::MelitzFocalLinkChamberModel, q_row_j::AbstractVector{Float64})
    s = 0.0
    @inbounds for d in eachindex(q_row_j)
        d == model.j && continue
        s += model.Njd[d] * model.Kjd[d] * exp((model.sigma - 1) * q_row_j[d])
    end
    return model.const_term - s
end

"Exact closed-form focal-link Jacobian entry d(FocalResidual)/d(q_jd), same chamber."
function melitz_focal_link_jacobian_entry(model::MelitzFocalLinkChamberModel, d::Int, q_jd::Real)
    d == model.j && return 0.0
    return -(model.sigma - 1) * model.Njd[d] * model.Kjd[d] * exp((model.sigma - 1) * q_jd)
end

# ----------------------------------------------------------------------------------------
# Phase 2: discrete A-gravity chamber selector -- bounded best-first/beam search over
# combinations of free-q chamber transitions, replacing the prior session's fixed
# two-lever-cell design. Every candidate is scored via the EXACT full cellwise recovery
# (`build_state`, `lfd_preserving_state.jl`'s own closure convention) -- never an additive
# approximation of switch effects.
# ----------------------------------------------------------------------------------------

"One scored node in the discrete beam search."
struct MelitzDiscreteNode
    qff::Vector{Float64}
    gravA::Float64
    total_disp::Float64
    depth::Int
end

"""
    melitz_discrete_chamber_selector(build_state, qff0, ctx, sorted_ctx; kwargs...)
        -> (qff_best, nodes_examined, best_node, all_frontiers)

Bounded best-first/beam search minimizing `|A-gravity residual|` (primary), then total
scaled free-q displacement from `qff0` (secondary), over combinations of up to
`max_depth` simultaneous single-coordinate chamber reassignments. Candidate coordinates are
the `lever_pool` free-q indices with the largest `|c_full|` leverage (a bounded, disclosed
high-leverage subspace, not literally every free coordinate); candidate new positions for
each lever are `melitz_candidate_q_positions`'s own bounded window (reused verbatim from
`lfd_preserving_state.jl`). `max_candidates` bounds the TOTAL number of distinct (lever,
position) transitions evaluated across the whole search (a hard cap, matching the governing
prompt's own "at most 200" scope).
"""
function melitz_discrete_chamber_selector(build_state::Function, qff0::Vector{Float64}, ctx,
        sorted_ctx::MelitzSortedTailContext;
        gravity_tol::Real=1e-10, max_depth::Int=3, beam_width::Int=50, lever_pool_size::Int=40,
        half_window::Int=40, max_candidates::Int=200)
    D = ctx.D
    q_pivot, cell_of_k = melitz_q_free_cell_map(ctx)
    n = length(qff0)

    leverage_order = sortperm(abs.(ctx.c_full[cell_of_k]); rev=true)
    lever_pool = leverage_order[1:min(lever_pool_size, n)]
    cell_o = [lin2od(cell_of_k[k], D)[1] for k in 1:n]

    gA0, status0 = let (A_t, _, _, _, st) = build_state(qff0)
        (melitz_gravity_A_residual(A_t, ctx), st)
    end
    all(s -> s == :ok, status0) || throw(ArgumentError("melitz_discrete_chamber_selector: initial qff0 is not :ok"))

    frontier = MelitzDiscreteNode[MelitzDiscreteNode(copy(qff0), gA0, 0.0, 0)]
    best = frontier[1]
    candidates_examined = 0
    nodes_log = MelitzDiscreteNode[frontier[1]]

    for depth in 1:max_depth
        abs(best.gravA) < gravity_tol && break
        new_pool = MelitzDiscreteNode[]
        for node in frontier
            for k in lever_pool
                candidates_examined >= max_candidates && break
                cands = melitz_candidate_q_positions(cell_o[k], sorted_ctx, node.qff[k]; half_window=half_window)
                for qc in cands
                    candidates_examined >= max_candidates && break
                    trial = copy(node.qff)
                    trial[k] = qc
                    A_t, _, _, _, status_t = build_state(trial)
                    candidates_examined += 1
                    all(s -> s == :ok, status_t) || continue
                    gA = melitz_gravity_A_residual(A_t, ctx)
                    disp = norm_disp(trial, qff0)
                    push!(new_pool, MelitzDiscreteNode(trial, gA, disp, depth))
                end
            end
        end
        isempty(new_pool) && break
        append!(new_pool, frontier)   # allow "keep current node" continuation across depths
        sort!(new_pool; by=nd -> (abs(nd.gravA) < gravity_tol ? 0 : 1, abs(nd.gravA), nd.total_disp))
        # dedupe by rounded qff vector
        seen = Set{Vector{Float64}}()
        deduped = MelitzDiscreteNode[]
        for nd in new_pool
            key = round.(nd.qff; digits=10)
            key in seen && continue
            push!(seen, key)
            push!(deduped, nd)
            length(deduped) >= beam_width && break
        end
        frontier = deduped
        push!(nodes_log, frontier[1])
        abs(frontier[1].gravA) < abs(best.gravA) && (best = frontier[1])
        abs(best.gravA) < gravity_tol && (best = frontier[1])
    end

    return best.qff, candidates_examined, best, nodes_log
end

"scaled displacement between two q_free_free vectors (Euclidean, unscaled -- disclosed simplification)."
norm_disp(a::AbstractVector{Float64}, b::AbstractVector{Float64}) = sqrt(sum((a .- b) .^ 2))

# ----------------------------------------------------------------------------------------
# Phase 3: continuous within-chamber focal corrector -- exact, smooth, damped Newton over
# the free row-j q cells (the only free coordinates with nonzero focal-link sensitivity),
# using the exact closed form (**)/its exact Jacobian above. A chamber-bin veto rejects any
# trial step that changes ANY cell's chamber bin (not just row j's) -- if repeated halving
# cannot avoid this, control returns to the discrete selector (status=:returned_to_discrete),
# per the governing prompt's own explicit instruction, rather than clipping silently.
# ----------------------------------------------------------------------------------------

"""
    melitz_continuous_focal_corrector(build_state, focal_eval, qff0, ctx, sorted_ctx, p_star; kwargs...)
        -> (qff_final, status::Symbol, iters, trace)

`focal_eval(A,f,gamma_prime_j) -> (trade_res::Matrix, focal_res::Float64)` is the caller's
exact witness closure (mirrors `lfd_preserving_state.jl`'s own `focal_residual_at` pattern --
passed explicitly rather than captured globally, so this function has no hidden dependence
on any particular `obj`/`ctx` binding).

`status` is one of `:converged` (|focal residual| < `focal_tol`), `:returned_to_discrete`
(a chamber-bin boundary could not be avoided within `max_shrinks` halvings at some
iteration), or `:max_iters` (bounded iteration budget exhausted without either).
"""
function melitz_continuous_focal_corrector(build_state::Function, focal_eval::Function,
        qff0::Vector{Float64}, ctx, sorted_ctx::MelitzSortedTailContext, p_star::Vector{Float64};
        focal_tol::Real=1e-9, max_iters::Int=50, max_shrinks::Int=20, min_step::Real=1e-14)
    D = ctx.D
    j = ctx.target_country
    q_pivot, cell_of_k = melitz_q_free_cell_map(ctx)
    n = length(qff0)
    lever_ks = [k for k in 1:n if lin2od(cell_of_k[k], D)[1] == j]

    qff = copy(qff0)
    A_t, f_t, gpj_t, q_t, status_t = build_state(qff)
    all(s -> s == :ok, status_t) || throw(ArgumentError("melitz_continuous_focal_corrector: initial state not :ok"))
    sig0 = melitz_chamber_signature(q_t, sorted_ctx)

    _, focal0 = focal_eval(A_t, f_t, gpj_t)
    trace = NamedTuple[]

    if isempty(lever_ks) || abs(focal0) < focal_tol
        return qff, :converged, 0, trace
    end

    for it in 1:max_iters
        model = melitz_build_focal_link_chamber_model(A_t, f_t, q_t, focal0, p_star, ctx, sorted_ctx)
        q_row_j_cur = [q_t[j, d] for d in 1:D]
        Jvec = [melitz_focal_link_jacobian_entry(model, d, q_row_j_cur[d]) for d in 1:D]
        # minimum-norm Newton step over the lever cells only: delta_q solves J*delta_q = -F
        # (single scalar equality), delta_q = -F * J / dot(J,J) restricted to lever cells.
        Jl = zeros(Float64, D)
        for k in lever_ks
            d = lin2od(cell_of_k[k], D)[2]
            Jl[d] = Jvec[d]
        end
        denom = dot(Jl, Jl)
        if !(denom > 0)
            return qff, :returned_to_discrete, it - 1, trace
        end
        step_scale = -focal0 / denom
        delta_full = step_scale .* Jl   # length D, indexed by destination d

        shrink = 0
        accepted = false
        local trial, A_tr, f_tr, gpj_tr, q_tr, status_tr, focal_tr, sig_tr
        damp = 1.0
        while shrink <= max_shrinks
            trial = copy(qff)
            for k in lever_ks
                d = lin2od(cell_of_k[k], D)[2]
                trial[k] = qff[k] + damp * delta_full[d]
            end
            A_tr, f_tr, gpj_tr, q_tr, status_tr = build_state(trial)
            if all(s -> s == :ok, status_tr)
                sig_tr = melitz_chamber_signature(q_tr, sorted_ctx)
                if melitz_chamber_signature_diff_count(sig_tr, sig0) == 0
                    _, focal_tr = focal_eval(A_tr, f_tr, gpj_tr)
                    if abs(focal_tr) < abs(focal0) || abs(focal0) < focal_tol
                        accepted = true
                        break
                    end
                end
            end
            damp /= 2
            shrink += 1
            damp < min_step && break
        end

        if !accepted
            return qff, :returned_to_discrete, it - 1, trace
        end

        push!(trace, (iter=it, damp=damp, focal_before=focal0, focal_after=focal_tr))
        qff = trial
        A_t, f_t, gpj_t, q_t = A_tr, f_tr, gpj_tr, q_tr
        focal0 = focal_tr

        abs(focal0) < focal_tol && return qff, :converged, it, trace
    end

    return qff, :max_iters, max_iters, trace
end

# ----------------------------------------------------------------------------------------
# Top-level hybrid constructor: Phase 2 (discrete) + Phase 3 (continuous), alternated up to
# `max_macro_rounds` times (returns to Phase 2 whenever Phase 3 reports
# `:returned_to_discrete`), then Phase 4 witness + round-trip audit.
# ----------------------------------------------------------------------------------------

"""
    MelitzHybridChamberState

Result of `melitz_construct_hybrid_chamber_state`. Same core fields as
`MelitzLFDPreservingState` (`lfd_preserving_state.jl`) plus explicit round-trip-audit fields
(Phase 4) and search diagnostics (discrete candidates examined, continuous iterations,
macro rounds).
"""
struct MelitzHybridChamberState
    A::Matrix{Float64}
    f::Matrix{Float64}
    q::Matrix{Float64}
    gamma_prime_j::Float64
    theta_free::Vector{Float64}
    A_status::Matrix{Symbol}
    trade_residuals::Matrix{Float64}
    focal_residual::Float64
    gravity_A_residual::Float64
    gravity_f_residual::Float64
    divergence_pstar::Float64
    feasible::Bool
    discrete_candidates_examined::Int
    continuous_iters::Int
    continuous_status::Symbol
    macro_rounds::Int
    roundtrip_max_abs_logA::Float64
    roundtrip_max_abs_logf::Float64
    roundtrip_max_abs_q::Float64
    roundtrip_exact::Bool
end

"""
    melitz_construct_hybrid_chamber_state(theta_anchor, p_star, ctx, obj, g_new, q_free_free_new;
        gravity_tol=1e-10, focal_tol=1e-9, moment_tol=1e-9, max_macro_rounds=5, kwargs...)
        -> MelitzHybridChamberState

Governing prompt Phases 2-4: discrete A-gravity chamber selection + continuous within-chamber
focal correction, alternated, then an exact witness + round-trip invariant check. Reuses
`lfd_preserving_state.jl`'s `melitz_cellwise_A_from_moments`/`melitz_f_from_Aq`/
`melitz_update_operator_at_Afg!`/`melitz_moment_residuals_under_p`/
`melitz_gravity_A_residual`/`melitz_gravity_f_residual` verbatim (no duplicated logic).
"""
function melitz_construct_hybrid_chamber_state(theta_anchor::AbstractVector, p_star::Vector{Float64},
        ctx, obj, g_new::Real, q_free_free_new::AbstractVector{Float64};
        gravity_tol::Real=1e-10, focal_tol::Real=1e-9, moment_tol::Real=1e-9,
        max_macro_rounds::Int=5, discrete_max_depth::Int=3, discrete_beam_width::Int=50,
        discrete_lever_pool_size::Int=40, discrete_half_window::Int=40, discrete_max_candidates::Int=200,
        continuous_max_iters::Int=50)

    D = ctx.D
    sigma = ctx.sigma
    sorted_ctx = ctx.sorted_tail_ctx
    nA = D^2 - 1
    theta_anchor_plain = melitz_unpower_theta_free(theta_anchor, ctx)
    A_free_placeholder = theta_anchor_plain[2:1+nA]

    A_anchor, _, _, _, q_anchor = expand_free_theta_logcutoff(theta_anchor_plain, ctx)

    function build_state(qff_trial::AbstractVector{Float64})
        theta_trial = vcat(g_new, A_free_placeholder, qff_trial)
        _, _, gpj_trial, _, q_trial = expand_free_theta_logcutoff(theta_trial, ctx)
        A_trial, status_trial = melitz_cellwise_A_from_moments(A_anchor, q_anchor, q_trial,
                                                                 p_star, sorted_ctx, sigma)
        f_trial = melitz_f_from_Aq(A_trial, q_trial, gpj_trial, ctx)
        return A_trial, f_trial, gpj_trial, q_trial, status_trial
    end

    function focal_eval(A_e, f_e, gpj_e)
        melitz_update_operator_at_Afg!(obj.op, A_e, f_e, gpj_e, ctx)
        return melitz_moment_residuals_under_p(obj.op, p_star)
    end

    qff = collect(Float64, q_free_free_new)
    total_discrete_candidates = 0
    total_continuous_iters = 0
    continuous_status = :not_run
    macro_round = 0

    for round in 1:max_macro_rounds
        macro_round = round
        A_t, _, _, _, status_t = build_state(qff)
        gA_now = all(s -> s == :ok, status_t) ? melitz_gravity_A_residual(A_t, ctx) : Inf

        if !(all(s -> s == :ok, status_t)) || abs(gA_now) >= gravity_tol
            qff, n_cand, _, _ = melitz_discrete_chamber_selector(build_state, qff, ctx, sorted_ctx;
                gravity_tol=gravity_tol, max_depth=discrete_max_depth, beam_width=discrete_beam_width,
                lever_pool_size=discrete_lever_pool_size, half_window=discrete_half_window,
                max_candidates=discrete_max_candidates)
            total_discrete_candidates += n_cand
        end

        qff, continuous_status, n_it, _ = melitz_continuous_focal_corrector(build_state, focal_eval, qff, ctx,
            sorted_ctx, p_star; focal_tol=focal_tol, max_iters=continuous_max_iters)
        total_continuous_iters += n_it

        continuous_status == :returned_to_discrete || break
    end

    A_new, f_new, gpj_new, q_new, status_new = build_state(qff)
    theta_free_recovered = reduce_to_free_theta_logcutoff(A_new, f_new, gpj_new, ctx)
    trade_res, focal_res = focal_eval(A_new, f_new, gpj_new)
    gA = melitz_gravity_A_residual(A_new, ctx)
    gF = melitz_gravity_f_residual(f_new, ctx)
    W = length(p_star)
    divp = melitz_primal_divergence(p_star, W)

    # Phase 4: round-trip audit -- expand the reduced theta_free back out and compare to the
    # EXPLICIT (A_new,f_new,q_new) directly, in log space (relative/absolute scale-appropriate).
    theta_rt_plain = melitz_unpower_theta_free(theta_free_recovered, ctx)
    A_rt, f_rt, gpj_rt, _, q_rt = expand_free_theta_logcutoff(theta_rt_plain, ctx)
    rt_logA = maximum(abs.(log.(A_rt) .- log.(A_new)))
    rt_logf = maximum(abs.(log.(f_rt) .- log.(f_new)))
    rt_q = maximum(abs.(q_rt .- q_new))
    rt_exact = rt_logA < 1e-9 && rt_logf < 1e-9 && rt_q < 1e-9 && isapprox(gpj_rt, gpj_new; atol=1e-12)

    all_ok = all(s -> s == :ok, status_new)
    feasible = all_ok && maximum(abs.(trade_res)) < moment_tol && abs(focal_res) < moment_tol &&
               abs(gA) < gravity_tol && abs(gF) < gravity_tol &&
               all(p_star .>= 0) && isapprox(sum(p_star), 1.0; atol=1e-8)

    return MelitzHybridChamberState(A_new, f_new, q_new, gpj_new, theta_free_recovered, status_new,
        trade_res, focal_res, gA, gF, divp, feasible, total_discrete_candidates, total_continuous_iters,
        continuous_status, macro_round, rt_logA, rt_logf, rt_q, rt_exact)
end
