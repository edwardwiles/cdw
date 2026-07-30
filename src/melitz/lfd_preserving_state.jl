# 2026-07-30 continuation session: joint (A,q) feasibility-preserving movements.
# Governing prompt: melitz_joint_Aq_feasibility_preserving_search_2026-07-30 (following up
# on docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md). EXPERIMENTAL -- does NOT
# touch production (A,f) search (`finite_delta_outer.jl`) or the default `:logf`/`:logcutoff`
# outer-search backends at all. A separate, additive module only.
#
# ============================================================================
# Phase 0 (derivation, verified against `moments.jl`/`firm_quantities.jl`/
# `origin_block_screen.jl` -- NOT assumed from the schematic prompt formula):
#
# `melitz_firm` (firm_quantities.jl): price_od(z) = markup*w_o*tau_od/(A_od*z),
# rev_od(z) = expenditure_d*price_od(z)^(1-sigma) = C_od*z^(sigma-1),
# C_od = expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma) (`melitz_C`), active iff
# rev_od(z)/sigma - w_o*f_od > 0 iff z > zhat_od = exp(q_od) (STRICT, `melitz_active_tail_start`).
#
# `melitz_moments!` (moments.jl): G[w,trade_col(o,d)] = C_od*z[w,o]^(sigma-1)*active/expenditure_d
# - lambda_od, lambda_od = X_data[o,d]/expenditure_d. Writing coef_od = C_od/expenditure_d
# and T_od(p,q_od) = sum_w p_w * z[w,o]^(sigma-1) * 1{z[w,o] > exp(q_od)} (the p-weighted
# active tail statistic), E_p[G[:,trade_col(o,d)]] = coef_od*T_od(p,q_od) - lambda_od
# (`mul_Gt!`, moment_operator.jl:314-362, confirms exactly this when sum(p)==1).
#
# coef_od = (markup*w_o*tau_od/A_od)^(1-sigma) = K_od * A_od^(sigma-1), K_od =
# (markup*w_o*tau_od)^(1-sigma) INDEPENDENT of A_od. So "E_p[trade moment od] == 0" iff
# T_od(p,q_od) == lambda_od/coef_od =: H_od (EXACTLY origin_block_screen.jl's own H[d]
# target, `melitz_origin_block_lp` line ~124-130 -- same object, re-derived independently
# here, not assumed). Given a FIXED p and holding K_od/lambda_od fixed (both depend only on
# w,tau,expenditure,X_data -- never on A or q), two states (anchor, new) both satisfying
# E_p[trade moment od]==0 exactly must have coef_od(new)*T_od(new) == coef_od(anchor)*T_od(anchor),
# i.e. (A_new/A_anchor)^(sigma-1) == T_od(anchor;p)/T_od(new;p), i.e.
#
#     a_od(new) = a_od(anchor) + [log(T_od(anchor;p)) - log(T_od(new;p))] / (sigma-1)     (*)
#
# -- EXACTLY the prompt's own proposed inversion, confirmed (not assumed) against the
# actual `melitz_C`/`melitz_firm`/`mul_Gt!` code. `T_od(p,q_od)` is a **piecewise-constant,
# non-increasing step function of q_od** (removing draws from an indicator sum), so (*) is
# EXACT (never a finite-difference/linearization) at every q_od, including across a
# participation switch -- but (*) is ONLY finite when T_od(new;p) > 0; if q_od moves past
# EVERY draw with p_s>0 in origin o's column, T_od(new;p)=0 and NO FINITE A_od can restore
# the moment under the FIXED p (an intrinsic infeasibility of holding p fixed at that q,
# not a solver failure -- see `:zero_tail_new` below and Phase 2/3 below).
#
# Failure/edge conditions (governing prompt Phase 0's own required list):
#   - `:zero_tail_new`   -- T_od(new;p)==0: q_od moved past every p-positive draw; a_od(new)
#                            would be +Inf. Intrinsic (matches the audited origin-14/dest-16/19
#                            cliff's own mechanism at the ORIGIN-BLOCK level).
#   - `:zero_tail_anchor` -- T_od(anchor;p)==0: should not occur for a verified FiniteSolved
#                            anchor (p exactly satisfies coef_od*T_od(anchor)=lambda_od>0
#                            there), guarded defensively (X_data[o,d]==0 -- zero observed
#                            trade -- makes lambda_od=H_od=0, satisfied trivially by
#                            T_od=0, and A_od is then genuinely INDETERMINATE by this
#                            moment alone; held at the anchor value, flagged, never silently
#                            treated as :ok).
#   - nonfinite / A_od<=0 -- guarded via `isfinite`/`>0` checks, never silently propagated.
#
# ============================================================================

using LinearAlgebra: dot

# ----------------------------------------------------------------------------------------
# Building block 1: exact p-weighted active-tail statistic T_od(p, q_od), via the SAME
# per-origin sorted-tail context (`sorted_tail.jl`) and the SAME strict active-tail
# convention (`melitz_active_tail_start`) production already uses -- not a re-derivation.
# ----------------------------------------------------------------------------------------

"""
    melitz_origin_suffix_tail(sorted_ctx, o, p) -> Vector{Float64} (length W+1)

`suffix[k] = sum_{m=k}^{W} p[sorted_ctx.permutation[m,o]] * sorted_ctx.sorted_z_power[m,o]`,
`suffix[W+1] = 0.0`. Precomputed ONCE per origin per fixed `p` (`O(W)`); every subsequent
`melitz_T_od` lookup at that origin is then `O(log W)` (one `searchsortedlast`) + `O(1)`.
"""
function melitz_origin_suffix_tail(sorted_ctx::MelitzSortedTailContext, o::Int,
                                    p::AbstractVector{Float64})
    W = sorted_ctx.W
    perm_o = @view sorted_ctx.permutation[:, o]
    zp_o = @view sorted_ctx.sorted_z_power[:, o]
    suffix = zeros(Float64, W + 1)
    @inbounds for k in W:-1:1
        suffix[k] = suffix[k+1] + p[perm_o[k]] * zp_o[k]
    end
    return suffix
end

"""
    melitz_T_od(q_od, o, sorted_ctx, suffix_o) -> Float64

`T_od(p,q_od) = sum` over draws `s` of origin `o` with `log(z_s) > q_od` (STRICT, matching
`melitz_active_tail_start`) of `p_s * z_s^(sigma-1)`, read off the precomputed suffix-tail
array (`melitz_origin_suffix_tail`). `O(log W)`.
"""
function melitz_T_od(q_od::Real, o::Int, sorted_ctx::MelitzSortedTailContext,
                      suffix_o::AbstractVector{Float64})
    k0 = melitz_active_tail_start(view(sorted_ctx.sorted_log_z, :, o), q_od)
    return suffix_o[k0]
end

# ----------------------------------------------------------------------------------------
# Building block 2: Step 1B -- exact cell-by-cell A recovery from formula (*).
# ----------------------------------------------------------------------------------------

"""
    melitz_cellwise_A_from_moments(A_anchor, q_anchor, q_new, p_star, sorted_ctx, sigma)
        -> (A_new, status)

Step 1B: for EVERY cell `(o,d)` (all `D^2`, including `(j,j)`), recomputes `A_new[o,d]` via
formula (*) above so that the FIXED distribution `p_star` exactly satisfies cell `(o,d)`'s
own bilateral trade-share moment at the NEW cutoff `q_new[o,d]`. Exact, not a finite-
difference approximation. `status[o,d]` is `:ok`, `:zero_tail_new`, or `:zero_tail_anchor`
(see module header) -- callers MUST check `status` before trusting `A_new` (a non-`:ok`
cell means no finite `A_od` exists that preserves the moment under `p_star` at this `q_od`).
"""
function melitz_cellwise_A_from_moments(A_anchor::AbstractMatrix{Float64}, q_anchor::AbstractMatrix{Float64},
                                         q_new::AbstractMatrix{Float64}, p_star::AbstractVector{Float64},
                                         sorted_ctx::MelitzSortedTailContext, sigma::Real)
    D = size(A_anchor, 1)
    A_new = zeros(Float64, D, D)
    status = fill(:ok, D, D)
    @inbounds for o in 1:D
        suffix_o = melitz_origin_suffix_tail(sorted_ctx, o, p_star)
        for d in 1:D
            T_anchor = melitz_T_od(q_anchor[o, d], o, sorted_ctx, suffix_o)
            T_new = melitz_T_od(q_new[o, d], o, sorted_ctx, suffix_o)
            if !(T_anchor > 0)
                status[o, d] = :zero_tail_anchor
                A_new[o, d] = A_anchor[o, d]
            elseif !(T_new > 0)
                status[o, d] = :zero_tail_new
                A_new[o, d] = Inf
            else
                a_new_od = log(A_anchor[o, d]) + (log(T_anchor) - log(T_new)) / (sigma - 1)
                A_new[o, d] = exp(a_new_od)
            end
        end
    end
    return A_new, status
end

# ----------------------------------------------------------------------------------------
# Building block 3: Step 1D -- f recovery from (A,q), SAME formulas
# `expand_free_theta_logcutoff` uses (log_cutoff_param.jl), applied to a caller-supplied
# (A,q) instead of a pivot-expanded one -- not duplicated logic, just re-invoked per cell.
# ----------------------------------------------------------------------------------------

"""
    melitz_f_from_Aq(A, q, gamma_prime_j, ctx) -> f

Step 1D: `f[o,d] = exp(melitz_log_f_from_q(q[o,d], log(A[o,d]), ...))` for every off-`(j,j)`
cell; `f[j,j] = derive_fjj_from_autarky_cutoff(gamma_prime_j, ...)` (independent of `q[j,j]`,
matching `expand_free_theta_logcutoff`'s own convention).
"""
function melitz_f_from_Aq(A::AbstractMatrix{Float64}, q::AbstractMatrix{Float64},
                           gamma_prime_j::Real, ctx)
    D = ctx.D
    j = ctx.target_country
    sigma = ctx.sigma
    f = zeros(Float64, D, D)
    @inbounds for o in 1:D, d in 1:D
        (o == j && d == j) && continue
        f[o, d] = exp(melitz_log_f_from_q(q[o, d], log(A[o, d]), ctx.w[o], ctx.tau[o, d],
                                           ctx.expenditure[d], sigma))
    end
    f[j, j] = derive_fjj_from_autarky_cutoff(gamma_prime_j, ctx.w_prime, 1.0, A[j, j],
                                              ctx.w_prime * ctx.L[j], sigma)
    return f
end

# ----------------------------------------------------------------------------------------
# Building block 4: operator update from an EXPLICIT (A,f,gamma_prime_j) state, bypassing
# `melitz_expand_theta`/the gravity pivot entirely. `melitz_update_operator_at_theta!`
# (cc_bundle.jl) ALWAYS re-imposes gravity via the pivot -- unusable here since Step 1B's
# cellwise-recovered `A` is deliberately allowed to violate A-gravity before Step 1C runs.
# Same downstream construction, reused verbatim.
# ----------------------------------------------------------------------------------------

"""
    melitz_update_operator_at_Afg!(op, A, f, gamma_prime_j, ctx) -> op

Matrix-free operator update from an EXPLICIT `(A,f,gamma_prime_j)` state (no pivot
involved). Same `MelitzPrimitives`/`MelitzEquilibrium`/`MelitzCounterfactual`/
`melitz_update_moment_operator!` construction `melitz_update_operator_at_theta!`
(cc_bundle.jl) uses, applied directly to caller-supplied primitives.
"""
function melitz_update_operator_at_Afg!(op::MelitzMomentOperator, A::AbstractMatrix{Float64},
                                         f::AbstractMatrix{Float64}, gamma_prime_j::Real, ctx)
    D = ctx.D
    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                   ctx.tau, ctx.w, A, f, gamma_prime_j)
    cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    eq = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
    cf = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)
    melitz_update_moment_operator!(op, primitives, eq, cf; X_data=ctx.X_data)
    return op
end

"""
    melitz_moment_residuals_under_p(op, p) -> (trade_residuals::Matrix, focal_residual::Float64)

Step 1E core witness primitive: `mul_Gt!(g, op, p)` gives the EXACT moment residual
`E_p[G[:,k]]` for every column at `op`'s CURRENT `(A,f,gpj)` state (production's own
matrix-free adjoint, reused verbatim -- not a hand re-derivation). Requires `sum(p)==1`
for `g[trade_index[o,d]]` to equal the trade-share residual exactly (see `mul_Gt!`'s own
`- lambda[o,d]*sumv` term, moment_operator.jl:314-362).
"""
function melitz_moment_residuals_under_p(op::MelitzMomentOperator, p::AbstractVector{Float64})
    g = zeros(Float64, op.layout.num_moments)
    mul_Gt!(g, op, p)
    D = op.D
    trade_res = zeros(Float64, D, D)
    @inbounds for o in 1:D, d in 1:D
        trade_res[o, d] = g[op.layout.trade_index[o, d]]
    end
    return trade_res, g[op.layout.focal_link_index]
end

"melitz_gravity_A_residual(A, ctx) -> dot(ctx.c_full, vec(log.(A))) -- the A-gravity restriction."
melitz_gravity_A_residual(A::AbstractMatrix{Float64}, ctx) = dot(ctx.c_full, vec(log.(A)))
"melitz_gravity_f_residual(f, ctx) -> dot(ctx.c_full, vec(log.(f))) -- the f/q-gravity restriction."
melitz_gravity_f_residual(f::AbstractMatrix{Float64}, ctx) = dot(ctx.c_full, vec(log.(f)))

# ----------------------------------------------------------------------------------------
# Building block 5: Step 1C corrector.
#
# STRUCTURAL FINDING (verified numerically, `scripts/melitz_aq_phase0_moment_map_2026-07-30.jl`):
# `T_od(p,q_od)` (hence `a_od(new)` via formula (*), hence GravityA/FocalLink) is EXACTLY,
# not merely approximately, constant in `q_od` between two adjacent draws with `p_s>0` in
# origin `o`'s own sorted column -- a genuine step function, never smooth. A first-order
# (Jacobian/FD) Newton corrector is therefore ill-posed almost everywhere (zero local
# slope) except exactly AT a draw crossing. The corrector below is accordingly a BOUNDED
# DISCRETE local search over candidate cutoff positions at/between actual sorted draws of
# the lever cell's own origin -- not a continuous SQP -- per the task's own allowance for
# "a robust local method" and consistent with the audited cliff's own step-function
# mechanism (Phase 3 below measures this directly).
#
# Two lever cells are chosen: `cell_grav` (largest |c_full| among free q cells, excluding
# (j,j)) to correct A-gravity, `cell_focal` (largest |c_full| among free q cells with
# origin==target_country, so it has genuine focal-link leverage through `f`/participation)
# to correct the focal link. Both corrections are routed through `q_free_free` (never the
# q-gravity pivot cell directly) so the FULL q state remains a valid point in the
# `2D^2-2`-free-coordinate parameterization (reconstructible via
# `reduce_to_free_theta_logcutoff` at the end) -- moving either lever's `q_free_free`
# component ALSO moves the (single, distinct) q-gravity-pivot cell via the SAME linear
# pivot formula `expand_free_theta_logcutoff` uses; this is accounted for by re-running
# the FULL cellwise recovery (`build_state`) at every trial, never by a partial/cell-local
# update that would silently ignore the pivot cell's own co-movement.
# ----------------------------------------------------------------------------------------

"""
    melitz_q_free_cell_map(ctx) -> (q_pivot::GravityPivot, cell_of_k::Vector{Int})

`cell_of_k[k]` is the FULL linear cell index (`od2lin`) of `q_free_free[k]`, for
`k=1:length(q_pivot.other)` -- the SAME q-gravity pivot `expand_free_theta_logcutoff` uses
(`build_q_gravity_pivot`), re-queried here (not rebuilt differently) so index bookkeeping
matches production exactly.
"""
function melitz_q_free_cell_map(ctx)
    q_pivot = build_q_gravity_pivot(ctx)
    cell_of_k = [ctx.f_free_lin[m] for m in q_pivot.other]
    return q_pivot, cell_of_k
end

"""
    melitz_candidate_q_positions(o, sorted_ctx, q_current; half_window=40) -> Vector{Float64}

Bounded discrete candidate set for a single cell's cutoff `q_od`, centered on `q_current`:
the midpoints between adjacent DISTINCT `sorted_log_z[:,o]` values within `half_window`
sorted positions of `q_current`'s own current rank (both directions), each candidate
landing strictly inside a chamber (never exactly on a draw). Bounded (`O(half_window)`),
not an exhaustive `O(W)` sweep -- a deliberate scope choice for the corrector's local
search (see module header).
"""
function melitz_candidate_q_positions(o::Int, sorted_ctx::MelitzSortedTailContext,
                                       q_current::Real; half_window::Int=40)
    col = view(sorted_ctx.sorted_log_z, :, o)
    W = length(col)
    k0 = melitz_active_tail_start(col, q_current)  # first active position at q_current
    lo = max(1, k0 - half_window)
    hi = min(W, k0 + half_window)
    positions = Float64[]
    prev = lo == 1 ? (col[1] - 1.0) : col[lo-1]
    for k in lo:hi
        cur = col[k]
        if cur > prev
            push!(positions, 0.5 * (prev + cur))
        end
        prev = cur
    end
    push!(positions, col[hi] + max(1e-9, 0.5 * (hi < W ? col[hi+1] - col[hi] : 1.0)))
    return unique(positions)
end

"""
    MelitzCorrectorTrial

One (cell moved, candidate q, achieved residual) record in the corrector trace.
"""
struct MelitzCorrectorTrial
    round::Int
    target::Symbol   # :gravity or :focal
    cell::Tuple{Int,Int}
    q_trial::Float64
    residual::Float64
end

"""
    melitz_lfd_corrector(build_state, q_free_free0, ctx, sorted_ctx, obj, p_star;
        gravity_tol=1e-6, moment_tol=1e-6, max_rounds=8, half_window=40)
        -> (q_free_free_final, trace::Vector{MelitzCorrectorTrial})

Step 1C: bounded discrete local search (see module header) restoring `A`-gravity and the
focal free-entry link, using `q_free_free` as the correction space. `build_state(qff) ->
(A,f,gpj,q,status)` is the caller's closure wrapping Steps 1A/1B/1D at a trial
`q_free_free`. Alternates (Gauss-Seidel, `max_rounds` rounds): grid-search `cell_grav`'s
`q_free_free` slot to minimize `|GravityA residual|` (cheap, no operator rebuild -- A alone
suffices), then grid-search `cell_focal`'s slot to minimize `|FocalLink residual|` (requires
one operator rebuild per candidate, `melitz_update_operator_at_Afg!`+`mul_Gt!`). Stops early
once both residuals are within tolerance.
"""
function melitz_lfd_corrector(build_state::Function, q_free_free0::Vector{Float64}, ctx,
                               sorted_ctx::MelitzSortedTailContext, obj, p_star::Vector{Float64};
                               gravity_tol::Real=1e-6, moment_tol::Real=1e-6,
                               max_rounds::Int=8, half_window::Int=40)
    D = ctx.D
    j = ctx.target_country
    q_pivot, cell_of_k = melitz_q_free_cell_map(ctx)
    k_for_cell = Dict(cell_of_k[k] => k for k in eachindex(cell_of_k))

    grav_order = sortperm(abs.(ctx.c_full[cell_of_k]); rev=true)
    cell_grav = cell_of_k[grav_order[1]]
    j_cells = [m for m in cell_of_k if lin2od(m, D)[1] == j]
    cell_focal = if !isempty(j_cells)
        j_cells[1] == cell_grav && length(j_cells) > 1 ? j_cells[2] :
            (j_cells[1] == cell_grav ? cell_of_k[grav_order[2]] : j_cells[1])
    else
        cell_of_k[grav_order[2]]
    end
    k_grav = k_for_cell[cell_grav]
    k_focal = k_for_cell[cell_focal]
    o_g, _ = lin2od(cell_grav, D)
    o_f, _ = lin2od(cell_focal, D)

    qff = copy(q_free_free0)
    trace = MelitzCorrectorTrial[]

    function gravity_residual_at(qff_trial::Vector{Float64})
        A_t, _, _, _, status_t = build_state(qff_trial)
        return melitz_gravity_A_residual(A_t, ctx), status_t
    end
    function focal_residual_at(qff_trial::Vector{Float64})
        A_t, f_t, gpj_t, _, status_t = build_state(qff_trial)
        melitz_update_operator_at_Afg!(obj.op, A_t, f_t, gpj_t, ctx)
        g = zeros(Float64, obj.op.layout.num_moments)
        mul_Gt!(g, obj.op, p_star)
        return g[obj.op.layout.focal_link_index], status_t
    end

    gA, _ = gravity_residual_at(qff)
    gF, _ = focal_residual_at(qff)

    for round in 1:max_rounds
        (abs(gA) < gravity_tol && abs(gF) < moment_tol) && break

        cands_g = melitz_candidate_q_positions(o_g, sorted_ctx, qff[k_grav]; half_window=half_window)
        best_g = (abs(gA), qff[k_grav])
        for qc in cands_g
            trial = copy(qff); trial[k_grav] = qc
            r, st = gravity_residual_at(trial)
            all(s -> s == :ok, st) || continue
            push!(trace, MelitzCorrectorTrial(round, :gravity, lin2od(cell_grav, D), qc, r))
            abs(r) < best_g[1] && (best_g = (abs(r), qc))
        end
        qff[k_grav] = best_g[2]
        gA, _ = gravity_residual_at(qff)

        cands_f = melitz_candidate_q_positions(o_f, sorted_ctx, qff[k_focal]; half_window=half_window)
        best_f = (abs(gF), qff[k_focal])
        for qc in cands_f
            trial = copy(qff); trial[k_focal] = qc
            r, st = focal_residual_at(trial)
            all(s -> s == :ok, st) || continue
            push!(trace, MelitzCorrectorTrial(round, :focal, lin2od(cell_focal, D), qc, r))
            abs(r) < best_f[1] && (best_f = (abs(r), qc))
        end
        qff[k_focal] = best_f[2]
        gF, _ = focal_residual_at(qff)
        gA, _ = gravity_residual_at(qff)
    end

    return qff, trace
end

# ----------------------------------------------------------------------------------------
# Top-level Phase 1 constructor.
# ----------------------------------------------------------------------------------------

"""
    MelitzLFDPreservingState

Result of `melitz_construct_lfd_preserving_state`. `A`,`f`,`q` (D x D) and `gamma_prime_j`
(scalar) are the new full economic state; `theta_free` is the SAME state reduced back to a
`:logcutoff` free vector (`reduce_to_free_theta_logcutoff`) for feeding into
`solve_melitz_delta!`/the ordinary typed classifier. `trade_residuals`/`focal_residual`/
`gravity_A_residual`/`gravity_f_residual` are the Step 1E witness residuals evaluated under
the UNCHANGED anchor LFD `p_star`. `divergence_pstar = melitz_primal_divergence(p_star,W)`.
`feasible` is the Step 1E pass/fail gate. `A_status` flags any `:zero_tail_*` cell (Phase 0
header) -- `feasible=false` whenever any cell is not `:ok`.
"""
struct MelitzLFDPreservingState
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
    corrector_trace::Vector{MelitzCorrectorTrial}
    feasible::Bool
end

"""
    melitz_construct_lfd_preserving_state(theta_anchor, p_star, ctx, obj, g_new, q_free_free_new;
        moment_tol=1e-6, gravity_tol=1e-6, correct=true, max_corrector_rounds=8,
        corrector_half_window=40) -> MelitzLFDPreservingState

Governing prompt Phase 1: builds a new full Melitz economic state at free-q coordinates
`q_free_free_new` and welfare coordinate `g_new = log(gamma_prime_j_new)` such that the
ANCHOR's own verified LFD `p_star` (unchanged) exactly satisfies every bilateral
trade-share moment (Step 1B, exact closed-form -- see module header formula (*)), then, if
`correct=true`, runs the discrete corrector (Step 1C, `melitz_lfd_corrector`) restoring
A-gravity and the focal free-entry link. `correct=false` returns the Step 1B
"cellwise-compensated, uncorrected" endpoint directly (Phase 2's endpoint B).

`theta_anchor` must be the `:logcutoff`-mode free theta at the anchor (`ctx.outer_parameterization
== :logcutoff`), `p_star` the anchor's own recovered LFD weights (`melitz_recover_lfd(obj,
theta_anchor).weights`), `obj`/`ctx` the SAME bundle/context the anchor was solved under
(`obj.op` is mutated as scratch by the corrector and by the final witness build -- restore
it via `melitz_update_operator_at_theta!(obj.op, theta_anchor, ctx)` afterward if the
caller needs `obj.op` to reflect the anchor again).
"""
function melitz_construct_lfd_preserving_state(theta_anchor::AbstractVector, p_star::Vector{Float64},
        ctx, obj, g_new::Real, q_free_free_new::AbstractVector{Float64};
        moment_tol::Real=1e-6, gravity_tol::Real=1e-6, correct::Bool=true,
        max_corrector_rounds::Int=8, corrector_half_window::Int=40)

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

    q_free_free_final = collect(Float64, q_free_free_new)
    trace = MelitzCorrectorTrial[]
    if correct
        q_free_free_final, trace = melitz_lfd_corrector(build_state, q_free_free_final, ctx,
            sorted_ctx, obj, p_star; gravity_tol=gravity_tol, moment_tol=moment_tol,
            max_rounds=max_corrector_rounds, half_window=corrector_half_window)
    end

    A_new, f_new, gpj_new, q_new, status_new = build_state(q_free_free_final)

    theta_free_recovered = reduce_to_free_theta_logcutoff(A_new, f_new, gpj_new, ctx)
    melitz_update_operator_at_Afg!(obj.op, A_new, f_new, gpj_new, ctx)
    trade_res, focal_res = melitz_moment_residuals_under_p(obj.op, p_star)
    gA = melitz_gravity_A_residual(A_new, ctx)
    gF = melitz_gravity_f_residual(f_new, ctx)
    W = length(p_star)
    divp = melitz_primal_divergence(p_star, W)

    all_ok = all(s -> s == :ok, status_new)
    feasible = all_ok && maximum(abs.(trade_res)) < moment_tol && abs(focal_res) < moment_tol &&
               abs(gA) < gravity_tol && abs(gF) < gravity_tol &&
               all(p_star .>= 0) && isapprox(sum(p_star), 1.0; atol=1e-8)

    return MelitzLFDPreservingState(A_new, f_new, q_new, gpj_new, theta_free_recovered,
        status_new, trade_res, focal_res, gA, gF, divp, trace, feasible)
end
