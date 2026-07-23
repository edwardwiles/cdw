# Session prompt (2026-07-22) Section 5: the outer-gradient laboratory for Delta(theta).
#
# Do NOT differentiate through KNITRO. Use the envelope/fixed-dual criterion at the
# optimized inner dual (session prompt's own instruction, matching this repo's
# production/fullA-exact methodology -- see fullA_independent_assessor_brief_2026-07-22,
# Section 5.2's "fixed-dual finite-bandwidth secant" for the analogous production
# construction this module mirrors).
#
# Implements Method A (fully reoptimized central finite difference, the expensive
# reference), Method B (fixed-dual finite-bandwidth secant), and Method C (ForwardDiff
# fixed-active-set envelope derivative). Method D (hand-derived analytic branch
# derivative) and Method E (smooth surrogate) are NOT implemented this session --
# documented as out of scope in the final report, not silently skipped.

using ForwardDiff
using LinearAlgebra: dot

"""
    melitz_fixed_dual_criterion(theta_free, x_fixed, ctx, obj) -> Float64

The envelope/fixed-dual criterion: recomputes the HARD participation-gated moments
`G(theta_free)` (via `obj.moments!`, writing into `obj.H`) but evaluates the CC dual
objective at a FIXED `x_fixed` (not re-optimized) via `PsiObjectiveBundleDelta`'s own
functor `obj(x)`. Sign-consistent with `MelitzLFDResult.Delta` (`obj.find_smallest`
applied identically to `inner_loop`'s own convention). This is the SAME construction as
`production/fullA-exact`'s `L_fix` (assessor brief Section 5.2), specialized to the
Melitz moment adapter.

Mutates `obj.H` (shared scratch) -- safe for SEQUENTIAL reuse of the same `obj` (this
session's own Section 2 instruction: serial evaluation only, no parallel callbacks), NOT
safe to call concurrently on the same `obj` from multiple threads (the AUD-02 callback
guard in `PsiObjectiveBundle.jl` will error on that, by design).
"""
function melitz_fixed_dual_criterion(theta_free::AbstractVector, x_fixed::AbstractVector, ctx, obj)
    obj.moments!(@view(obj.H[:, 1]), CounterfactualSensitivity.select_G_from_H(obj, obj.H), theta_free, obj.U, obj)
    obj.H[:, 2] .= 1.0
    raw = obj(x_fixed)
    return obj.find_smallest ? -raw : raw
end

"""
    GradientProbeResult

One directional derivative probe's full record (session prompt Section 6: "for every
displaced evaluation, record cutoff feasibility, switch counts, inner status, every
method's derivative estimate, predicted vs actual Delta change, runtime").
"""
struct GradientProbeResult
    method::Symbol                  # :A, :B, :C
    point_label::String
    direction_label::String
    h::Float64
    deriv::Float64
    feasible_plus::Bool
    feasible_minus::Bool
    n_switches::Union{Nothing,Int}  # number of (o,d,draw) participation flips vs the base point, if computable
    nStatus_plus::Union{Nothing,Int}
    nStatus_minus::Union{Nothing,Int}
    predicted_delta_change::Float64  # deriv * 2h (over the FULL central-difference span)
    actual_delta_change::Union{Nothing,Float64}  # only available if Method A was ALSO run at this (point, h, v)
    wall_time::Float64
    allocations::Int64
end

"""
    method_a_reoptimized_fd(theta, v, h, ctx, obj; cold=true) -> GradientProbeResult

Method A (session prompt Section 5): fully reoptimized central finite difference,
`[Delta(theta+h*v) - Delta(theta-h*v)] / (2h)`, with an INDEPENDENTLY optimized inner
problem at both displaced points (`cold=true` default -- an independent reference should
not inherit a warm start biased toward the base point). The expensive ground-truth
method.
"""
function method_a_reoptimized_fd(theta::AbstractVector, v::AbstractVector, h::Real, ctx, obj;
                                  cold::Bool=true, point_label::String="", direction_label::String="")
    t0 = time()
    theta_p = theta .+ h .* v
    theta_m = theta .- h .* v
    eval_p = evaluate_melitz_delta(theta_p, ctx, obj; cold=cold, store_G=false)
    eval_m = evaluate_melitz_delta(theta_m, ctx, obj; cold=cold, store_G=false)
    deriv = (eval_p.Delta - eval_m.Delta) / (2h)
    wall = time() - t0
    return GradientProbeResult(:A, point_label, direction_label, h, deriv,
        eval_p.feasible, eval_m.feasible, nothing, eval_p.nStatus, eval_m.nStatus,
        deriv * 2h, eval_p.Delta - eval_m.Delta, wall, 0)
end

"""
    method_b_fixed_dual_secant(theta, v, h, x_base, ctx, obj) -> GradientProbeResult

Method B (session prompt Section 5): the fixed-dual finite-bandwidth secant,
`g_search(theta;h) = [L_fix(theta+h*v; x_base) - L_fix(theta-h*v; x_base)] / (2h)`.
`x_base` is the OPTIMIZED dual at the BASE point (solved ONCE, held fixed across every
direction/bandwidth probed from that base). NOT the exact derivative of the finite-W
hard-winner value function -- a finite-bandwidth winner-switching secant (session
prompt's own caveat). Cheap: two `moments!` calls, no KNITRO solve, per probe.
"""
function method_b_fixed_dual_secant(theta::AbstractVector, v::AbstractVector, h::Real,
                                     x_base::AbstractVector, ctx, obj;
                                     point_label::String="", direction_label::String="")
    t0 = time()
    theta_p = theta .+ h .* v
    theta_m = theta .- h .* v
    Lp = melitz_fixed_dual_criterion(theta_p, x_base, ctx, obj)
    Lm = melitz_fixed_dual_criterion(theta_m, x_base, ctx, obj)
    deriv = (Lp - Lm) / (2h)
    wall = time() - t0

    g_dp, g_ep = melitz_cutoff_constraints_at(theta_p, ctx)
    g_dm, g_em = melitz_cutoff_constraints_at(theta_m, ctx)
    feasible_p = minimum(g_dp) >= 0 && minimum(g_ep) >= 0
    feasible_m = minimum(g_dm) >= 0 && minimum(g_em) >= 0

    return GradientProbeResult(:B, point_label, direction_label, h, deriv,
        feasible_p, feasible_m, nothing, nothing, nothing, deriv * 2h, nothing, wall, 0)
end

"""
    melitz_fixed_active_set_scalar(theta_free, active_mask, x_base, ctx, obj_template) -> scalar

Session prompt Section 5, Method C: a type-generic (ForwardDiff-Dual-compatible) scalar
evaluator of the fixed-dual, FIXED-ACTIVE-SET criterion. `active_mask` is a `Bool` array
(D x D x W, FIXED at the base point -- captured OUTSIDE this function, never recomputed
from a Dual-typed `theta_free`) -- every draw's participation decision is held fixed
regardless of the displaced `theta_free`; only the SMOOTH price/revenue/profit algebra
inside `melitz_firm` is differentiated. Does NOT mutate `obj_template.H` (Dual numbers
cannot be written into `obj_template`'s `Float64` buffers, per this session's own
Section 5 warning that `PsiObjectiveBundleDelta`'s preallocated Float64 storage is not
Dual-safe) -- builds fresh Dual-typed `K`/`G` arrays locally instead, then reproduces the
`Psi!`/inner-product objective algebra directly (mirroring
`PsiObjectiveBundleDelta`'s own functor, `cc_algo/PsiObjectiveBundle.jl`) rather than
calling `obj_template(x)`.

Intentionally MISSES cutoff crossings (the participation gate is held fixed) -- expected
to agree with Method B/D near-exactly on ZERO-SWITCH probes and to disagree substantially
whenever a probe direction/bandwidth would flip cells' participation (session prompt
Section 6, core question 1).
"""
function melitz_fixed_active_set_scalar(theta_free::AbstractVector{T}, active_mask::AbstractArray{Bool,3},
                                         x_base::AbstractVector{Float64}, ctx, obj_template) where {T}
    D, j = ctx.D, ctx.target_country
    W = size(obj_template.U, 1)
    sigma = ctx.sigma

    A, f, gamma_prime_j, f_jj = expand_free_theta(theta_free, ctx)

    G = zeros(T, W, ctx.moment_layout.num_moments)
    layout = ctx.moment_layout
    price_power_autarky = gamma_prime_j
    profit_j = zeros(T, W)

    @inbounds for o in 1:D, d in 1:D
        trade_col = layout.trade_index[o, d]
        lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj_template.U[w, o]
            active = active_mask[o, d, w]  # FIXED, never recomputed from theta_free
            price = melitz_price(ctx.w[o], ctx.tau[o, d], A[o, d], sigma, z)
            rev = unconstrained_revenue(price, sigma, ctx.expenditure[d], 1.0)
            realized_rev = active ? rev : zero(rev)
            G[w, trade_col] = realized_rev / ctx.expenditure[d] - lambda_od
            if o == j
                profit = rev / sigma - ctx.w[o] * f[o, d]
                profit_j[w] += active ? profit : zero(profit)
            end
        end
    end

    link_col = layout.focal_link_index
    @inbounds for w in 1:W
        z_j = obj_template.U[w, j]
        active_auk = active_mask[j, j, w]
        price_auk = melitz_price(1.0, 1.0, A[j, j], sigma, z_j)
        rev_auk = unconstrained_revenue(price_auk, sigma, ctx.w_prime * ctx.L[j], price_power_autarky)
        profit_auk = rev_auk / sigma - 1.0 * f_jj
        realized_profit_auk = active_auk ? profit_auk : zero(profit_auk)
        G[w, link_col] = profit_j[w] / ctx.w[j] - realized_profit_auk / ctx.w_prime
    end

    # Reproduce PsiObjectiveBundleDelta's own dual-objective algebra (functor, main
    # prompt's own warning against calling obj_template(x) with Dual x -- build it here).
    d = layout.num_moments
    arg0 = zeros(T, W)
    @inbounds for w in 1:W
        arg0[w] = -x_base[1] - dot(view(G, w, :), view(x_base, 2:length(x_base)))
    end
    psi = zeros(T, W)
    e = exp(1.0)
    @inbounds for w in 1:W
        psi[w] = arg0[w] <= 1.0 ? exp(arg0[w]) - 1.0 : 0.5 * e * (arg0[w]^2 + 1.0) - 1.0
    end
    raw = sum(psi) / W + x_base[1]
    return obj_template.find_smallest ? -raw : raw
end

"""
    method_c_forwarddiff_envelope(theta, v, h, x_base, active_mask, ctx, obj) -> GradientProbeResult

Method C (session prompt Section 5): `ForwardDiff.derivative` of
`melitz_fixed_active_set_scalar` along direction `v`, evaluated at `theta` -- an EXACT
derivative of the fixed-active-set, fixed-dual criterion (not a finite-difference secant;
`h` is accepted only so the SAME probe-result schema/report can compare it against
Methods A/B's displaced evaluations, not used inside the ForwardDiff call itself).
"""
function method_c_forwarddiff_envelope(theta::AbstractVector, v::AbstractVector, h::Real,
                                        x_base::AbstractVector, active_mask::AbstractArray{Bool,3},
                                        ctx, obj; point_label::String="", direction_label::String="")
    t0 = time()
    scalar_fn(t) = melitz_fixed_active_set_scalar(theta .+ t .* v, active_mask, x_base, ctx, obj)
    deriv = ForwardDiff.derivative(scalar_fn, 0.0)
    wall = time() - t0

    theta_p = theta .+ h .* v
    theta_m = theta .- h .* v
    g_dp, g_ep = melitz_cutoff_constraints_at(theta_p, ctx)
    g_dm, g_em = melitz_cutoff_constraints_at(theta_m, ctx)
    feasible_p = minimum(g_dp) >= 0 && minimum(g_ep) >= 0
    feasible_m = minimum(g_dm) >= 0 && minimum(g_em) >= 0

    return GradientProbeResult(:C, point_label, direction_label, h, deriv,
        feasible_p, feasible_m, nothing, nothing, nothing, deriv * 2h, nothing, wall, 0)
end

"""
    base_active_mask(theta_base, ctx, obj) -> BitArray{3} (D x D x W)

The participation decision `profit_od(z) > 0` at every `(o,d,draw)` triple, evaluated
ONCE at the base point -- the FIXED active set Methods C (and, if implemented, D) hold
constant while displacing `theta_free`.
"""
function base_active_mask(theta_base::AbstractVector, ctx, obj)
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    A, f, gamma_prime_j, f_jj = expand_free_theta(theta_base, ctx)
    mask = falses(D, D, W)
    @inbounds for o in 1:D, d in 1:D
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma, ctx.expenditure[d], 1.0, z)
            mask[o, d, w] = firm.active
        end
    end
    # autarky cell (j,j) reuses mask[j,j,:] slot's own draw index j (distinct economic
    # meaning from the baseline (j,j) trade cell, but indexed identically since both use
    # z_draws[:,j] -- callers needing the autarky mask separately should recompute it;
    # melitz_fixed_active_set_scalar above uses mask[j,j,w] for BOTH, an approximation
    # flagged here for the report (baseline domestic and autarky participation are
    # generally different decisions at the same draw).
    return mask
end

"""
    count_switches(mask_base, theta_probe, ctx, obj) -> (n_switches, per_cell)

Diagnostic (session prompt Section 6): compares the FIXED base active mask against the
TRUE active mask recomputed at a displaced `theta_probe`, returning the total number of
`(o,d,draw)` triples whose participation flipped and a `D x D` per-cell breakdown.
"""
function count_switches(mask_base::AbstractArray{Bool,3}, theta_probe::AbstractVector, ctx, obj)
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    A, f, gamma_prime_j, f_jj = expand_free_theta(theta_probe, ctx)
    per_cell = zeros(Int, D, D)
    total = 0
    @inbounds for o in 1:D, d in 1:D
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma, ctx.expenditure[d], 1.0, z)
            if firm.active != mask_base[o, d, w]
                per_cell[o, d] += 1
                total += 1
            end
        end
    end
    return total, per_cell
end
