# Session prompt (2026-07-22) Section 5: the outer-gradient laboratory for Delta(theta).
#
# Do NOT differentiate through KNITRO. Use the envelope/fixed-dual criterion at the
# optimized inner dual (session prompt's own instruction, matching this repo's
# production/fullA-exact methodology -- the analogous production "fixed-dual
# finite-bandwidth secant" construction this module mirrors, [[method-b-eliminates-dense-
# jacobian]]).
#
# 2026-07-23 governing-correction session, Section 1: Methods B and C now call ONE
# authoritative `fixed_dual_scalar` (no separate duplicate implementation, per that
# session's own instruction) -- see its docstring for the zero-switch discrepancy this
# replaces and fixes. Implements Method A (fully reoptimized central finite difference,
# the expensive reference), Method B (fixed-dual finite-bandwidth secant), Method C
# (ForwardDiff exact derivative of the SAME fixed-dual-scalar chain), and Method D
# (hand-derived closed-form branch derivative, an independently-coded cross-check of
# Method C). Method E (smooth surrogate) is NOT implemented -- out of scope, not silently
# skipped.

using ForwardDiff
using LinearAlgebra: dot

"""
    _fill_fixed_active_set_moments!(G, profit_j, theta_free, ctx, obj) -> G

Section 12.4/7.5 optimization: the shared, type-generic (Float64 or ForwardDiff.Dual) fill
body for `fixed_active_set_moments`/`fixed_active_set_moments!` -- takes caller-owned
`G`/`profit_j` buffers rather than allocating its own, so the allocating (`T`-generic,
used by Method C's ForwardDiff path) and in-place (`Float64`-only, used by Method B's
coordinate-probe loop) entry points below share ONE implementation instead of duplicating
the moment-construction logic a second time.
"""
function _fill_fixed_active_set_moments!(G::AbstractMatrix{T}, profit_j::AbstractVector{T},
                                          theta_free::AbstractVector, ctx, obj) where {T}
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    layout = ctx.moment_layout

    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_free, ctx)

    fill!(G, zero(T))
    fill!(profit_j, zero(T))
    price_power_d = 1.0
    @inbounds for o in 1:D, d in 1:D
        trade_col = layout.trade_index[o, d]
        lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma,
                                ctx.expenditure[d], price_power_d, z)
            G[w, trade_col] = firm.realized_revenue / ctx.expenditure[d] - lambda_od
            o == j && (profit_j[w] += firm.realized_operating_profit)
        end
    end

    link_col = layout.focal_link_index
    expenditure_prime = ctx.w_prime * ctx.L[j]
    price_power_autarky = gamma_prime_j
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_autarky = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma,
                                    expenditure_prime, price_power_autarky, z_j)
        G[w, link_col] = profit_j[w] / ctx.w[j] - firm_autarky.realized_operating_profit / ctx.w_prime
    end
    return G
end

function fixed_active_set_moments(theta_free::AbstractVector{T}, ctx, obj) where {T}
    W = size(obj.U, 1)
    layout = ctx.moment_layout
    G = zeros(T, W, layout.num_moments)
    profit_j = zeros(T, W)
    return _fill_fixed_active_set_moments!(G, profit_j, theta_free, ctx, obj)
end

"""
    fixed_active_set_moments!(G, profit_j, theta_free, ctx, obj) -> G

Section 12.4/7.5 optimization: in-place `Float64`-only variant of
`fixed_active_set_moments`, for callers that already own a persistent `(W x num_moments)`
buffer `G` and length-`W` scratch `profit_j` and want to avoid allocating a fresh matrix
on every call -- e.g. `finite_delta_outer.jl`'s `make_melitz_moments_jacobian_b`, whose
coordinate-probe loop calls this `2*n` times per outer gradient callback (the governing
prompt's own Section 7.5 concern: "up to 2*30=60 full displaced moment builds per
gradient"). ForwardDiff/Method C callers must keep using the allocating
`fixed_active_set_moments` (their `G` is `Dual`-typed and cannot share this `Float64`
buffer). Bit-identical output to `fixed_active_set_moments` at the same input (shared fill
body) -- validated in the test suite ("Section 12.4: fixed_active_set_moments! (in-place)
matches the allocating reference").
"""
function fixed_active_set_moments!(G::AbstractMatrix{Float64}, profit_j::AbstractVector{Float64},
                                    theta_free::AbstractVector{Float64}, ctx, obj)
    return _fill_fixed_active_set_moments!(G, profit_j, theta_free, ctx, obj)
end

"""
    dual_scalar_at_fixed_G(G, x_base, obj) -> scalar

Section 1.1 step 5: the exact CC dual scalar evaluated at a FIXED dual `x_base` given an
already-built moment matrix `G` -- reproduces `PsiObjectiveBundleDelta`'s own functor
algebra directly (never calls `obj(x)`, never mutates `obj.H`).
"""
function dual_scalar_at_fixed_G(G::AbstractMatrix{T}, x_base::AbstractVector{Float64}, obj) where {T}
    W = size(G, 1)
    arg0 = zeros(T, W)
    @inbounds for w in 1:W
        arg0[w] = -x_base[1] - dot(view(G, w, :), view(x_base, 2:length(x_base)))
    end
    e = exp(1.0)
    psi = zeros(T, W)
    @inbounds for w in 1:W
        psi[w] = arg0[w] <= 1.0 ? exp(arg0[w]) - 1.0 : 0.5 * e * (arg0[w]^2 + 1.0) - 1.0
    end
    raw = sum(psi) / W + x_base[1]
    return obj.find_smallest ? -raw : raw
end

"""
    fixed_dual_scalar(theta_free, x_base, ctx, obj) -> scalar

THE single authoritative fixed-dual criterion (2026-07-23 governing-correction session,
Section 1.1): `dual_scalar_at_fixed_G(fixed_active_set_moments(theta_free, ctx, obj),
x_base, obj)`. Both Method B (`method_b_fixed_dual_secant`, central finite difference in
`Float64`) and Method C (`method_c_forwarddiff_envelope`, `ForwardDiff.derivative`) call
EXACTLY this function -- the former separate `melitz_fixed_dual_criterion`/
`melitz_fixed_active_set_scalar` pair is retired.

**The bug this replaces**: the former Method C (`melitz_fixed_active_set_scalar`) held
participation fixed via a PRECOMPUTED `active_mask` array (`base_active_mask`), whose
`(j,j)` autarky slot incorrectly REUSED the baseline `(j,j)` domestic decision (computed
with `price_power=1`) as a proxy for the autarky decision (which needs `price_power=
gamma_prime_j`) -- two different economic gates that generally disagree. Method B was
already correct (it called the real `obj.moments!`, which has always used the right
autarky formula). This explains a zero-switch B/C discrepancy: `count_switches`'s own
diagnostic only ever recomputed the BASELINE-formula mask, so it could report "zero
switches" while the (never separately checked) autarky decision had in fact flipped --
Method B, using the true autarky formula, then correctly captured that jump while the old
Method C's frozen (and wrong-formula) proxy could not.

Evaluating this function through `ForwardDiff.derivative(a -> fixed_dual_scalar(theta .+
a.*v, x_base, ctx, obj), 0.0)` (Method C) freezes the active set EXACTLY at `theta`'s own
CORRECTLY-recomputed participation decision, per cell, with no separate explicit mask
needed at all: ForwardDiff's `Dual` comparisons (`profit > 0` inside `melitz_firm`)
resolve on the `Dual`'s VALUE component, which at `a=0` equals `theta` itself.
"""
function fixed_dual_scalar(theta_free::AbstractVector{T}, x_base::AbstractVector{Float64},
                            ctx, obj) where {T}
    G = fixed_active_set_moments(theta_free, ctx, obj)
    return dual_scalar_at_fixed_G(G, x_base, obj)
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
    Lp = fixed_dual_scalar(theta_p, x_base, ctx, obj)
    Lm = fixed_dual_scalar(theta_m, x_base, ctx, obj)
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
    method_c_forwarddiff_envelope(theta, v, h, x_base, ctx, obj) -> GradientProbeResult

Method C (2026-07-23 governing-correction session, Section 1.1): `ForwardDiff.derivative`
of `fixed_dual_scalar` along direction `v`, evaluated at `theta` -- an EXACT derivative of
the fixed-active-set, fixed-dual criterion (not a finite-difference secant; `h` is
accepted only so the SAME probe-result schema/report can compare it against Methods A/B's
displaced evaluations, not used inside the ForwardDiff call itself). No `active_mask`
argument is needed (formerly required, now removed): `fixed_dual_scalar` recomputes
participation fresh, type-generically, from whatever `theta_free` (`Float64` or `Dual`) it
is given, which at `a=0` is exactly `theta`'s own decision.
"""
function method_c_forwarddiff_envelope(theta::AbstractVector, v::AbstractVector, h::Real,
                                        x_base::AbstractVector, ctx, obj;
                                        point_label::String="", direction_label::String="")
    t0 = time()
    scalar_fn(t) = fixed_dual_scalar(theta .+ t .* v, x_base, ctx, obj)
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
    MelitzActiveSetSnapshot

Session prompt Section 1.6 bug fix companion: the baseline `(o,d)` participation decision
(`D x D x W`, `price_power=1` always -- correct for every cell, INCLUDING `(j,j)`'s own
baseline-domestic decision) and the FOCAL AUTARKY participation decision (`W`-vector,
`price_power=gamma_prime_j`, `tau=1`, wage `w_prime`, expenditure `expenditure_prime`)
tracked SEPARATELY. Before the Section 1.1 fix, the switch-count diagnostic (like the old
Method C) only ever checked the baseline `(j,j)` decision -- so a probe could be reported
as "zero switches" while the (never independently checked) autarky decision had actually
flipped. `count_switches` below now checks both.
"""
struct MelitzActiveSetSnapshot
    baseline::BitArray{3}
    autarky::BitVector
end

"""
    base_active_mask(theta_base, ctx, obj) -> MelitzActiveSetSnapshot

The participation decision `profit_od(z) > 0` at every `(o,d,draw)` triple (baseline
formula, `price_power=1`), AND the SEPARATE focal autarky participation decision
(`price_power=gamma_prime_j`), both evaluated ONCE at the base point.
"""
function base_active_mask(theta_base::AbstractVector, ctx, obj)
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_base, ctx)
    mask = falses(D, D, W)
    @inbounds for o in 1:D, d in 1:D
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma, ctx.expenditure[d], 1.0, z)
            mask[o, d, w] = firm.active
        end
    end
    expenditure_prime = ctx.w_prime * ctx.L[j]
    autarky = falses(W)
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_auk = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, gamma_prime_j, z_j)
        autarky[w] = firm_auk.active
    end
    return MelitzActiveSetSnapshot(mask, autarky)
end

"""
    count_switches(snap, theta_probe, ctx, obj) -> (n_switches, per_cell, n_autarky_switches)

Diagnostic (session prompt Section 6, fixed Section 1.6): compares the FIXED base active
snapshot against the TRUE active decisions recomputed at a displaced `theta_probe`,
returning the total number of `(o,d,draw)` triples whose BASELINE participation flipped
(`per_cell`, `D x D` breakdown), plus the number of draws whose separately-tracked AUTARKY
participation flipped (`n_autarky_switches`, folded into the returned `n_switches` total).
A probe is genuinely zero-switch only when BOTH are zero -- checking only the baseline
mask (the pre-fix behavior) can miss autarky-side flips entirely.
"""
function count_switches(snap::MelitzActiveSetSnapshot, theta_probe::AbstractVector, ctx, obj)
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_probe, ctx)
    per_cell = zeros(Int, D, D)
    total = 0
    @inbounds for o in 1:D, d in 1:D
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma, ctx.expenditure[d], 1.0, z)
            if firm.active != snap.baseline[o, d, w]
                per_cell[o, d] += 1
                total += 1
            end
        end
    end
    expenditure_prime = ctx.w_prime * ctx.L[j]
    autarky_switches = 0
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_auk = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, gamma_prime_j, z_j)
        if firm_auk.active != snap.autarky[w]
            autarky_switches += 1
        end
    end
    total += autarky_switches
    return total, per_cell, autarky_switches
end

# ============================================================================
# Section 1.5: Method D, a hand-derived, INDEPENDENTLY-CODED cross-check of Method C.
# ============================================================================

"""
    expand_theta_econ_vector(theta_free, ctx) -> Vector

`vcat(vec(A), vec(f), gamma_prime_j, f_jj)` -- `expand_free_theta`'s four outputs packed
into one vector purely so `ForwardDiff.derivative` (which needs a single array-valued
function) can differentiate the gravity-pivot/autarky-cutoff chain in one call. Used ONLY
by Method D to get `d(A,f,gamma_prime_j,f_jj)/dtheta_free` along a direction -- Method D
does NOT use ForwardDiff for the firm-level revenue/profit algebra itself (that is hand-
derived below, in closed form), unlike Method C, which differentiates the ENTIRE chain
(pivot expansion AND firm algebra) in one `ForwardDiff.derivative` call on
`fixed_dual_scalar`. This makes Method D a genuinely different code path, even though both
should give the identical answer at a fixed active set (both are exact derivatives of the
same smooth composition) -- agreement is the cross-check, not a foregone conclusion (a bug
in either `expand_free_theta`'s chain rule or `melitz_firm`'s hand-derived partials would
show up as a Method C/D disagreement even where both individually look self-consistent).
"""
function expand_theta_econ_vector(theta_free::AbstractVector{T}, ctx) where {T}
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_free, ctx)
    D = ctx.D
    return vcat(vec(A), vec(f), gamma_prime_j, f_jj)
end

"""
    method_d_hand_derived(theta, v, h, x_base, ctx, obj) -> GradientProbeResult

Method D (session prompt Section 1.5): the closed-form active-set-conditional
derivatives given in the governing prompt --

    d share_od / d logA_od = (sigma-1)*share_od          (share_od = price_od^(1-sigma))
    d share_od / d logf_od = 0
    d pi_od    / d logA_od = (sigma-1)*revenue_od/sigma
    d pi_od    / d logf_od = -w_o*f_od

applied at the BASE point's own (fixed) active set for every `(o,d)` trade cell, and,
SEPARATELY, for the focal autarky cell (its own `price_power_autarky=gamma_prime_j`,
`w_prime`, `expenditure_prime` inputs -- structurally different from the baseline `(j,j)`
cell, the same distinction Section 1.1's bug fix turned on). Assembled into
`d(fixed_dual_scalar)/dtheta_free` along direction `v` via:

    d(scalar)/d(arg0[w]) = dPsi/darg0[w]           (closed form: Psi'(a) = a<=1 ? exp(a) : e*a)
    d(arg0[w])/dG[w,:]   = -x_base[2:end]
    d(scalar)/dv         = sum_w dPsi/darg0[w] * ( -dot(x_base[2:end], dG[w,:]/dv) )

where `dG[w,:]/dv` uses the closed-form firm-level partials above, chained through
`d(A,f,gamma_prime_j,f_jj)/dv` obtained from `ForwardDiff.derivative` on
`expand_theta_econ_vector` ALONE (the gravity-pivot/autarky-cutoff chain, already
independently validated against central finite differences to `~1e-10` in Section 1.3 --
reused here rather than re-derived a third time by hand, since that chain is linear/
near-linear and not the subject of the zero-switch investigation). `f_jj`'s directional
derivative (`dfjj` below) is therefore the FULL total derivative already (including its
own dependence on `A[j,j]` and `gamma_prime_j` via `derive_fjj_from_autarky_cutoff`) --
do not add a separate correction for that, or it double-counts.
"""
function method_d_hand_derived(theta::AbstractVector, v::AbstractVector, h::Real,
                                x_base::AbstractVector, ctx, obj;
                                point_label::String="", direction_label::String="")
    t0 = time()
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    layout = ctx.moment_layout

    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    dvec = ForwardDiff.derivative(a -> expand_theta_econ_vector(theta .+ a .* v, ctx), 0.0)
    D2 = D * D
    dA = reshape(dvec[1:D2], D, D)
    df = reshape(dvec[D2+1:2*D2], D, D)
    dgamma = dvec[2*D2+1]
    dfjj = dvec[2*D2+2]

    expenditure_prime = ctx.w_prime * ctx.L[j]
    price_power_autarky = gamma_prime_j

    G = zeros(Float64, W, layout.num_moments)
    dG = zeros(Float64, W, layout.num_moments)
    profit_j = zeros(Float64, W)
    dprofit_j = zeros(Float64, W)

    @inbounds for o in 1:D, d in 1:D
        trade_col = layout.trade_index[o, d]
        lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma, ctx.expenditure[d], 1.0, z)
            G[w, trade_col] = firm.realized_revenue / ctx.expenditure[d] - lambda_od
            if firm.active
                dG[w, trade_col] = (sigma - 1) * (firm.unconstrained_revenue / ctx.expenditure[d]) / A[o, d] * dA[o, d]
            end
            if o == j
                profit_j[w] += firm.realized_operating_profit
                if firm.active
                    dprofit_j[w] += (sigma - 1) * firm.unconstrained_revenue / sigma / A[o, d] * dA[o, d] -
                                     ctx.w[o] * df[o, d]
                end
            end
        end
    end

    link_col = layout.focal_link_index
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_auk = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, price_power_autarky, z_j)
        G[w, link_col] = profit_j[w] / ctx.w[j] - firm_auk.realized_operating_profit / ctx.w_prime
        dpi_auk = 0.0
        if firm_auk.active
            dpi_auk = (sigma - 1) * firm_auk.unconstrained_revenue / sigma / A[j, j] * dA[j, j] -
                      firm_auk.unconstrained_revenue / sigma / price_power_autarky * dgamma -
                      ctx.w_prime * dfjj
        end
        dG[w, link_col] = dprofit_j[w] / ctx.w[j] - dpi_auk / ctx.w_prime
    end

    e = exp(1.0)
    total = 0.0
    @inbounds for w in 1:W
        arg0 = -x_base[1] - dot(view(G, w, :), view(x_base, 2:length(x_base)))
        dPsi_darg0 = arg0 <= 1.0 ? exp(arg0) : e * arg0
        darg0 = -dot(view(dG, w, :), view(x_base, 2:length(x_base)))
        total += dPsi_darg0 * darg0
    end
    deriv = total / W
    deriv = obj.find_smallest ? -deriv : deriv
    wall = time() - t0

    theta_p = theta .+ h .* v
    theta_m = theta .- h .* v
    g_dp, g_ep = melitz_cutoff_constraints_at(theta_p, ctx)
    g_dm, g_em = melitz_cutoff_constraints_at(theta_m, ctx)
    feasible_p = minimum(g_dp) >= 0 && minimum(g_ep) >= 0
    feasible_m = minimum(g_dm) >= 0 && minimum(g_em) >= 0

    return GradientProbeResult(:D, point_label, direction_label, h, deriv,
        feasible_p, feasible_m, nothing, nothing, nothing, deriv * 2h, nothing, wall, 0)
end

# ============================================================================
# Section 1.6: a genuinely high-switch-inducing f direction (fixes the Gate B session's
# `f_high_switch`, which accidentally duplicated `ordinary_f`).
# ============================================================================

"""
    f_high_switch_direction(theta_base, ctx, obj; test_h=1e-3) -> (v, coord_index, n_switches)

Session prompt Section 1.6: constructs a genuinely high-switch-inducing `f` direction,
replacing the prior Gate B session's `f_high_switch`, which accidentally duplicated
`ordinary_f` (the SAME coordinate index -- confirmed by identical results at every
bandwidth in that session's own CSV, docs/melitz_delta_star.md Section 15.7). For every
free `f` coordinate in `theta_free` (positions `2+(D^2-1) : end`, per `expand_free_theta`'s
own layout: `theta_free = vcat(log gamma_prime_j, A_free (D^2-1), f_free_free (D^2-2))`),
probes a unit bump of size `test_h` and counts participation switches (`count_switches`,
including the separately-tracked autarky decision, Section 1.6's own companion fix)
against a shared base active-set snapshot; returns the coordinate producing the MOST
switches (ties broken by lowest index) as a unit direction vector, its `theta_free` index,
and its switch count -- so a caller can directly verify it exceeds `ordinary_f`'s own
switch count at a matched `h` (the first free-f coordinate, `theta_free` index `2+D^2-1`,
by convention).
"""
function f_high_switch_direction(theta_base::AbstractVector, ctx, obj; test_h::Real=1e-3)
    n = length(theta_base)
    D = ctx.D
    nA = D^2 - 1
    f_free_range = (2 + nA):n
    snap = base_active_mask(theta_base, ctx, obj)
    best_k = first(f_free_range)
    best_switches = -1
    for k in f_free_range
        v = zeros(n); v[k] = 1.0
        nsw, _, _ = count_switches(snap, theta_base .+ test_h .* v, ctx, obj)
        if nsw > best_switches
            best_switches = nsw
            best_k = k
        end
    end
    v = zeros(n); v[best_k] = 1.0
    return v, best_k, best_switches
end
