# q-bandwidth convergence campaign session (2026-07-29), Phase 1: one centralized, typed
# q-bandwidth-policy interface, callable identically from diagnostics and (Phase 11) the
# experimental outer-gradient backend -- per the governing prompt's Rule 10 ("do not create a
# second, script-only implementation of q bandwidth logic").
#
# Builds directly on the prior session's own `scripts/melitz_aq_phase6_9_q_bandwidth_2026-07-29.jl`
# (`q_crossing_report`/`find_h_for_target_crossings`, attributed not re-derived), fixing the
# ONE flaw the prior session's own doc flagged but did not fix: crossing counts were computed
# ONLY for the `+h` perturbation even though the derivative itself is a central (two-sided)
# difference. Every function below computes `+h` and `-h` crossing counts SEPARATELY and
# reports both, plus their minimum (the two-sided target criterion this session's governing
# prompt specifies: "the smallest h for which min(crossings_plus, crossings_minus) >= target").

"Abstract supertype for a q-coordinate bandwidth-selection rule."
abstract type MelitzQBandwidthPolicy end

"""
    FixedRawQBandwidth(h)

Family A (governing prompt Phase 5): a fixed raw step size, independent of `W` or achieved
crossing count. A baseline, not a presumed-optimal choice.
"""
struct FixedRawQBandwidth <: MelitzQBandwidthPolicy
    h::Float64
end

"""
    PowerScaledQBandwidth(h_ref, W_ref, alpha)

Family B: `h_W = h_ref * (W_ref/W)^alpha`. `alpha=1/2` is the governing prompt's own central
theoretical candidate (mandatory); `alpha in {1/3, 2/3}` are the other required grid points.
"""
struct PowerScaledQBandwidth <: MelitzQBandwidthPolicy
    h_ref::Float64
    W_ref::Int
    alpha::Float64
end

"""
    FixedCrossingQBandwidth(target; h_lo=1e-8, h_hi=3e-2, max_iter=40)

Family C: choose `h` (by bisection) so that `min(crossings_plus, crossings_minus) >= target`,
independent of `W` -- reproduces (and corrects, via the two-sided fix above) the prior
session's own target-crossing scheme.
"""
struct FixedCrossingQBandwidth <: MelitzQBandwidthPolicy
    target::Int
    h_lo::Float64
    h_hi::Float64
    max_iter::Int
end
FixedCrossingQBandwidth(target::Int; h_lo::Real=1e-8, h_hi::Real=3e-2, max_iter::Int=40) =
    FixedCrossingQBandwidth(target, Float64(h_lo), Float64(h_hi), max_iter)

"""
    GrowingCrossingQBandwidth(target_ref, W_ref; h_lo=1e-8, h_hi=3e-2, max_iter=40)

Family D: `T_W = ceil(T_ref * sqrt(W/W_ref))` -- the crossing-density-adaptive analogue of
`h_W ∝ W^(-1/2)`: the neighborhood shrinks (via the same bisection as `FixedCrossingQBandwidth`)
while the number of crossed draws GROWS with `W`, rather than staying pinned at `target_ref`.
"""
struct GrowingCrossingQBandwidth <: MelitzQBandwidthPolicy
    target_ref::Int
    W_ref::Int
    h_lo::Float64
    h_hi::Float64
    max_iter::Int
end
GrowingCrossingQBandwidth(target_ref::Int, W_ref::Int; h_lo::Real=1e-8, h_hi::Real=3e-2, max_iter::Int=40) =
    GrowingCrossingQBandwidth(target_ref, W_ref, Float64(h_lo), Float64(h_hi), max_iter)

melitz_q_target_crossings(pol::GrowingCrossingQBandwidth, W::Integer) =
    ceil(Int, pol.target_ref * sqrt(W / pol.W_ref))

"""
    melitz_q_two_sided_crossings(theta0, m, h, ctx, sorted_ctx) -> (total_plus, total_minus, cells)

The corrected, two-sided crossing evaluator (see file header). `cells` is a
`Vector{MelitzQCellCrossing}`, one entry per FULL q cell (direct or q-pivot-adjoint) that
moves under either `+h` or `-h` -- for a free q coordinate that also moves the q-gravity
pivot cell, BOTH cells are counted, never only the nominal coordinate (governing prompt
Phase 1's explicit instruction).
"""
struct MelitzQCellCrossing
    o::Int
    d::Int
    crossings_plus::Int
    crossings_minus::Int
end

function melitz_q_two_sided_crossings(theta0::AbstractVector, m::Int, h::Real, ctx, sorted_ctx)
    D = ctx.D
    nA = D^2 - 1
    theta_p = copy(theta0); theta_p[1+nA+m] += h
    theta_m = copy(theta0); theta_m[1+nA+m] -= h
    _, _, _, _, q0 = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta0, ctx), ctx)
    _, _, _, _, qp = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta_p, ctx), ctx)
    _, _, _, _, qm = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta_m, ctx), ctx)
    cells = MelitzQCellCrossing[]
    total_plus = 0
    total_minus = 0
    @inbounds for o in 1:D, d in 1:D
        moved_p = abs(qp[o, d] - q0[o, d]) > 1e-14
        moved_m = abs(qm[o, d] - q0[o, d]) > 1e-14
        (moved_p || moved_m) || continue
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        kb = melitz_active_tail_start(sorted_z_o, exp(q0[o, d]))
        kp = melitz_active_tail_start(sorted_z_o, exp(qp[o, d]))
        km = melitz_active_tail_start(sorted_z_o, exp(qm[o, d]))
        cp = abs(kp - kb)
        cm = abs(km - kb)
        total_plus += cp
        total_minus += cm
        push!(cells, MelitzQCellCrossing(o, d, cp, cm))
    end
    return total_plus, total_minus, cells
end

"""
    _melitz_bisect_h_two_sided(target, theta0, m, ctx, sorted_ctx; h_lo, h_hi, max_iter)

Bisects on raw step `h` for the smallest value with `min(crossings_plus, crossings_minus) >=
target` (the governing prompt's own two-sided target criterion), geometric bisection exactly
as the prior session's `find_h_for_target_crossings`, but on the MIN of the two sides rather
than a single one-sided total.
"""
function _melitz_bisect_h_two_sided(target::Int, theta0::AbstractVector, m::Int, ctx, sorted_ctx;
                                     h_lo::Real=1e-8, h_hi::Real=3e-2, max_iter::Int=40)
    lo, hi = Float64(h_lo), Float64(h_hi)
    total_plus_hi, total_minus_hi, _ = melitz_q_two_sided_crossings(theta0, m, hi, ctx, sorted_ctx)
    min_hi = min(total_plus_hi, total_minus_hi)
    if min_hi < target
        return hi, total_plus_hi, total_minus_hi   # cannot reach target within h_hi -- best effort
    end
    for _ in 1:max_iter
        mid = sqrt(lo * hi)
        total_plus_mid, total_minus_mid, _ = melitz_q_two_sided_crossings(theta0, m, mid, ctx, sorted_ctx)
        if min(total_plus_mid, total_minus_mid) >= target
            hi = mid
        else
            lo = mid
        end
        hi / lo < 1.01 && break
    end
    total_plus_final, total_minus_final, _ = melitz_q_two_sided_crossings(theta0, m, hi, ctx, sorted_ctx)
    return hi, total_plus_final, total_minus_final
end

"Resolve the raw step `h` a policy selects for free q-coordinate `m` at sample size `W`."
melitz_q_select_h(pol::FixedRawQBandwidth, W::Integer, theta0, m, ctx, sorted_ctx) = pol.h
melitz_q_select_h(pol::PowerScaledQBandwidth, W::Integer, theta0, m, ctx, sorted_ctx) =
    pol.h_ref * (pol.W_ref / W)^pol.alpha
function melitz_q_select_h(pol::FixedCrossingQBandwidth, W::Integer, theta0, m, ctx, sorted_ctx)
    h, _, _ = _melitz_bisect_h_two_sided(pol.target, theta0, m, ctx, sorted_ctx;
                                          h_lo=pol.h_lo, h_hi=pol.h_hi, max_iter=pol.max_iter)
    return h
end
function melitz_q_select_h(pol::GrowingCrossingQBandwidth, W::Integer, theta0, m, ctx, sorted_ctx)
    target = melitz_q_target_crossings(pol, W)
    h, _, _ = _melitz_bisect_h_two_sided(target, theta0, m, ctx, sorted_ctx;
                                          h_lo=pol.h_lo, h_hi=pol.h_hi, max_iter=pol.max_iter)
    return h
end

"""
    MelitzQCoordinateProbeResult

The central evaluator's per-coordinate return type (governing prompt Phase 1's required
field list): actual `h` used, per-cell two-sided crossing counts (`cells`, including the
q-gravity pivot cell whenever it moves), aggregate `+`/`-`/min-side crossing totals, whether
either reoptimized endpoint failed to certify (`boundary_hit_plus`/`boundary_hit_minus` --
an LFD-classification-based PROXY for "hit a support/affine-constraint boundary", not a
literal geometric constraint check against the outer KNITRO affine cutoff system; disclosed,
not silently assumed exact), the requested secant estimate, and timing/allocation.
"""
struct MelitzQCoordinateProbeResult
    m::Int
    h::Float64
    cells::Vector{MelitzQCellCrossing}
    crossings_plus_total::Int
    crossings_minus_total::Int
    crossings_min_side::Int
    boundary_hit_plus::Bool
    boundary_hit_minus::Bool
    secant::Float64
    mode::Symbol
    elapsed_s::Float64
    bytes_allocated::Int64
end

"""
    melitz_q_coordinate_probe(theta0, m, policy, obj, ctx; x0=nothing, mode=:fixed_dual) -> MelitzQCoordinateProbeResult

The one source-level q-bandwidth evaluator -- called identically by diagnostic scripts and
(Phase 11) the experimental outer-gradient backend (governing prompt Rule 10).

`mode=:fixed_dual` (default): central difference of `-obj(x0)` at the SAME dual `x0`
(matches what a production gradient-backend callback would actually use inside one outer
KNITRO iteration -- cheap, `O(W)` per side, no re-solve). `mode=:reoptimized`: central
difference of the FULLY reoptimized `DeltaStar` (via `melitz_recover_lfd`, `CappedEvaluation`
governed) -- expensive, diagnostic-only, used to assess ground truth.

Mutates `obj.op` (via `melitz_update_operator_at_theta!`/`melitz_recover_lfd`) and restores it
to `theta0` before returning either way.
"""
function melitz_q_coordinate_probe(theta0::AbstractVector, m::Int, policy::MelitzQBandwidthPolicy,
                                    obj::MelitzCCBundle, ctx; x0::Union{Nothing,AbstractVector}=nothing,
                                    mode::Symbol=:fixed_dual)
    mode in (:fixed_dual, :reoptimized) || throw(ArgumentError("mode must be :fixed_dual or :reoptimized, got $mode"))
    D = ctx.D
    nA = D^2 - 1
    sorted_ctx = ctx.sorted_tail_ctx
    W = obj.op.W

    t0 = time()
    h = melitz_q_select_h(policy, W, theta0, m, ctx, sorted_ctx)
    total_plus, total_minus, cells = melitz_q_two_sided_crossings(theta0, m, h, ctx, sorted_ctx)

    theta_p = copy(theta0); theta_p[1+nA+m] += h
    theta_m = copy(theta0); theta_m[1+nA+m] -= h

    boundary_p = false
    boundary_m = false
    secant = NaN
    bytes = @allocated begin
        if mode == :fixed_dual
            x0 === nothing && throw(ArgumentError("melitz_q_coordinate_probe: mode=:fixed_dual requires x0"))
            melitz_update_operator_at_theta!(obj.op, theta_p, ctx)
            Dp = -obj(x0)
            melitz_update_operator_at_theta!(obj.op, theta_m, ctx)
            Dm = -obj(x0)
            secant = (Dp - Dm) / (2h)
        else
            obj.use_cached_x = false; obj.x .= NaN
            lfdp = melitz_recover_lfd(obj, theta_p)
            obj.use_cached_x = false; obj.x .= NaN
            lfdm = melitz_recover_lfd(obj, theta_m)
            boundary_p = !lfdp.lfd_ok
            boundary_m = !lfdm.lfd_ok
            secant = (lfdp.lfd_ok && lfdm.lfd_ok) ? (lfdp.Delta - lfdm.Delta) / (2h) : NaN
        end
    end
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    elapsed = time() - t0

    return MelitzQCoordinateProbeResult(m, h, cells, total_plus, total_minus, min(total_plus, total_minus),
                                         boundary_p, boundary_m, secant, mode, elapsed, bytes)
end
