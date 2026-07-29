# Sequential reduced-q-subspace outer-search backend (2026-07-29 continuation session).
#
# EXPERIMENTAL. Does NOT change the production (A,f) backend (finite_delta_outer.jl's
# `:auto`/`:B*` gradient_backend symbols) or the full-coordinate experimental (A,q) backend
# (`aq_experimental_backend.jl`) -- both are left byte-for-byte untouched. Every function
# here is additive, new-file-only, and is referenced from `include_melitz.jl`/
# `test/melitz/runtests.jl` via new `include(...)` lines only.
#
# MOTIVATION (docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md, Phase 7/Conclusion 2):
# individual free-q coordinate secants are essentially exact and improve with W (Phase 6),
# and a DIRECT dense-block fixed-dual secant along an arbitrary q direction also tracks the
# fully reoptimized DeltaStar change essentially exactly and improves with W (Phase 7, Q2).
# But SUMMING the coordinatewise secants into a dense-direction prediction (Phase 7, Q1/Q3)
# does not improve with W and disagrees in sign 5-9% of the time. The reduced-q-subspace
# backend never sums coordinatewise secants into a KNITRO-facing gradient: it collapses the
# free-q block to ONE scalar KNITRO variable `s` per stage, moving along one fixed dense
# direction `b_q`, and supplies KNITRO with the DIRECT block secant's derivative w.r.t. `s`
# (Phase 5 below) -- never `dot(b_q, coordinatewise_secants)`.
#
# ARCHITECTURE (one stage):
#   x_reduced = (g, A_free..., s)                        length 2 + (D^2-1) = nA+2
#   q_free_free(s) = q_anchor_free + s .* q_basis_free    length D^2-2  (Phase 1)
#   theta_full(x_reduced) = (g, A_free..., q_free_free(s))  length 1+(D^2-1)+(D^2-2), fed
#       DIRECTLY into the EXISTING `expand_free_theta_logcutoff`/`melitz_expand_theta`
#       dispatcher -- no second gravity-reconstruction implementation anywhere in this file.
#
# Rules preserved (governing prompt's own non-negotiables, checked throughout):
#   - the four typed `MelitzInnerResult` classifications are used AS-IS via
#     `solve_melitz_delta!`/`MelitzInnerSession` (inner_session.jl/inner_screening.jl) --
#     never reimplemented, never anything called "infeasible" for `NumericalFailure`;
#   - `CappedEvaluation(cap)` => raw `lower_limit=-cap` is enforced by the SAME
#     `MelitzInnerSession` constructor assert every other Melitz driver already goes through;
#   - matrix-free `MelitzCCBundle`/structured Hessians/no dense G are unaffected -- this file
#     adds an OUTER-loop KNITRO problem only, never touches the inner CC dual solve's own
#     matrix-free machinery;
#   - 20-Julia-thread policy: this file's own new code (state-map arithmetic, direct-block
#     secants, crossing bisection) is O(D^2) or O(W) per call with NO `Threads.@threads`
#     kernel of its own (mirrors the 2026-07-29 daytime/afternoon sessions' own disclosed
#     convention for their diagnostic code -- `melitz_exact_a_gradient_full!`/
#     `melitz_q_coordinate_probe`, which THIS file calls, already run single-threaded
#     per-call and are the actual O(W*D)/O(W) hot loops).

using LinearAlgebra: dot, norm

# ============================================================================
# Phase 1: reduced-q state representation and state map.
# ============================================================================

"""
    MelitzReducedQStage(q_anchor_free, q_basis_free, stage_id, bandwidth_policy,
                         theta_anchor, s_lo, s_hi, fingerprint)

One sequential-search stage's frozen reduced-q geometry (governing prompt Phase 1).
`q_anchor_free`/`q_basis_free` are length `D^2-2` (the SAME domain as `theta_free`'s own
`q_free_free` block, i.e. `theta_free[2+nA:end]` under `:logcutoff` -- NOT the `D^2-1`-length
`f_free_lin` domain that additionally includes the analytically-reconstructed q-gravity pivot
cell). `theta_anchor` is the FULL verified `theta_free` this stage was built from (kept for
diagnostics/fingerprinting, not read by the state map itself beyond `s_lo`/`s_hi` construction).
`s_lo`/`s_hi` (Phase 4) is the EXACT feasible interval for `s` at this anchor's own
`(g,A_free)` values, derived in closed form from the transformed affine cutoff system
(Phase 2) -- not sampled, not a heuristic default.
"""
struct MelitzReducedQStage
    q_anchor_free::Vector{Float64}
    q_basis_free::Vector{Float64}
    stage_id::Int
    bandwidth_policy::MelitzQBandwidthPolicy
    theta_anchor::Vector{Float64}
    s_lo::Float64
    s_hi::Float64
    fingerprint::UInt64
end

"""
    melitz_reduced_q_nq(stage) -> Int

Free-q block length (`D^2-2`) this stage's `q_anchor_free`/`q_basis_free` are defined over.
"""
melitz_reduced_q_nq(stage::MelitzReducedQStage) = length(stage.q_anchor_free)

"""
    melitz_reduced_q_dim(ctx) -> (nA, nq, n_full, n_reduced)

Dimension bookkeeping shared by every function below. `ctx` must have
`outer_parameterization == :logcutoff` and `technology_coordinate == :logA` (the plain,
unscaled technology coordinate -- this experimental backend does not support the
`:theta_logA`/`:sigma_minus_one_logA` rescaled coordinates; a disclosed scope restriction,
not silently ignored -- see `melitz_reduced_q_check_ctx`).
"""
function melitz_reduced_q_dim(ctx)
    D = ctx.D
    nA = D^2 - 1
    nq = D^2 - 2
    return nA, nq, 1 + nA + nq, 2 + nA
end

function melitz_reduced_q_check_ctx(ctx)
    get(ctx, :outer_parameterization, :logf) == :logcutoff || throw(ArgumentError(
        "reduced_q_subspace: requires ctx.outer_parameterization == :logcutoff (got " *
        "$(get(ctx, :outer_parameterization, :logf))) -- this experimental backend never " *
        "silently reinterprets a :logf ctx's A-block as q."))
    get(ctx, :technology_coordinate, :logA) == :logA || throw(ArgumentError(
        "reduced_q_subspace: requires ctx.technology_coordinate == :logA (got " *
        "$(get(ctx, :technology_coordinate, :logA))) -- disclosed scope restriction, not " *
        "wired for the rescaled technology coordinates this session."))
    return nothing
end

"""
    melitz_reduced_full_theta(x_reduced, stage, ctx) -> theta_free

The Phase 1 state map. `x_reduced = (g, A_free..., s)`, length `2+nA`. Reconstructs
`theta_free = (g, A_free..., q_anchor_free + s*q_basis_free)`, length `1+nA+nq`, and hands it
straight to the EXISTING `expand_free_theta_logcutoff`/`melitz_expand_theta` dispatcher at
every call site that needs the full economic state -- this function performs NO gravity
reconstruction itself, only the affine free-vector assembly.

`s=0` reproduces `stage.theta_anchor` bit-for-bit whenever `x_reduced[1:1+nA] ==
stage.theta_anchor[1:1+nA]` (Phase 1's own acceptance criterion) -- exact by construction,
not merely approximately: `q_anchor_free + 0*q_basis_free === q_anchor_free` in floating
point (zero times anything finite is exactly `0.0`, added to `q_anchor_free` unchanged).
"""
function melitz_reduced_full_theta(x_reduced::AbstractVector{<:Real}, stage::MelitzReducedQStage, ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    length(x_reduced) == n_reduced || throw(ArgumentError(
        "melitz_reduced_full_theta: x_reduced must have length 2+nA=$n_reduced, got $(length(x_reduced))"))
    s = x_reduced[end]
    q_free_free = stage.q_anchor_free .+ s .* stage.q_basis_free
    return vcat(x_reduced[1], x_reduced[2:1+nA], q_free_free)
end

"""
    melitz_reduced_q_lift_matrix(stage, ctx) -> (M, c0)

`theta_free = M*x_reduced + c0` exactly (Phase 1/2) -- `M` is `n_full x n_reduced`, `c0` is
`n_full`. Used only to derive the reduced affine cutoff system (Phase 2) and by tests that
cross-check the state map against the production full-state reconstruction; never on any hot
path (the state map itself, `melitz_reduced_full_theta`, is the direct `vcat` above, not a
matrix multiply).
"""
function melitz_reduced_q_lift_matrix(stage::MelitzReducedQStage, ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    M = zeros(n_full, n_reduced)
    c0 = zeros(n_full)
    M[1, 1] = 1.0
    @inbounds for i in 1:nA
        M[1+i, 1+i] = 1.0
    end
    @inbounds for k in 1:nq
        M[1+nA+k, end] = stage.q_basis_free[k]
        c0[1+nA+k] = stage.q_anchor_free[k]
    end
    return M, c0
end

"""
    melitz_reduced_q_stage_fingerprint(stage_id, q_anchor_free, q_basis_free, bandwidth_policy,
                                        ctx, W, qmc_seed) -> UInt64

Phase 1's own fingerprint requirement: changes whenever the anchor, basis, basis
normalization, bandwidth policy, `W`, or QMC seed changes -- so a stale reduced-state
reconstruction or stale gradient can never be silently reused across a direction refresh.
`string(bandwidth_policy)` is used (not `hash(bandwidth_policy, ...)` directly) because the
`MelitzQBandwidthPolicy` concrete types have no custom `Base.hash` method and are NOT
`isbits`-derived-hash types in this Julia version -- `string(...)` on a plain `Float64`/`Int`-
field struct prints every field's actual value via `Base.show`'s default struct printer, so
two policies differing in ANY field produce different strings (verified directly in the
regression test, not merely assumed).
"""
function melitz_reduced_q_stage_fingerprint(stage_id::Integer, q_anchor_free::AbstractVector,
                                             q_basis_free::AbstractVector,
                                             bandwidth_policy::MelitzQBandwidthPolicy, ctx,
                                             W::Integer, qmc_seed)
    h = hash(:melitz_reduced_q_stage_v1)
    h = hash(melitz_context_fingerprint(ctx), h)
    h = hash(Int(stage_id), h)
    h = hash(Vector{Float64}(q_anchor_free), h)
    h = hash(Vector{Float64}(q_basis_free), h)
    h = hash(string(bandwidth_policy), h)
    h = hash(Int(W), h)
    h = hash(qmc_seed, h)
    return UInt(h)
end

# ============================================================================
# Phase 2: transformed affine cutoff constraints.
# ============================================================================

"""
    melitz_reduced_affine_cutoff_system(stage, ctx) -> (C_r, b_r, sys)

Phase 2: the EXACT reduced-coordinate affine cutoff system `C_r*x_reduced + b_r >= 0`,
equivalent (by construction, not approximation) to the production full system `sys.C*theta_free
+ sys.b >= 0` (`build_melitz_affine_cutoff_system`, `affine_cutoff.jl`) composed with the
Phase 1 state map: `C_r = sys.C*M`, `b_r = sys.C*c0 + sys.b`, where `(M,c0)` is
`melitz_reduced_q_lift_matrix`'s own affine map. Because `theta_free` is affine in
`x_reduced` and the production system is affine in `theta_free`, `C_r`/`b_r` are EXACT --
verified (not merely derived) in the regression test by comparing `C_r*x+b_r` against
`sys.C*theta_full(x)+sys.b` at randomized reduced points.

Uses `sys.C`/`sys.b` (row-scaled -- the SAME pair `melitz_register_finite_delta_knitro_problem!`
registers for the production `:linear` cutoff backend), so a caller registering `C_r`/`b_r`
directly with KNITRO gets a numerically-scaled system on the SAME convention production uses.
"""
function melitz_reduced_affine_cutoff_system(stage::MelitzReducedQStage, ctx)
    melitz_reduced_q_check_ctx(ctx)
    sys = build_melitz_affine_cutoff_system(ctx)
    M, c0 = melitz_reduced_q_lift_matrix(stage, ctx)
    C_r = sys.C * M
    b_r = sys.C * c0 .+ sys.b
    return C_r, b_r, sys
end

"""
    melitz_reduced_q_s_interval(stage, ctx; s_box=1.0) -> (s_lo, s_hi)

Phase 4: the EXACT feasible interval for `s` at fixed `(g,A_free) = stage.theta_anchor`'s own
values (i.e. the reduced affine cutoff system evaluated along the 1-D line
`x_reduced = (g_anchor, A_anchor_free..., s)`), intersected with the symmetric trust range
`[-s_box, s_box]`. Closed form (each cutoff row is affine in the SCALAR `s` at fixed
`(g,A_free)`, so this is an exact 1-D LP, not a sampled/bisected approximation): row `r` reads
`base_r + C_r[r,end]*s >= 0` where `base_r = dot(C_r[r,1:end-1], x_anchor_ga) + b_r[r]`; solved
per-row for the implied one-sided bound on `s`, then intersected across all `m_cut` rows.
"""
function melitz_reduced_q_s_interval(stage::MelitzReducedQStage, ctx; s_box::Real=1.0)
    C_r, b_r, _ = melitz_reduced_affine_cutoff_system(stage, ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    x_ga = stage.theta_anchor[1:1+nA]   # (g, A_free...) at the anchor
    base = C_r[:, 1:end-1] * x_ga .+ b_r
    slope = C_r[:, end]
    s_lo = -Float64(s_box)
    s_hi = Float64(s_box)
    tol = 1e-12
    @inbounds for r in eachindex(slope)
        if slope[r] > tol
            s_lo = max(s_lo, -base[r] / slope[r])
        elseif slope[r] < -tol
            s_hi = min(s_hi, base[r] / (-slope[r]))
        end
        # slope[r] ~ 0: this row does not depend on s at this anchor -- its own feasibility at
        # s=0 (base[r]>=0, true since the anchor itself is verified feasible) bounds nothing
        # further along s.
    end
    return s_lo, s_hi
end

# ============================================================================
# Phase 3/4 support: q-direction two-sided crossing counts and direct dense-block secants,
# promoted to source level from the validated Phase 7/9 diagnostic-script pattern (governing
# prompt Rule 10: no second, script-only implementation) -- generalizes
# `q_bandwidth_policy.jl`'s single-COORDINATE `melitz_q_two_sided_crossings` to an arbitrary
# DIRECTION over the free-q block, and generalizes the diagnostic scripts'
# `direction_two_sided_crossings`/the Phase 7 "direct block fixed-dual secant" pattern into
# one reusable, tested function each.
# ============================================================================

"""
    melitz_q_direction_two_sided_crossings(theta_full, d, t, ctx, sorted_ctx) -> (plus, minus)

Two-sided total crossing count (summed over every full q cell that moves, including the
q-gravity pivot cell) for moving the FREE-q block of `theta_full` by `+-t*d` (`d` length `nq`)
-- the direct generalization of `q_bandwidth_policy.jl`'s own
`melitz_q_two_sided_crossings(theta0,m,h,...)` (single coordinate `m`) to an arbitrary dense
direction `d`, matching the Phase 7 diagnostic script's own `direction_two_sided_crossings`
exactly (promoted here so both the diagnostics and this file share one implementation).
"""
function melitz_q_direction_two_sided_crossings(theta_full::AbstractVector, d::AbstractVector,
                                                 t::Real, ctx, sorted_ctx)
    D = ctx.D
    nA = D^2 - 1
    theta_p = copy(theta_full); theta_p[1+nA+1:end] .+= t .* d
    theta_m = copy(theta_full); theta_m[1+nA+1:end] .-= t .* d
    theta_plain = melitz_unpower_theta_free(theta_full, ctx)
    theta_p_plain = melitz_unpower_theta_free(theta_p, ctx)
    theta_m_plain = melitz_unpower_theta_free(theta_m, ctx)
    _, _, _, _, q0 = expand_free_theta_logcutoff(theta_plain, ctx)
    _, _, _, _, qp = expand_free_theta_logcutoff(theta_p_plain, ctx)
    _, _, _, _, qm = expand_free_theta_logcutoff(theta_m_plain, ctx)
    total_plus = 0
    total_minus = 0
    @inbounds for o in 1:D, dd in 1:D
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        kb = melitz_active_tail_start(sorted_z_o, exp(q0[o, dd]))
        kp = melitz_active_tail_start(sorted_z_o, exp(qp[o, dd]))
        km = melitz_active_tail_start(sorted_z_o, exp(qm[o, dd]))
        total_plus += abs(kp - kb)
        total_minus += abs(km - kb)
    end
    return total_plus, total_minus
end

"""
    melitz_bisect_amplitude_for_target_crossings(theta_full, d, target, ctx, sorted_ctx;
        t_lo=1e-8, t_hi=0.5, max_iter=40) -> t

Bisects (geometric midpoint, matching `q_bandwidth_policy.jl`'s own
`_melitz_bisect_h_two_sided` convention) for the smallest amplitude `t` with
`min(plus,minus) >= target` along direction `d` -- the Phase 4/Phase 7 crossing-target
amplitude selector, promoted to source.
"""
function melitz_bisect_amplitude_for_target_crossings(theta_full::AbstractVector, d::AbstractVector,
                                                        target::Integer, ctx, sorted_ctx;
                                                        t_lo::Real=1e-8, t_hi::Real=0.5, max_iter::Int=40)
    lo, hi = Float64(t_lo), Float64(t_hi)
    tp_hi, tm_hi = melitz_q_direction_two_sided_crossings(theta_full, d, hi, ctx, sorted_ctx)
    min(tp_hi, tm_hi) < target && return hi   # cannot reach target within t_hi -- best effort
    for _ in 1:max_iter
        mid = sqrt(lo * hi)
        tp, tm = melitz_q_direction_two_sided_crossings(theta_full, d, mid, ctx, sorted_ctx)
        if min(tp, tm) >= target
            hi = mid
        else
            lo = mid
        end
        hi / lo < 1.01 && break
    end
    return hi
end

"""
    melitz_q_direct_block_secant(theta_full, d, t, obj, ctx, x_dual; mode=:fixed_dual)
        -> (secant, ok_p, ok_m)

The direct dense-block secant along direction `d` (length `nq`), step `t`, promoted from the
Phase 7 diagnostic script's own validated pattern (`docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md`
Section "Phase 7 (PRIMARY TEST)", Q2: 100% sign agreement with the fully reoptimized secant at
every tested W, error shrinking to zero as W grows). `mode=:fixed_dual` (default, cheap, what
this file's own reduced-gradient callback uses): central difference of `-obj(x_dual)` at the
SAME dual, no re-solve -- `ok_p`/`ok_m` are always `true` (no solve to fail). `mode=:reoptimized`
(diagnostic/validation use only): central difference of the fully reoptimized `DeltaStar` via
`melitz_recover_lfd` -- `ok_p`/`ok_m` report whether each displaced endpoint verified
(`lfd_ok`); `secant` is `NaN` if either side failed to verify.

Restores `obj.op` to `theta_full` before returning either way (matches
`melitz_q_coordinate_probe`'s own restoration contract).
"""
function melitz_q_direct_block_secant(theta_full::AbstractVector, d::AbstractVector, t::Real,
                                       obj, ctx, x_dual::AbstractVector; mode::Symbol=:fixed_dual)
    mode in (:fixed_dual, :reoptimized) || throw(ArgumentError(
        "melitz_q_direct_block_secant: mode must be :fixed_dual or :reoptimized, got $mode"))
    D = ctx.D
    nA = D^2 - 1
    theta_p = copy(theta_full); theta_p[1+nA+1:end] .+= t .* d
    theta_m = copy(theta_full); theta_m[1+nA+1:end] .-= t .* d
    local secant, ok_p, ok_m
    if mode == :fixed_dual
        melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Bp = -obj(x_dual)
        melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Bm = -obj(x_dual)
        secant = (Bp - Bm) / (2t)
        ok_p = true; ok_m = true
    else
        obj.use_cached_x = false; obj.x .= NaN; lfd_p = melitz_recover_lfd(obj, theta_p)
        obj.use_cached_x = false; obj.x .= NaN; lfd_m = melitz_recover_lfd(obj, theta_m)
        ok_p = lfd_p.lfd_ok; ok_m = lfd_m.lfd_ok
        secant = (ok_p && ok_m) ? (lfd_p.Delta - lfd_m.Delta) / (2t) : NaN
    end
    melitz_update_operator_at_theta!(obj.op, theta_full, ctx)
    return secant, ok_p, ok_m
end

"""
    melitz_q_direct_block_secant_one_sided(theta_full, d, t, obj, ctx, x_dual; sign=+1)
        -> secant

One-sided fixed-dual secant along `d` (Phase 5's own "carefully implemented one-sided secant
if a symmetric probe is unavailable" fallback): `sign=+1` gives `(B(s+t)-B(s))/t`, `sign=-1`
gives `(B(s)-B(s-t))/t` -- both a genuine forward/backward difference at `theta_full`'s own
base point (not a silently-clipped central probe mislabeled as central).
"""
function melitz_q_direct_block_secant_one_sided(theta_full::AbstractVector, d::AbstractVector,
                                                 t::Real, obj, ctx, x_dual::AbstractVector; sign::Int=1)
    sign in (1, -1) || throw(ArgumentError("sign must be +1 or -1"))
    D = ctx.D
    nA = D^2 - 1
    melitz_update_operator_at_theta!(obj.op, theta_full, ctx); B0 = -obj(x_dual)
    theta_pert = copy(theta_full); theta_pert[1+nA+1:end] .+= sign .* t .* d
    melitz_update_operator_at_theta!(obj.op, theta_pert, ctx); Bpert = -obj(x_dual)
    melitz_update_operator_at_theta!(obj.op, theta_full, ctx)
    return sign * (Bpert - B0) / t
end

# ============================================================================
# Phase 3: candidate q-direction proposal.
# ============================================================================

"""
    melitz_reduced_q_propose_direction(theta_anchor, x0, ctx, obj; bandwidth_policy,
        var_scale_q=nothing, sign_check_target=10) -> (d, g_q)

Phase 3: proposes ONE unit-norm dense direction `d` (length `nq`) in the free-q block.
`g_q` (length `nq`) is the coordinatewise fixed-dual secant vector (via the EXISTING
`melitz_q_coordinate_probe`, `q_bandwidth_policy.jl` -- used here ONLY as a proposal signal,
never assembled into the KNITRO-facing gradient). The scaled steepest-descent proposal is
`d_tilde = -S_q^{-2} .* g_q` (`S_q=var_scale_q`, default `ones(nq)` -- no separate q-variable
KNITRO scaling is currently set on this experimental backend, so the default reduces to plain
steepest descent `-g_q`), normalized to unit norm.

The SIGN of `d` is chosen so that increasing `s` locally REDUCES the direct fixed-dual
estimate of `DeltaStar` (governing prompt Phase 3 step 7 -- literal, independent of
`direction=:upper/:lower`; KNITRO itself decides which sign of `s` to move given this fixed
convention), verified via ONE `melitz_q_direct_block_secant` call at a small
crossing-calibrated probe amplitude (`sign_check_target` two-sided crossings, default 10).

Returns `(nothing, g_q)` if either the steepest-descent proposal has non-finite/negligible
norm, or the direct-block secant at the probe amplitude is non-finite/negligible -- Phase 3's
own "do not invent a useful direction" requirement; the caller (`melitz_build_reduced_q_stage`)
falls back to a welfare-plus-A-only stage in that case.
"""
function melitz_reduced_q_propose_direction(theta_anchor::AbstractVector, x0::AbstractVector, ctx, obj;
                                             bandwidth_policy::MelitzQBandwidthPolicy=PowerScaledQBandwidth(1e-3, 80_000, 0.5),
                                             var_scale_q::Union{Nothing,AbstractVector}=nothing,
                                             sign_check_target::Integer=10)
    melitz_reduced_q_check_ctx(ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    sorted_ctx = ctx.sorted_tail_ctx
    g_q = zeros(nq)
    for m in 1:nq
        r = melitz_q_coordinate_probe(theta_anchor, m, bandwidth_policy, obj, ctx; x0=x0, mode=:fixed_dual)
        g_q[m] = r.secant
    end
    Sq2inv = var_scale_q === nothing ? ones(nq) : 1.0 ./ (Vector{Float64}(var_scale_q) .^ 2)
    length(Sq2inv) == nq || throw(ArgumentError("var_scale_q must have length nq=$nq"))
    d_tilde = -Sq2inv .* g_q
    nrm = norm(d_tilde)
    (isfinite(nrm) && nrm > 1e-300) || return nothing, g_q
    d = d_tilde ./ nrm

    t_probe = melitz_bisect_amplitude_for_target_crossings(theta_anchor, d, sign_check_target, ctx, sorted_ctx)
    secant_s, _, _ = melitz_q_direct_block_secant(theta_anchor, d, t_probe, obj, ctx, x0; mode=:fixed_dual)
    (isfinite(secant_s) && abs(secant_s) > 1e-300) || return nothing, g_q
    secant_s > 0 && (d = -d)
    return d, g_q
end

"""
    melitz_reduced_q_choose_direction(theta_anchor, x0, ctx, obj; bandwidth_policy,
        previous_direction=nothing, sign_check_target=10) -> (d, g_q, source)

Phase 3's optional "retain the previous stage's own successful direction as one alternative"
step: proposes a fresh direction (`melitz_reduced_q_propose_direction`), and if
`previous_direction !== nothing` also re-signs/re-evaluates the previous unit direction AT
THE NEW ANCHOR (never assumed still descent-oriented), then picks whichever of the (at most
two) candidates has the LARGER-magnitude direct fixed-dual secant per unit scaled norm
(both candidates are already unit-norm, so this is simply the larger `|secant|`).
`source` is `:new`, `:previous`, or `:none` (Phase 3's own "no useful direction found").
"""
function melitz_reduced_q_choose_direction(theta_anchor::AbstractVector, x0::AbstractVector, ctx, obj;
                                            bandwidth_policy::MelitzQBandwidthPolicy=PowerScaledQBandwidth(1e-3, 80_000, 0.5),
                                            previous_direction::Union{Nothing,AbstractVector}=nothing,
                                            sign_check_target::Integer=10)
    sorted_ctx = ctx.sorted_tail_ctx
    d_new, g_q = melitz_reduced_q_propose_direction(theta_anchor, x0, ctx, obj;
        bandwidth_policy=bandwidth_policy, sign_check_target=sign_check_target)

    candidates = NamedTuple[]
    if d_new !== nothing
        t_probe = melitz_bisect_amplitude_for_target_crossings(theta_anchor, d_new, sign_check_target, ctx, sorted_ctx)
        s_new, _, _ = melitz_q_direct_block_secant(theta_anchor, d_new, t_probe, obj, ctx, x0; mode=:fixed_dual)
        push!(candidates, (d=d_new, secant=s_new, source=:new))
    end
    if previous_direction !== nothing
        nq = length(previous_direction)
        d_prev = previous_direction ./ norm(previous_direction)
        t_probe = melitz_bisect_amplitude_for_target_crossings(theta_anchor, d_prev, sign_check_target, ctx, sorted_ctx)
        s_prev, _, _ = melitz_q_direct_block_secant(theta_anchor, d_prev, t_probe, obj, ctx, x0; mode=:fixed_dual)
        if isfinite(s_prev) && abs(s_prev) > 1e-300
            s_prev > 0 && (d_prev = -d_prev; s_prev = -s_prev)
            push!(candidates, (d=d_prev, secant=s_prev, source=:previous))
        end
    end
    isempty(candidates) && return nothing, g_q, :none
    best = argmax(c -> abs(c.secant), candidates)   # argmax(f, itr) returns the ELEMENT, not an index
    return best.d, g_q, best.source
end

# ============================================================================
# Phase 4: normalize and bound the scalar q coordinate; assemble a stage.
# ============================================================================

"""
    melitz_build_reduced_q_stage(theta_anchor, x0, ctx, obj, stage_id; bandwidth_policy,
        target_switches=100, s_box=1.0, previous_direction=nothing, qmc_seed=nothing,
        sign_check_target=10) -> Union{Nothing,MelitzReducedQStage}

Phases 3-4: builds one `MelitzReducedQStage` at a verified anchor `theta_anchor` (with dual
`x0`). `target_switches` (default 100, per Phase 4's own recommended default) is the total
two-sided crossing-count target used to set the raw amplitude `r_q` of the unit direction
`d` (`b_q = r_q*d`), via `melitz_bisect_amplitude_for_target_crossings`. `s_lo`/`s_hi`
(Phase 4) are the exact feasible interval for `s` at this anchor, `[-s_box,s_box]` by
default before the affine-constraint intersection.

Returns `nothing` if no useful direction was found (Phase 3's own "do not invent a useful
direction" -- the caller should run a welfare-plus-A-only stage instead in that case, per
Phase 8 Step 6).
"""
function melitz_build_reduced_q_stage(theta_anchor::AbstractVector, x0::AbstractVector, ctx, obj,
                                       stage_id::Integer;
                                       bandwidth_policy::MelitzQBandwidthPolicy=PowerScaledQBandwidth(1e-3, 80_000, 0.5),
                                       target_switches::Integer=100, s_box::Real=1.0,
                                       previous_direction::Union{Nothing,AbstractVector}=nothing,
                                       qmc_seed=nothing, sign_check_target::Integer=10)
    melitz_reduced_q_check_ctx(ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    length(theta_anchor) == n_full || throw(ArgumentError(
        "melitz_build_reduced_q_stage: theta_anchor must have length n_full=$n_full, got $(length(theta_anchor))"))
    sorted_ctx = ctx.sorted_tail_ctx

    d, g_q, source = melitz_reduced_q_choose_direction(theta_anchor, x0, ctx, obj;
        bandwidth_policy=bandwidth_policy, previous_direction=previous_direction,
        sign_check_target=sign_check_target)
    d === nothing && return nothing

    r_q = melitz_bisect_amplitude_for_target_crossings(theta_anchor, d, target_switches, ctx, sorted_ctx)
    q_basis_free = r_q .* d
    q_anchor_free = collect(Float64.(theta_anchor[1+nA+1:end]))

    fp = melitz_reduced_q_stage_fingerprint(stage_id, q_anchor_free, q_basis_free, bandwidth_policy,
                                             ctx, obj.op.W, qmc_seed)
    stage0 = MelitzReducedQStage(q_anchor_free, q_basis_free, Int(stage_id), bandwidth_policy,
                                  collect(Float64.(theta_anchor)), -Float64(s_box), Float64(s_box), fp)
    s_lo, s_hi = melitz_reduced_q_s_interval(stage0, ctx; s_box=s_box)
    return MelitzReducedQStage(q_anchor_free, q_basis_free, Int(stage_id), bandwidth_policy,
                                collect(Float64.(theta_anchor)), s_lo, s_hi, fp)
end

# ============================================================================
# Phase 5: the reduced outer gradient.
# ============================================================================

"""
    melitz_reduced_q_gradient!(g_reduced, x_reduced, stage, ctx, obj, x_dual;
        gamma_h=1e-6, s_crossing_target=50, s_h_lo=1e-8, s_h_hi=1.0) -> (g_reduced, info)

Phase 5. Fills `g_reduced` (length `2+nA`) with `d(DeltaStar)/d(x_reduced)` at the reduced
point `x_reduced=(g,A_free...,s)`:

  1. welfare block (`g_reduced[1]`): cheap fixed-dual central secant on `g` -- the SAME
     construction every existing direct FD backend uses for this coordinate
     (`aq_experimental_backend.jl`'s own Step 1, `make_melitz_gradient_delta_direct_aq_experimental`).
  2. A block (`g_reduced[2:1+nA]`): the EXACT envelope-theorem gradient
     (`exact_a_gradient.jl`), evaluated at the CURRENT reconstructed `theta_full` (which may
     have `s != 0`) -- zero finite-difference probes for this block, at any `s`.
  3. scalar-s block (`g_reduced[end]`): a DIRECT dense-block fixed-dual CENTRAL secant along
     `stage.q_basis_free`, via `melitz_q_direct_block_secant` -- literally
     `(B(s+h_s)-B(s-h_s))/(2h_s)`, NEVER `dot(stage.q_basis_free, coordinatewise_secants)`.
     `h_s` is chosen (Phase 5's own instruction) via the two-sided crossing infrastructure,
     targeting `s_crossing_target` (default 50, within the governing prompt's own 25-100
     recommended range) total two-sided switches. Near a stage/support boundary (either
     `s+h_s` or `s-h_s` would violate `stage.s_lo/s_hi`, checked via the EXACT Phase 2 reduced
     affine system, not a heuristic), falls back to a genuine one-sided secant on whichever
     side remains feasible, and reports this explicitly via `info.one_sided` -- never silently
     clips a central probe while still labeling it central.

`obj.op` must already reflect `x_reduced`'s own reconstructed `theta_full` with `x_dual` its
verified optimal dual (mirrors `melitz_exact_a_gradient_full!`'s own contract) -- this function
does not itself re-verify optimality, only reconstructs/restores `obj.op`'s state around its
own internal probes.
"""
function melitz_reduced_q_gradient!(g_reduced::AbstractVector{Float64}, x_reduced::AbstractVector{Float64},
                                     stage::MelitzReducedQStage, ctx, obj, x_dual::AbstractVector{Float64};
                                     gamma_h::Real=1e-6, s_crossing_target::Integer=50,
                                     s_h_lo::Real=1e-8, s_h_hi::Real=1.0)
    melitz_reduced_q_check_ctx(ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    length(g_reduced) == n_reduced || throw(ArgumentError(
        "melitz_reduced_q_gradient!: g_reduced must have length 2+nA=$n_reduced"))
    D = ctx.D
    theta_full = melitz_reduced_full_theta(x_reduced, stage, ctx)

    # 1. welfare block.
    theta_p = copy(theta_full); theta_p[1] += gamma_h
    theta_m = copy(theta_full); theta_m[1] -= gamma_h
    melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Dp = -obj(x_dual)
    melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Dm = -obj(x_dual)
    melitz_update_operator_at_theta!(obj.op, theta_full, ctx)
    g_reduced[1] = (Dp - Dm) / (2gamma_h)

    # 2. exact A block (fixed q, valid at any s -- exact_a_gradient.jl reads obj.op/state at
    # theta_full as-is, no assumption s==0 anywhere in that file).
    A0, f0, gpj0, fjj0 = melitz_expand_theta(theta_full, ctx)
    state0 = MelitzExpandedState(D)
    state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
    ws_a = MelitzExactAGradientWorkspace(obj.op)
    dDelta_da_full = zeros(D, D)
    melitz_exact_a_gradient_full!(dDelta_da_full, obj, x_dual, state0, ctx, ws_a)
    g_reduced[2:1+nA] .= melitz_exact_a_gradient_free(dDelta_da_full, ctx)

    # 3. scalar-s block: direct dense-block fixed-dual secant, crossing-calibrated h_s.
    sorted_ctx = ctx.sorted_tail_ctx
    h_s = melitz_bisect_amplitude_for_target_crossings(theta_full, stage.q_basis_free, s_crossing_target,
                                                         ctx, sorted_ctx; t_lo=s_h_lo, t_hi=s_h_hi)
    s_now = x_reduced[end]
    C_r, b_r, _ = melitz_reduced_affine_cutoff_system(stage, ctx)
    ga = x_reduced[1:1+nA]
    feas(s_trial) = all(>=(-1e-9), C_r[:, 1:end-1] * ga .+ C_r[:, end] .* s_trial .+ b_r)
    plus_ok = feas(s_now + h_s)
    minus_ok = feas(s_now - h_s)
    one_sided = false
    if plus_ok && minus_ok
        secant_s, _, _ = melitz_q_direct_block_secant(theta_full, stage.q_basis_free, h_s, obj, ctx, x_dual;
                                                        mode=:fixed_dual)
    elseif plus_ok
        secant_s = melitz_q_direct_block_secant_one_sided(theta_full, stage.q_basis_free, h_s, obj, ctx, x_dual; sign=1)
        one_sided = true
    elseif minus_ok
        secant_s = melitz_q_direct_block_secant_one_sided(theta_full, stage.q_basis_free, h_s, obj, ctx, x_dual; sign=-1)
        one_sided = true
    else
        # both sides violate the stage boundary at this h_s -- shrink geometrically a few
        # times before giving up (rare: only if the CURRENT s is within h_s of BOTH ends of a
        # tight stage interval simultaneously).
        h_try = h_s
        for _ in 1:6
            h_try /= 4
            if feas(s_now + h_try)
                secant_s = melitz_q_direct_block_secant_one_sided(theta_full, stage.q_basis_free, h_try, obj, ctx, x_dual; sign=1)
                one_sided = true
                h_s = h_try
                @goto s_done
            elseif feas(s_now - h_try)
                secant_s = melitz_q_direct_block_secant_one_sided(theta_full, stage.q_basis_free, h_try, obj, ctx, x_dual; sign=-1)
                one_sided = true
                h_s = h_try
                @goto s_done
            end
        end
        secant_s = 0.0   # genuinely pinned at both boundaries within tolerance -- report zero, not a fabricated slope
        @label s_done
    end
    g_reduced[end] = secant_s
    return g_reduced, (h_s=h_s, one_sided=one_sided)
end

# ============================================================================
# Phase 7: safe one-sided fixed-dual cap screen.
#
# Algebraic basis (ALREADY established and documented in this codebase,
# inner_screening.jl's own file header: "the inner CC dual problem minimizes a raw functor
# value f(zeta,lambda;G) ... over UNCONSTRAINED (zeta,lambda) -- every point in R^{1+d} is
# dual-feasible, so for ANY (zeta,lambda), weak duality gives f(zeta,lambda;G) >= f* =
# -Delta(G), i.e. -f(zeta,lambda;G) <= Delta(G) for literally EVERY (zeta,lambda), not just a
# verified optimum." `obj(x) == f(x;G)` (the MelitzCCBundle functor, cc_bundle.jl) and
# `DeltaStar(theta) = -f(x*(theta);theta) = -min_x f(x;theta)`, so `-obj(x) <= DeltaStar(theta)`
# for ANY x, at ANY theta -- this is the SAME inequality `melitz_stored_dual_lower_bound`
# already exploits for the `:stored_dual` AboveEvaluationCap screen, applied here to a
# TRIAL reduced-q point using whatever dual `x_ref` the caller supplies (the current stage's
# own verified anchor dual is the natural, always-available choice).
# ============================================================================

"""
    melitz_reduced_q_cap_screen(x_reduced, stage, ctx, obj, x_ref, cap; tol=1e-6)
        -> Union{Nothing,Float64}

One-sided-safe screen (Phase 7): returns the fixed-dual lower bound
`lb = -obj(x_ref)` evaluated at `x_reduced`'s reconstructed `theta_full`, IF `lb > cap+tol`
(certifying `AboveEvaluationCap` without a real inner solve -- weak duality, proven above, is
unconditional, so this certificate is always valid regardless of whether `x_ref` is optimal
for this trial point). Returns `nothing` otherwise (the screen CANNOT certify anything else --
callers must fall through to a real typed inner solve, per Phase 7's own explicit restriction:
"can only safely establish AboveEvaluationCap, and only after the inequality orientation is
proven"). Leaves `obj.op` pointed at `theta_full` on return either way (mirrors
`melitz_exact_a_gradient_full!`'s own "caller owns the surrounding theta state" contract, NOT
`melitz_q_coordinate_probe`'s restore-before-return convention) -- a caller chaining this
screen into a real typed inner solve (`solve_melitz_delta!`, which itself calls
`melitz_update_operator_at_theta!` at the start of every attempt) needs no extra restore step;
a caller using this screen as a pure side-query on an unrelated `theta` must restore `obj.op`
itself afterward.
"""
function melitz_reduced_q_cap_screen(x_reduced::AbstractVector, stage::MelitzReducedQStage, ctx, obj,
                                      x_ref::AbstractVector, cap::Real; tol::Real=1e-6)
    theta_full = melitz_reduced_full_theta(x_reduced, stage, ctx)
    melitz_update_operator_at_theta!(obj.op, theta_full, ctx)
    lb = -obj(x_ref)
    return isfinite(lb) && lb > Float64(cap) + Float64(tol) ? lb : nothing
end
