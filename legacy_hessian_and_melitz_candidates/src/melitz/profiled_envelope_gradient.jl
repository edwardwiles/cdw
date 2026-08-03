# 2026-07-31 profiled-cutoff-gradient go/no-go session. Governing prompt: determine whether
# the profiled value function Phi(g,q) = min_A DeltaStar(g,q,A) (the fixed-q A-middle-loop
# architecture, `fixed_q_a_middle_loop.jl`) has a coherent SMOOTH within-chamber envelope
# derivative that can support a gradient-informed outer cutoff search. This file implements
# ONLY the exact smooth envelope derivative (Phase 1) -- it does not touch the middle solver,
# the Ricardian implementation, or any other Melitz source file.
#
# ============================================================================
# PHASE 0 DERIVATION (full writeup: docs/melitz_profiled_q_envelope_gradient_2026-07-31.md).
# Summary reproduced here as the load-bearing derivation this file's code implements.
#
# The exact implemented middle problem (fixed_q_a_middle_loop.jl, verified against source,
# not the schematic governing-prompt formula):
#
#   Phi(g,q) = min_{A_free} DeltaStar(theta(A_free,q,g))
#              s.t. c(A_free) := 0                          [A-gravity, algebraic identity]
#                   r(A_free) := rows_A*A_free - rhs_A {>=,==} 0   [ordering/same-bin]
#
# where `theta(A_free,q,g)` packs (a) the full A matrix via the SAME A-gravity pivot used
# throughout production (`pivot_expand`, g0=0 always -- an ALGEBRAIC IDENTITY of the pivot for
# ANY A_free, never a constraint that needs a KKT multiplier: c(A_free)==0 identically, so
# nabla_{g,q} c == 0 trivially and this term drops out of every envelope formula below,
# regardless of any multiplier value); (b) f, reconstructed from the FIXED q and the CURRENT
# A via `melitz_f_from_Aq` (lfd_preserving_state.jl); (c) the fixed welfare/autarky scalar
# `gamma_prime_j = exp(g)`.
#
# KEY STRUCTURAL FACT, proved directly from `melitz_fixed_q_middle_constraint_system`
# (fixed_q_a_middle_loop.jl:265-325), not assumed: `rows_A`/`rhs_A` are built ENTIRELY from
# `ctx` (data-only: sigma, w, tau, X_data, expenditure) and the destination RANK ORDER within
# each origin (`melitz_origin_intervals(o,theta_fixed_q,ctx,obj).rank`) -- the numeric VALUE
# of q never appears in a row or rhs, only the ORDER/BINNING it induces. Within a fixed
# finite-QMC chamber (rank/bin structure constant), `rows_A` and `rhs_A` are therefore
# CONSTANT in (g,q) -- i.e. `nabla_{(g,q)} r == 0` identically inside a chamber, confirming
# (not merely assuming, per the governing prompt's own instruction) that "fixed-order and
# same-bin constraints have no continuous q derivative within a chamber." `r` has no `g`
# dependence at all, at any q (the rows never involve gamma_prime_j).
#
# ENVELOPE THEOREM. Writing the middle Lagrangian L(A_free,g,q,mu) = DeltaStar(theta) +
# mu^T*r(A_free) (c drops out, per above), the standard envelope theorem gives
#
#   nabla_{(g,q)} Phi = nabla_{(g,q)} DeltaStar(A*,g,q) + mu^T * nabla_{(g,q)} r(A*)
#                     = nabla_{(g,q)} DeltaStar(A*,g,q) + mu^T * 0
#                     = nabla_{(g,q)} DeltaStar(A*,g,q)                              (EXACT)
#
# i.e. the middle multipliers contribute EXACTLY ZERO to the profiled gradient, independent of
# their values -- the only object needed is the ORDINARY partial derivative of DeltaStar at
# the middle optimum A*(g,q), holding A fixed. This is itself an envelope-theorem object (the
# INNER envelope theorem `exact_a_gradient.jl`'s own header already establishes for A/q at a
# fixed optimal dual x*), so no new finite-difference machinery is needed for either
# component:
#
#   - d(DeltaStar)/d(q_free) [fixed A, fixed g] = `melitz_exact_q_smooth_gradient_full!`/
#     `_free` (exact_q_smooth_gradient.jl, REUSED VERBATIM, unmodified) -- nonzero only on the
#     focal-origin (j,:) row, through the f_od=f_od(a_od,q_od) fixed-cost channel alone (trade
#     -share block has IDENTICALLY zero smooth q-sensitivity, per that file's own header).
#
#   - d(DeltaStar)/dg [fixed A, fixed free-q] -- NOT previously implemented anywhere in this
#     codebase as a closed form (the production outer gradient differentiates gamma_prime_j
#     via a small-h fixed-dual finite difference, `direct_gradient.jl`'s
#     `make_melitz_gradient_delta_direct_*`; the governing prompt's Phase 1 explicitly
#     prohibits finite differences for this required within-chamber derivative). DERIVED HERE
#     from source (`firm_quantities.jl`'s `melitz_firm`/`equilibrium.jl`'s
#     `derive_fjj_from_autarky_cutoff`), mirroring `exact_a_gradient.jl`'s own Step-3 autarky
#     construction exactly:
#
#     The autarky call is `melitz_firm(w_prime, 1.0, A[j,j], f_jj, sigma, expenditure_prime,
#     gamma_prime_j, z)`, i.e. `price_power_autarky = gamma_prime_j` DIRECTLY (not via f_jj
#     alone). `price(z) = markup*w_prime/(A[j,j]*z)` has NO gamma_prime_j dependence.
#     `f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_j, A[j,j], ...)` solves the
#     zero-profit-at-cutoff-1 identity `expenditure_prime*price(1)^(1-sigma)/(sigma*gamma_prime_j)
#     = w_prime*f_jj`, i.e. `f_jj = K(A[j,j]) / gamma_prime_j` for a gamma_prime_j-INDEPENDENT
#     constant `K`. Substituting into the raw profit formula and using this SAME identity to
#     eliminate `K` (exactly `exact_a_gradient.jl`'s own "zero-profit-cutoff-identity
#     substitution", re-derived here for the gamma_prime_j direction instead of the A[j,j]
#     direction) gives, for EVERY active autarky draw z>=1:
#
#         profit_autarky(z) = w_prime*f_jj*(z^(sigma-1) - 1)                (same as Step 3)
#         d(profit_autarky(z)) / d(gamma_prime_j) [fixed A[j,j]] = -profit_autarky(z)/gamma_prime_j
#         d(profit_autarky(z)) / dg  [g := log(gamma_prime_j), fixed A[j,j]] = -profit_autarky(z)
#
#     (the log-chain-rule factor is EXACTLY -1, versus the A-cell's own EXACTLY (sigma-1) --
#     both are clean, closed-form, data-independent multiplicative constants; no separate
#     formula is needed for the trade-share block, which per `exact_q_smooth_gradient.jl`'s
#     header point 4 has zero gamma_prime_j dependence at all).
#
#     CORRECTION (found live: an initial "autarky-only" version of this formula, `(mu_link/M)*
#     f_jj*(Sya-S1a)`, disagreed with reprofiled central differences at FIXED (A,q) by a
#     consistent ~5x factor, at both D4 and D20 -- pinned down by a harness sanity check
#     against the ALREADY-validated `A[j,j]` gradient, which showed the naive comparison
#     "Step-3-term-only" is not what `exact_a_gradient.jl`'s own `dDelta_da_full[j,j]` computes
#     either: `f[j,j] = derive_fjj_from_autarky_cutoff(gamma_prime_j, A[j,j], ...)` is a SINGLE
#     physical primitive used in TWO places in that function's Step 2/3 decomposition of cell
#     `(j,j)`, not one. Step 3 (autarky, masked by the HARDWIRED autarky-cutoff-1 `z_orig_j>=1`
#     condition) is only HALF of `gamma_prime_j`'s effect. `f[j,j]` is ALSO used, unchanged, in
#     Step 2's ordinary "current/baseline destination" loop AT `d=j` (the focal country's own
#     domestic cell IS one of its `D` destinations there): `link_term[d=j] = (expenditure[j]/
#     sigma)*Rt_jj - w_j*f[j,j]*tail1_j`, `tail1_j` the REGULAR participation-weighted
#     active-tail sum at cell `(j,j)`'s own rank (`op.bin`/`op.rank`), NOT the raw autarky mask.
#     `Rt_jj` (trade-share) has zero `gamma_prime_j` sensitivity (`coef_jj` depends only on
#     `A[j,j]`; `active_jj` depends only on the FIXED `q_jj`, held fixed throughout this
#     derivative) -- only the `-w_j*f[j,j]*tail1_j` piece carries `gamma_prime_j` dependence,
#     contributing an ADDITIONAL `(mu_link/M)*f_jj*tail1_j` term (same `+` Step-2 sign
#     convention `exact_a_gradient.jl` already uses for ordinary destinations, combined with
#     the same `d(f_jj)/dg=-f_jj` factor). The corrected exact formula:
#
#         d(DeltaStar)/dg [fixed A,q] = (mu_link/M) * f_jj * (tail1_j + Sya - S1a)
#
#     where `mu_link`, `M`, `f_jj`, `tail1` (hence `tail1_j`), `Sya`, `S1a` are EXACTLY the
#     same quantities `melitz_exact_a_gradient_full!`'s own Step 2/3 already compute (`tail1`
#     from `op.bin[:,j]`/`op.rank[:,j]`; `Sya = sum_{s: z_orig_j[s]>=1} dPsi(u_s)*z_power_j[s]`,
#     `S1a = sum_{s: z_orig_j[s]>=1} dPsi(u_s)`) -- this file's own `melitz_exact_g_gradient`
#     reproduces these small sub-computations standalone (not a full A-gradient call) rather
#     than duplicating the whole function. Verified (Phase 1, isolated fixed-(A,q) test,
#     bypassing any profiling/q-gravity-pivot confound) to `relerr` consistent with this
#     codebase's own established exact-gradient validation standard.
#
# REGULARITY CONDITIONS required for the above (governing prompt's own list, checked/reported
# by `profiled_envelope_derivative` below, not merely assumed):
#   1. A locally unique/stable middle solution (`r.nStatus` a locally-optimal KNITRO code).
#   2. Middle KKT stationarity: since `mu^T*nabla_{(g,q)}r == 0` REGARDLESS of `mu`'s value
#      (the structural fact above), this formula does NOT require extracting KNITRO's own
#      constraint multipliers at all -- a genuine simplification, not a gap: their value is
#      irrelevant to this specific gradient, though the ACTIVE SET (which rows bind, i.e.
#      the chamber's own same-bin/ordering structure) still matters for defining "which
#      chamber," and is fingerprinted below.
#   3. Fixed middle active set (chamber): fingerprinted via bit-identical
#      `melitz_origin_intervals(o,.).rank` for every origin, and the constraint system's own
#      `sense` vector, at both the anchor and any perturbed evaluation compared against it.
#   4. `partial_g Phi != 0` (only needed for Phase 5's boundary-gradient formula, not Phase 1).
# ============================================================================

using LinearAlgebra: dot

"""
    melitz_exact_g_gradient(obj::MelitzCCBundle, x::AbstractVector{Float64},
        state::MelitzExpandedState, ctx) -> Float64

`d(DeltaStar)/dg` at FIXED `A` (hence fixed `q`, `f`), `g := log(gamma_prime_j)` -- see this
file's header derivation. `obj.op`/`state` must reflect the SAME already-converged theta as
`x` (identical contract to `melitz_exact_a_gradient_full!`/`melitz_exact_q_smooth_gradient_full!`).
Zero finite-difference probes, zero re-solves -- reproduces exactly the two sub-computations
`melitz_exact_a_gradient_full!`'s own Step 2 (ordinary destination `d=j`) and Step 3 (autarky)
perform for cell `(j,j)`, standalone (both carry `gamma_prime_j` dependence through `f[j,j]`,
per this file's header CORRECTION note).
"""
function melitz_exact_g_gradient(obj, x::AbstractVector{Float64}, state::MelitzExpandedState, ctx)
    op = obj.op
    D = op.D
    W = op.W
    M = obj.M
    j = ctx.target_country

    zeta = x[1]
    mu = @view x[2:end]
    mu_link = mu[op.layout.focal_link_index]
    mu_link == 0.0 && return 0.0   # no focal link active at this dual -> g-derivative is exactly zero

    u = Vector{Float64}(undef, W)
    dpsi = Vector{Float64}(undef, W)
    mul_G!(u, op, zeta, mu)
    melitz_cc_dPsi!(dpsi, u)

    # Step-2-style regular participation-weighted active tail at cell (j,j)'s own rank
    # (op.bin[:,j]/op.rank[:,j] -- identical construction to exact_a_gradient.jl's Step 2).
    binsum1 = zeros(D + 1)
    tail1 = zeros(D + 1)
    bin_j = @view op.bin[:, j]
    @inbounds for s in 1:W
        binsum1[bin_j[s]+1] += dpsi[s]
    end
    tail1[D+1] = binsum1[D+1]
    @inbounds for k in D:-1:1
        tail1[k] = tail1[k+1] + binsum1[k]
    end
    rank_j = @view op.rank[:, j]
    tail1_j = tail1[rank_j[j]+1]

    # Step-3-style raw autarky-mask (z_orig_j>=1, the HARDWIRED autarky cutoff) active sum.
    z_orig_j = @view op.sorted_ctx.z_original[:, j]
    z_power_j = @view op.sorted_ctx.z_power_original[:, j]
    Sya = 0.0
    S1a = 0.0
    @inbounds for s in 1:W
        if z_orig_j[s] >= 1.0
            Sya += dpsi[s] * z_power_j[s]
            S1a += dpsi[s]
        end
    end
    f_jj = state.f_jj
    return (mu_link / M) * f_jj * (tail1_j + Sya - S1a)
end

"""
    MelitzProfiledEnvelopeDerivative

Output of `profiled_envelope_derivative`. `dPhi_dg`/`dPhi_dq_free` are the exact profiled
envelope derivative components (this file's header derivation -- both equal the ORDINARY
`DeltaStar` partial derivative at the middle optimum `A*`, since the middle multiplier
contribution is exactly zero by construction). `dPhi_dq_full` is the full `D x D` matrix
before the free-coordinate pivot reduction (diagnostic). `chamber_rank`/`chamber_sense` are
the chamber fingerprint (bit-identical comparison certifies "same chamber" between two
evaluations). `active_rows`/`n_active` report which middle ordering/same-bin rows are (near-)
binding at `A_free` (residual `< tol`) -- a feasibility diagnostic; per the header derivation
their multiplier VALUES are irrelevant to `dPhi_dg`/`dPhi_dq_free` regardless, so they are not
extracted from KNITRO. `nStatus`/`Delta` are the middle-loop verification this derivative is
valid AT (must be a `FiniteSolved`, locally-optimal-equivalent status).
"""
struct MelitzProfiledEnvelopeDerivative
    dPhi_dg::Float64
    dPhi_dq_free::Vector{Float64}
    dPhi_dq_full::Matrix{Float64}
    Delta::Float64
    nStatus::Int
    chamber_rank::Matrix{Int}
    chamber_sense::Vector{Symbol}
    constraint_residuals::Vector{Float64}
    active_rows::Vector{Bool}
    n_active::Int
    theta_free_at::Vector{Float64}
end

"""
    melitz_chamber_fingerprint(theta_free, ctx, obj) -> Matrix{Int}   (D x D, rank[o,d])

`melitz_origin_intervals(o,theta_free,ctx,obj).rank` for every origin `o`, stacked as a
`D x D` matrix (`fp[o,d]`) -- the active-set/chamber fingerprint two states must match
EXACTLY (bit-identical, integer-valued) to certify "same chamber."
"""
function melitz_chamber_fingerprint(theta_free::AbstractVector, ctx, obj)
    D = ctx.D
    fp = Matrix{Int}(undef, D, D)
    for o in 1:D
        iv = melitz_origin_intervals(o, theta_free, ctx, obj)
        fp[o, :] = iv.rank
    end
    return fp
end

"""
    profiled_envelope_derivative(session, r_incumbent::FiniteSolved, theta_free_incumbent,
        q_fixed, ctx; sys=nothing, active_tol=1e-8) -> MelitzProfiledEnvelopeDerivative

Phase 1 core. `r_incumbent`/`theta_free_incumbent` should be a middle-loop result already
COLD-RE-VERIFIED at the reported incumbent (e.g. `sol.r_incumbent`/`sol.theta_free_incumbent`
from `solve_melitz_fixed_q_A_profile_v2`) -- this function re-solves ONCE MORE at
`theta_free_incumbent` (cheap: a single inner solve, matching this codebase's own established
cold-reverification convention) to guarantee `session.obj.op`/the returned dual `x` genuinely
reflect that exact theta (the middle driver's own cache can return the incumbent's VALUE
without its `obj.op` being in that exact post-solve state at return time).

Reuses `melitz_exact_q_smooth_gradient_full!`/`_free` (exact_q_smooth_gradient.jl, UNMODIFIED)
for the q-component and `melitz_exact_g_gradient` (this file) for the g-component -- both
exact, zero finite differences, per this file's header derivation.
"""
function profiled_envelope_derivative(session::MelitzInnerSession, theta_free_incumbent::AbstractVector{Float64},
                                       q_fixed::AbstractMatrix{Float64}, ctx;
                                       sys::Union{Nothing,MelitzFixedQMiddleConstraintSystem}=nothing,
                                       A_free_incumbent::Union{Nothing,AbstractVector{Float64}}=nothing,
                                       active_tol::Float64=1e-8)
    theta = Vector{Float64}(theta_free_incumbent)
    r = solve_melitz_delta!(session, theta, session.policy; warm_start_source=:neutral)
    r isa FiniteSolved || throw(DomainError(r, "profiled_envelope_derivative: re-verification at " *
        "theta_free_incumbent did not return FiniteSolved (got $(typeof(r))) -- the caller's " *
        "incumbent is not a valid point to differentiate a smooth envelope derivative at."))

    obj = session.obj
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    state = MelitzExpandedState(A, f, gamma_prime_j, f_jj)

    dPhi_dg = melitz_exact_g_gradient(obj, r.x, state, ctx)

    ws_q = MelitzExactQSmoothGradientWorkspace(obj.op)
    dPhi_dq_full = zeros(ctx.D, ctx.D)
    melitz_exact_q_smooth_gradient_full!(dPhi_dq_full, obj, r.x, state, ctx, ws_q)
    dPhi_dq_free = melitz_exact_q_smooth_gradient_free(dPhi_dq_full, ctx)

    fp = melitz_chamber_fingerprint(theta, ctx, obj)

    resid = Float64[]
    sense = Symbol[]
    if sys !== nothing && A_free_incumbent !== nothing
        resid = melitz_middle_constraint_residuals(sys, Vector{Float64}(A_free_incumbent); coordinate=:logA)
        sense = sys.sense
    end
    active = [abs(resid[i]) < active_tol for i in eachindex(resid)]

    return MelitzProfiledEnvelopeDerivative(dPhi_dg, dPhi_dq_free, dPhi_dq_full, r.Delta, r.nStatus,
                                             fp, sense, resid, active, count(active), theta)
end
