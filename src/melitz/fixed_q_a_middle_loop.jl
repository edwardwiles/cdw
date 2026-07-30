# 2026-07-30 three-level outer-search architecture session. Governing prompt: bounded
# experiment for a NEW middle loop -- for FIXED welfare coordinate `g` and FIXED full cutoff
# vector `q`, genuinely optimize DeltaStar over `A` (not select F first and infer A). This is
# a separate, additive module: does NOT touch production `(A,f)` search
# (`finite_delta_outer.jl`), the default `:logf`/`:logcutoff` outer-search backends, the
# reduced-q backend (`reduced_q_*.jl`), or the joint-`(A,q)` LFD-preserving corrector
# (`lfd_preserving_state.jl`, `hybrid_chamber_corrector.jl`) -- it REUSES pieces of the last
# two (`melitz_f_from_Aq`, `reduce_to_free_theta_logcutoff`) rather than duplicating them.
#
# ============================================================================
# PHASE 0 (derivation, verified against `firm_quantities.jl`/`origin_block_screen.jl`/
# `log_cutoff_param.jl`, not assumed from the schematic governing-prompt formula):
#
# `melitz_C(w_o,tau_od,A_od,sigma,expenditure_d) = expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma)`
# (`firm_quantities.jl`). Writing `coef_od = C_od/expenditure_d` and `H_od = lambda_od/coef_od`
# (`lambda_od = X_data[o,d]/expenditure_d`, EXACTLY `origin_block_screen.jl`'s own `H[d]`
# target, line ~127):
#
#     coef_od = (markup*w_o*tau_od)^(1-sigma) * A_od^(sigma-1)
#     H_od    = lambda_od * (markup*w_o*tau_od)^(sigma-1) * A_od^(-(sigma-1))
#
#     h_od := log(H_od) = const_od - (sigma-1)*log(A_od) = const_od - (sigma-1)*a_od     (EXACT)
#
# where `const_od = log(lambda_od) + (sigma-1)*log(markup*w_o*tau_od)` is a FIXED (data-only)
# constant (`melitz_h_constant_od` below). Inverting: `a_od = [const_od - h_od]/(sigma-1)`,
# matching the governing prompt's own Phase 0 formula exactly (confirmed against the actual
# `melitz_C` code, not merely assumed).
#
# ORDERING/EQUALITY (per origin `o`, over ITS OWN `D` bilateral destinations -- different
# origins live on unrelated draw axes `z[:,o]`/`z[:,o']` and impose NO cross-origin
# restriction; the FOCAL origin's extra autarky "virtual (D+1)-th destination"
# (`melitz_origin_intervals`) does not correspond to a genuine `(o,d)` A-cell and is
# deliberately OUT OF SCOPE here -- see "SCOPE LIMITATIONS" below):
#
#   `melitz_origin_block_lp`'s own moment condition (re-derived independently in
#   `lfd_preserving_state.jl`'s header, and again here): `E_p[trade moment od]=0` iff
#   `T_od(p,q_od) = H_od`, `T_od` a NON-INCREASING step function of `q_od` (bigger cutoff ->
#   smaller active set -> smaller or equal tail sum, for ANY p). So:
#
#     q_od1 <= q_od2  (destination 1's active set is a superset of destination 2's, for ANY p)
#         => T_od1(p,.) >= T_od2(p,.) for ANY p
#         => for the SPECIFIC p that would need to satisfy BOTH moments exactly,
#            H_od1 >= H_od2, i.e. h_od1 >= h_od2                                    (ordering)
#
#     q_od1 == q_od2 in the SAME finite-QMC active-set bin (no draw with any p-mass strictly
#     between them, i.e. `melitz_origin_intervals`'s own `rank[d1]==rank[d2]`)
#         => T_od1(p,.) == T_od2(p,.) IDENTICALLY, for EVERY p (not merely at the solution)
#         => H_od1 == H_od2 required for ANY p to satisfy both moments, i.e. h_od1==h_od2
#                                                                                     (same-bin)
#
#   This is EXACTLY `melitz_origin_block_monotonicity_check`'s own necessary condition
#   (`origin_block_screen.jl`), re-derived here in `h`-space and promoted from a diagnostic
#   check into genuine LINEAR KNITRO constraints on the middle loop's free A/H coordinates
#   (Phase 2 below) -- the reason to do this rather than rely purely on the inner solve's own
#   post-hoc infeasibility signal is exactly the governing prompt's own stated rationale: an
#   inner solve that ignores gravity would happily walk into a same-bin-violating A and only
#   discover infeasibility (a large/`AboveEvaluationCap`/`InfiniteDeltaCertified` objective,
#   with NO informative local gradient) after the fact, whereas registering the necessary
#   condition as a true linear pre-constraint keeps KNITRO's trust region inside (or at the
#   boundary of) the region where a finite, differentiable optimum can even exist.
#
# A-GRAVITY: `dot(c_full, vec(log A)) = 0` is an ALGEBRAIC IDENTITY of `ctx.A_pivot`'s own
# `pivot_expand` (`g0=0` always, per `equilibrium.jl`/`delta_star.jl:110`), for ANY `A_free` --
# not a constraint the middle loop needs to add, as long as the middle loop's own free
# coordinates are mapped to the full `A` matrix via the SAME `ctx.A_pivot` (Phase 1.A) or an
# affine reparameterization of it (Phase 1.B) -- confirmed exactly this way in doc 1
# (`melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md`, Section 2) and re-verified
# by this session's own tests below.
#
# F-GRAVITY REDUNDANCY (governing prompt's explicit ask: "derive whether f-gravity is exactly
# redundant conditional on fixed q (reconstructed through q-gravity) and A-gravity"):
#
#   `build_q_gravity_offset(q_jj, ctx)` (`log_cutoff_param.jl`, 2026-07-29 fix) computes
#   `g0_q` as a function of `q_jj` (hence of `g` alone) and FIXED DATA ONLY -- it no longer
#   reads `A` at all (the 2026-07-29 session proved `dot(c_full,vec(logA))==0` identically, so
#   the old `A`-dependent term was dropped). This means: for a FIXED `q` (reconstructed once,
#   at a FIXED `g`, via the ordinary q-gravity pivot) and ANY `A_free` produced by the SAME
#   `ctx.A_pivot`, the identity `dot(c_full[f_free_lin],q_free_full) + c_full[jj_lin]*q_jj ==
#   -g0_q(q_jj)` holds UNCHANGED regardless of which `A_free` the middle loop is currently
#   trying (since it was already true at the anchor `A_free`, and `g0_q` never reads `A`).
#   Combined with `dot(c_full,vec(loga))==0` (A-gravity, an identity for ANY `A_free`) and the
#   ALGEBRAIC relation `dot(c_full,vec(logf)) = (sigma-1)*dot(c_full,vec(q)) +
#   (sigma-1)*dot(c_full,vec(loga)) + dot(c_full,const_vec)` (doc 1, Section 2 -- derived by
#   substituting `melitz_log_f_from_q` into the f-gravity restriction), f-gravity
#   (`dot(c_full,vec(logf))=0`) is therefore an EXACT ALGEBRAIC IDENTITY throughout the ENTIRE
#   middle-loop A-search, for EVERY `A_free` tried, not merely at the anchor -- **f-gravity is
#   exactly redundant given fixed q (reconstructed via q-gravity) and A-gravity; no additional
#   fixed-q linear restriction is needed for it.** Verified numerically below (test file), not
#   merely argued.
#
# PARTICIPATION INVARIANT: since `q` never moves within the middle loop (by construction --
# `melitz_fixed_q_state_theta` below always recovers `f` from the SAME fixed target `q` via
# `melitz_log_f_from_q`'s exact inverse, so `melitz_baseline_cutoff(A_new,f_new,...)`
# reproduces `q` bit-for-bit regardless of `A_new`), the active set at every `(o,d)` cell
# (`z[w,o] > exp(q_od)`) is IDENTICALLY FIXED throughout the middle-loop search -- zero
# participation switches by CONSTRUCTION, not merely as an empirically-verified property. This
# is exactly what makes the existing exact envelope-theorem A-gradient
# (`melitz_exact_a_gradient_full!`/`_free`, `exact_a_gradient.jl`, doc 1) valid at EVERY
# middle-loop trial point where the inner solve returns `FiniteSolved`, with no re-derivation
# needed.
#
# SCOPE LIMITATIONS (disclosed, not glossed over):
#   1. The focal origin's autarky "virtual (D+1)-th destination" breakpoint (`o==target_country`
#      only) is NOT included as an extra ordering restriction -- it does not correspond to a
#      genuine free `(o,d)` A-cell in the SAME sense (the autarky good's own technology is
#      `A[j,j]`, already covered by the ordinary same-bin/ordering rows for origin `j`'s
#      destinations; the autarky cutoff itself has no counterpart A-cell to order against). At
#      the audited real-D20 cliff (origin 14=Korea != focal=fra), this is provably immaterial
#      (doc 2's own Phase 7 confirms the focal-link row plays no role in origin 14's
#      infeasibility). A future extension for cliffs at the focal origin itself would need to
#      add this row; flagged, not built here.
#   2. Only the MINIMAL adjacent-pair generating set (per origin, `D-1` rows) is registered,
#      not all `O(D^2)` pairs -- transitivity of `>=`/`==` along the sorted-rank chain makes
#      the full pairwise set implied by the adjacent one; registering only the minimal set
#      keeps the middle KNITRO problem's constraint count at `O(D^2)` total (`D*(D-1)` rows)
#      rather than `O(D^3)`.
# ============================================================================

using LinearAlgebra: dot

# ----------------------------------------------------------------------------------------
# Phase 0 primitives: the exact h_od <-> a_od affine relation.
# ----------------------------------------------------------------------------------------

"""
    melitz_h_constant_od(o, d, ctx) -> const_od

`const_od = log(lambda_od) + (sigma-1)*log(markup*w_o*tau_od)`, `lambda_od =
X_data[o,d]/expenditure[d]` -- the fixed (data-only) constant in `h_od = const_od -
(sigma-1)*log(A_od)` (module header Phase 0). `-Inf` if `X_data[o,d]==0` (zero observed
trade -- `H_od=0` is then trivially satisfied by `T_od=0`, matching
`lfd_preserving_state.jl`'s own `:zero_tail_anchor` treatment; callers building constraint
rows should guard against this exactly as that module does).
"""
function melitz_h_constant_od(o::Int, d::Int, ctx)
    sigma = ctx.sigma
    markup = melitz_markup(sigma)
    lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
    return log(lambda_od) + (sigma - 1) * log(markup * ctx.w[o] * ctx.tau[o, d])
end

"`melitz_h_od(a_od, o, d, ctx) -> const_od - (sigma-1)*a_od` -- see `melitz_h_constant_od`."
melitz_h_od(a_od::Real, o::Int, d::Int, ctx) = melitz_h_constant_od(o, d, ctx) - (ctx.sigma - 1) * a_od

# ----------------------------------------------------------------------------------------
# Phase 1.A: free log-A coordinates. `a_full = M_A * A_free` EXACTLY (ctx.A_pivot has g0=0
# always), built by applying the existing `pivot_expand` to unit vectors rather than
# re-deriving the pivot formula by hand.
# ----------------------------------------------------------------------------------------

"""
    melitz_A_free_linear_map(ctx) -> M_A::Matrix{Float64}   (D^2 x nA)

`vec(log.(A)) == M_A * A_free` EXACTLY, for ANY `A_free` (an algebraic identity of
`ctx.A_pivot`, `g0=0` always -- module header). `nA=D^2-1` columns; at real D=20,
`400x399`, trivial to materialize densely (`O(D^4)` entries but `D<=20` here).
"""
function melitz_A_free_linear_map(ctx)
    nA = length(ctx.A_pivot.other)
    D2 = ctx.A_pivot.n
    M = zeros(Float64, D2, nA)
    e = zeros(Float64, nA)
    @inbounds for k in 1:nA
        e[k] = 1.0
        M[:, k] = pivot_expand(e, ctx.A_pivot)
        e[k] = 0.0
    end
    return M
end

# ----------------------------------------------------------------------------------------
# Phase 1.B: log-H coordinates. Since the FREE A-cells (`ctx.A_pivot.other`) already ARE the
# search variables (only the single PIVOT physical cell is dependent), h_free is a purely
# DIAGONAL (elementwise, invertible) reparameterization of A_free -- log-H is a per-
# coordinate rescale of log-A into "moment units," not a second independent pivot. The
# PHYSICAL pivot cell's own h-value remains a (now merely AFFINE, not diagonal) function of
# ALL of h_free -- see module header / `melitz_fixed_q_middle_constraint_system` below for
# where this shows up as a genuinely denser constraint row.
# ----------------------------------------------------------------------------------------

"""
    melitz_h_free_constants(ctx) -> Vector{Float64}   (length nA)

`const_free[k] = melitz_h_constant_od` at the physical cell `ctx.A_pivot.other[k]` -- the
per-free-coordinate constant `melitz_h_free_from_A_free`/`melitz_A_free_from_h_free` use.
"""
function melitz_h_free_constants(ctx)
    D = ctx.D
    other = ctx.A_pivot.other
    return [melitz_h_constant_od(lin2od(other[k], D)..., ctx) for k in eachindex(other)]
end

"""
    melitz_h_free_from_A_free(A_free, ctx) -> h_free

Phase 1.B forward map: `h_free[k] = const_free[k] - (sigma-1)*A_free[k]` (diagonal, exact).
"""
function melitz_h_free_from_A_free(A_free::AbstractVector{Float64}, ctx)
    sigma = ctx.sigma
    const_free = melitz_h_free_constants(ctx)
    return const_free .- (sigma - 1) .* A_free
end

"""
    melitz_A_free_from_h_free(h_free, ctx) -> A_free

Phase 1.B inverse map: `A_free[k] = (const_free[k] - h_free[k])/(sigma-1)` (diagonal, exact).
"""
function melitz_A_free_from_h_free(h_free::AbstractVector{Float64}, ctx)
    sigma = ctx.sigma
    const_free = melitz_h_free_constants(ctx)
    return (const_free .- h_free) ./ (sigma - 1)
end

"""
    melitz_gradient_A_free_to_h_free(grad_A_free, ctx) -> grad_h_free

Exact chain rule for the diagonal map: `d(DeltaStar)/d(h_free[k]) =
-(1/(sigma-1))*d(DeltaStar)/d(A_free[k])`, matching the governing prompt's own stated
transform exactly (module header derivation).
"""
melitz_gradient_A_free_to_h_free(grad_A_free::AbstractVector{Float64}, ctx) =
    grad_A_free ./ (-(ctx.sigma - 1))

# ----------------------------------------------------------------------------------------
# Phase 0/2: exact fixed-q ordering/same-bin linear constraint system, in BOTH coordinate
# spaces simultaneously (built from the same M_A + per-cell h-constants).
# ----------------------------------------------------------------------------------------

"""
    MelitzFixedQMiddleConstraintSystem

`n` free coordinates (`nA = D^2-1`). Row `i` reads (in the A-space convention)
`rows_A[i,:] . A_free {>=,==} rhs_A[i]` (`sense[i]`), equivalently (H-space)
`rows_H[i,:] . H_free {>=,==} rhs_H[i]` -- the SAME economic restriction, two coordinate
reparameterizations, so BOTH must certify the same trial point identically feasible/
infeasible (a cross-check the tests exploit). `kind[i]` is `:ordering` or `:same_bin`;
`o_cell[i]`/`d_lo[i]`/`d_hi[i]` record which origin/destination pair produced the row
(`d_lo` has the lower-or-equal cutoff rank, i.e. `h[d_lo] >= h[d_hi]` is the restriction).
"""
struct MelitzFixedQMiddleConstraintSystem
    D::Int
    n::Int
    rows_A::Matrix{Float64}
    rhs_A::Vector{Float64}
    rows_H::Matrix{Float64}
    rhs_H::Vector{Float64}
    sense::Vector{Symbol}
    kind::Vector{Symbol}
    o_cell::Vector{Int}
    d_lo::Vector{Int}
    d_hi::Vector{Int}
end

"""
    melitz_fixed_q_middle_constraint_system(theta_fixed_q, ctx, obj) -> MelitzFixedQMiddleConstraintSystem

Phase 0/1/2: builds the minimal adjacent-pair generating set (module header, SCOPE
LIMITATIONS #2) of linear ordering/same-bin constraints implied by the FIXED q reconstructed
at `theta_fixed_q` (any valid `:logcutoff` theta reproducing the target q -- the rank/same-bin
structure `melitz_origin_intervals` returns depends only on `q` and the fixed draws, never on
`theta`'s own A-block, so ANY theta with the correct q gives byte-identical rows). Skips the
focal origin's autarky virtual destination (SCOPE LIMITATIONS #1).
"""
function melitz_fixed_q_middle_constraint_system(theta_fixed_q::AbstractVector, ctx, obj)
    D = ctx.D
    sigma = ctx.sigma
    M_A = melitz_A_free_linear_map(ctx)
    n = size(M_A, 2)
    const_free = melitz_h_free_constants(ctx)

    rows_A = Vector{Float64}[]
    rows_H = Vector{Float64}[]
    rhs_A = Float64[]
    rhs_H = Float64[]
    sense = Symbol[]
    kind = Symbol[]
    o_cell = Int[]
    d_lo_v = Int[]
    d_hi_v = Int[]

    for o in 1:D
        iv = melitz_origin_intervals(o, theta_fixed_q, ctx, obj)
        rank = iv.rank
        order = sortperm(rank)
        for idx in 1:D-1
            d_lo = order[idx]
            d_hi = order[idx+1]
            @assert rank[d_lo] <= rank[d_hi] "sortperm invariant violated"
            lin_lo = od2lin(o, d_lo, D)
            lin_hi = od2lin(o, d_hi, D)
            row_af = M_A[lin_hi, :] .- M_A[lin_lo, :]   # coefficient of (a_hi - a_lo) on A_free
            const_hi = melitz_h_constant_od(o, d_hi, ctx)
            const_lo = melitz_h_constant_od(o, d_lo, ctx)
            # h_lo - h_hi >= 0 (or ==) <=> (sigma-1)*(a_hi-a_lo) >= const_hi-const_lo
            #                          <=> a_hi - a_lo >= (const_hi-const_lo)/(sigma-1)
            rhs_af = (const_hi - const_lo) / (sigma - 1)
            # H-space: A_free[k] = (const_free[k]-h_free[k])/(sigma-1), so
            # a_hi-a_lo = row_af.const_free/(sigma-1) - row_af.h_free/(sigma-1)
            intercept = dot(row_af, const_free) / (sigma - 1)
            row_hf = (-1.0 / (sigma - 1)) .* row_af
            rhs_hf = rhs_af - intercept

            push!(rows_A, row_af)
            push!(rows_H, row_hf)
            push!(rhs_A, rhs_af)
            push!(rhs_H, rhs_hf)
            sense_k = rank[d_lo] == rank[d_hi] ? :eq : :ge
            push!(sense, sense_k)
            push!(kind, sense_k == :eq ? :same_bin : :ordering)
            push!(o_cell, o)
            push!(d_lo_v, d_lo)
            push!(d_hi_v, d_hi)
        end
    end

    m = length(rhs_A)
    RA = Matrix{Float64}(undef, m, n)
    RH = Matrix{Float64}(undef, m, n)
    for i in 1:m
        RA[i, :] = rows_A[i]
        RH[i, :] = rows_H[i]
    end
    return MelitzFixedQMiddleConstraintSystem(D, n, RA, rhs_A, RH, rhs_H, sense, kind, o_cell, d_lo_v, d_hi_v)
end

"""
    melitz_middle_constraint_residuals(sys, x; coordinate=:logA) -> Vector{Float64}

`row.x - rhs` for every row (`coordinate=:logA` uses `rows_A`/`rhs_A`, `:logH` uses
`rows_H`/`rhs_H`) -- `>=0` (within tolerance) required for `:ordering`/`:ge` rows, `~=0` for
`:same_bin`/`:eq` rows. Used by tests and by the middle-loop diagnostics (Phase 5's "H
ordering/equality residuals").
"""
function melitz_middle_constraint_residuals(sys::MelitzFixedQMiddleConstraintSystem, x::AbstractVector{Float64};
                                             coordinate::Symbol=:logA)
    rows = coordinate == :logH ? sys.rows_H : sys.rows_A
    rhs = coordinate == :logH ? sys.rhs_H : sys.rhs_A
    return rows * x .- rhs
end

# ----------------------------------------------------------------------------------------
# Phase 1: fixed-q state construction (reuses `melitz_f_from_Aq`/`reduce_to_free_theta_logcutoff`
# verbatim -- lfd_preserving_state.jl / log_cutoff_param.jl).
# ----------------------------------------------------------------------------------------

"""
    melitz_fixed_q_state_theta(A_free_middle, q_fixed, gamma_prime_j_fixed, ctx) -> theta_free_middle

Phase 1: builds a valid `:logcutoff` free-theta vector at the CURRENT middle-loop `A_free`,
holding `(g,q)` fixed at their target values. `f` is recovered from the FIXED target `q` and
the CURRENT `A` via `melitz_f_from_Aq` (`lfd_preserving_state.jl`, reused verbatim -- exact
inverse of the baseline-cutoff formula, `melitz_log_f_from_q`). Because f-gravity is
redundant given fixed q + A-gravity (module header), `reduce_to_free_theta_logcutoff`'s own
`q = log.(melitz_baseline_cutoff(A_new,f_new,...))` reproduces `q_fixed` EXACTLY (bit-for-bit
up to floating noise) for EVERY `A_free_middle` -- verified in the test suite, not merely
argued.
"""
function melitz_fixed_q_state_theta(A_free_middle::AbstractVector{Float64}, q_fixed::AbstractMatrix{Float64},
                                     gamma_prime_j_fixed::Float64, ctx)
    D = ctx.D
    logA_full = pivot_expand(A_free_middle, ctx.A_pivot)
    A_new = exp.(reshape(logA_full, D, D))
    f_new = melitz_f_from_Aq(A_new, q_fixed, gamma_prime_j_fixed, ctx)
    return reduce_to_free_theta_logcutoff(A_new, f_new, gamma_prime_j_fixed, ctx)
end

"""
    melitz_fixed_q_state_theta_h(h_free_middle, q_fixed, gamma_prime_j_fixed, ctx) -> theta_free_middle

Phase 1.B convenience: same as `melitz_fixed_q_state_theta` but taking `h_free` (log-H
coordinates) directly, converting via `melitz_A_free_from_h_free` first.
"""
function melitz_fixed_q_state_theta_h(h_free_middle::AbstractVector{Float64}, q_fixed::AbstractMatrix{Float64},
                                       gamma_prime_j_fixed::Float64, ctx)
    A_free_middle = melitz_A_free_from_h_free(h_free_middle, ctx)
    return melitz_fixed_q_state_theta(A_free_middle, q_fixed, gamma_prime_j_fixed, ctx)
end

# ----------------------------------------------------------------------------------------
# Phase 2: middle-loop objective/gradient evaluation -- routes through the ONE authoritative
# typed inner API (`solve_melitz_delta!`), never a bespoke re-implementation.
# ----------------------------------------------------------------------------------------

"""
    MelitzMiddleEvalResult

One classified middle-loop objective evaluation. `Delta` is the VALUE fed to KNITRO (the
true `FiniteSolved.Delta`, or a sentinel for non-`FiniteSolved` classifications -- see
`melitz_middle_objective_value`); `grad_free` is the exact envelope gradient (zero for
non-`FiniteSolved` classifications, since no local optimum exists to differentiate around);
`classification` is the underlying `MelitzInnerResult` (`FiniteSolved`/`AboveEvaluationCap`/
`InfiniteDeltaCertified` -- `NumericalFailure` is NEVER wrapped here, see
`melitz_middle_objective_and_gradient!`'s own docstring).
"""
struct MelitzMiddleEvalResult
    Delta::Float64
    grad_free::Vector{Float64}
    classification::MelitzInnerResult
    theta_free::Vector{Float64}
end

"""
    MELITZ_MIDDLE_INFINITE_SENTINEL

Fixed sentinel objective value fed to KNITRO for an `InfiniteDeltaCertified` middle-loop
trial point (`DeltaStar` is EXACTLY `+Inf` there -- KNITRO cannot accept a non-finite
objective, so a large, fixed, self-consistent constant is used instead, matching this
codebase's own established `AboveEvaluationCap` sentinel convention, e.g.
`finite_delta_outer.jl`'s own `delta_evaluation_cap/delta` fixed-sentinel-with-zero-gradient
pattern). Deliberately much larger than any realistic `CappedEvaluation` cap used in this
session's own experiments (10.0-100.0), so KNITRO always sees `InfiniteDeltaCertified` as
strictly worse than any `AboveEvaluationCap` certificate.
"""
const MELITZ_MIDDLE_INFINITE_SENTINEL = 1.0e6

"""
    melitz_middle_objective_and_gradient!(session, A_free_middle, q_fixed, gamma_prime_j_fixed, ctx;
        coordinate=:logA) -> MelitzMiddleEvalResult

Phase 2 core: builds `theta_free_middle` (`melitz_fixed_q_state_theta`), classifies it via
the ONE authoritative typed inner API (`solve_melitz_delta!`), and returns a middle-loop
objective/gradient pair:

  - `FiniteSolved`: `Delta = result.Delta` (the true value); `grad_free` = the exact
    envelope-theorem A-block gradient (`melitz_exact_a_gradient_full!`/`_free`,
    `exact_a_gradient.jl`) at the verified optimal dual `result.x`, chain-ruled to `:logH` if
    requested (`melitz_gradient_A_free_to_h_free`).
  - `AboveEvaluationCap`: `Delta = result.certified_lower_bound` (a genuine, valid
    weak-duality lower bound -- NOT an arbitrary sentinel), `grad_free = zeros` (no local
    optimum to differentiate around at a capped, non-`FiniteSolved` point).
  - `InfiniteDeltaCertified`: `Delta = MELITZ_MIDDLE_INFINITE_SENTINEL`, `grad_free = zeros`.
  - `NumericalFailure`: NEVER wrapped into a `MelitzMiddleEvalResult` -- this function
    `throw`s a `DomainError` instead (matching `finite_delta_outer.jl`'s/`nuisance_profile.jl`'s
    own established convention: a genuine inner-solve failure is an evaluation ERROR, not a
    value, so KNITRO.jl's callback wrapper converts it to a proper eval-error return code --
    "no `NumericalFailure` may be passed back as a normal middle objective value," per the
    governing prompt).

`coordinate`: `:logA` (default) means `A_free_middle` IS `A_free`; `:logH` means
`A_free_middle` is actually `h_free` (Phase 1.B) and is converted via
`melitz_A_free_from_h_free` before building `theta_free_middle`; `grad_free` is returned in
the SAME coordinate the caller passed in.
"""
function melitz_middle_objective_and_gradient!(session::MelitzInnerSession, A_free_middle::AbstractVector{Float64},
                                                q_fixed::AbstractMatrix{Float64}, gamma_prime_j_fixed::Float64, ctx;
                                                coordinate::Symbol=:logA,
                                                policy::MelitzInnerSolvePolicy=session.policy,
                                                warm_start_source::Symbol=:previous,
                                                origin_block_screen::Bool=false)
    A_free = coordinate == :logH ? melitz_A_free_from_h_free(A_free_middle, ctx) : A_free_middle
    theta_free_middle = melitz_fixed_q_state_theta(A_free, q_fixed, gamma_prime_j_fixed, ctx)

    result = solve_melitz_delta!(session, theta_free_middle, policy;
                                  warm_start_source=warm_start_source,
                                  origin_block_screen=origin_block_screen)

    if result isa FiniteSolved
        D = ctx.D
        state = MelitzExpandedState(D)
        ws_exp = MelitzThetaExpansionWorkspace(D)
        melitz_expand_theta!(state, theta_free_middle, ctx, ws_exp)
        ws_grad = MelitzExactAGradientWorkspace(session.obj.op)
        dDelta_da_full = zeros(D, D)
        melitz_exact_a_gradient_full!(dDelta_da_full, session.obj, result.x, state, ctx, ws_grad)
        grad_A_free = melitz_exact_a_gradient_free(dDelta_da_full, ctx)
        grad_free = coordinate == :logH ? melitz_gradient_A_free_to_h_free(grad_A_free, ctx) : grad_A_free
        return MelitzMiddleEvalResult(result.Delta, grad_free, result, theta_free_middle)
    elseif result isa AboveEvaluationCap
        n = length(A_free_middle)
        return MelitzMiddleEvalResult(result.certified_lower_bound, zeros(n), result, theta_free_middle)
    elseif result isa InfiniteDeltaCertified
        n = length(A_free_middle)
        return MelitzMiddleEvalResult(MELITZ_MIDDLE_INFINITE_SENTINEL, zeros(n), result, theta_free_middle)
    else
        # NumericalFailure (or any other non-enumerated MelitzInnerResult): a genuine
        # inner-solve failure has no certificate of any kind -- an evaluation ERROR, not a
        # value (governing prompt's own explicit requirement).
        throw(DomainError(A_free_middle,
            "melitz_middle_objective_and_gradient!: inner solve returned $(typeof(result)) " *
            "(no certificate) -- rejecting this middle-loop trial point as an evaluation error, " *
            "not a normal objective value."))
    end
end

# ----------------------------------------------------------------------------------------
# Phase 2: solve_melitz_fixed_q_A_profile -- the standalone experimental middle-loop KNITRO
# solver. Pattern mirrors `nuisance_profile.jl`'s own direct "minimize DeltaStar(theta)"
# driver (objective-only nonlinear callback, `Int32[]` constraint indices, linear cutoff-style
# rows registered natively via `KN_add_con_linear_struct`) rather than
# `finite_delta_outer.jl`'s delta-BOUND formulation (no divergence-budget row is needed here --
# the middle objective IS DeltaStar directly). No finite differences over A, no cutoff
# movement (by construction, module header), no dense G (obj is the SAME matrix-free
# `MelitzCCBundle`/`MelitzMomentOperator` production uses).
# ----------------------------------------------------------------------------------------

"""
    MelitzMiddleEvalLogEntry

One evaluation record kept by `solve_melitz_fixed_q_A_profile` (Phase 2's own "record the
inner warm start used at every middle evaluation" / "strict time and evaluation logging"
requirement).
"""
struct MelitzMiddleEvalLogEntry
    call_kind::Symbol        # :fc or :ga
    n_call::Int
    Delta::Float64
    classification::Symbol
    warm_start_source::Symbol
    elapsed_s::Float64
end

"""
    MelitzFixedQAProfileResult

Result of `solve_melitz_fixed_q_A_profile`. `Delta_min` is KNITRO's own terminal objective
(in whichever `coordinate` the problem was solved under); `A_free_final`/`theta_free_final`
are the terminal point converted to plain `A_free`/a ready-to-reuse `:logcutoff` theta;
`r_final` is an INDEPENDENT cold re-verification (fresh warm start, same policy) of the
terminal point -- `r_final` is what should be reported as "the answer," not the raw KNITRO
trajectory value alone (this codebase's own established cold-reverification convention,
`finite_delta_outer.jl`/`nuisance_profile.jl`). `nStatus` is KNITRO's own terminal status.
`n_fc_calls`/`n_ga_calls`/`wall_s` are Phase 7's own cost bookkeeping; `eval_log` is the full
per-call trace.
"""
struct MelitzFixedQAProfileResult
    nStatus::Int
    Delta_min::Float64
    coordinate::Symbol
    x_start::Vector{Float64}
    x_final::Vector{Float64}
    A_free_final::Vector{Float64}
    theta_free_final::Vector{Float64}
    r_final::MelitzInnerResult
    Delta_final_verified::Float64
    n_fc_calls::Int
    n_ga_calls::Int
    n_finite_solved::Int
    n_above_cap::Int
    n_infinite_certified::Int
    wall_s::Float64
    eval_log::Vector{MelitzMiddleEvalLogEntry}
end

"""
    solve_melitz_fixed_q_A_profile(session, q_fixed, gamma_prime_j_fixed, x_start, ctx;
        coordinate=:logA, policy=session.policy, max_evals=150, box=2.0,
        outer_loop_opt=<default>, warm_start_source=:previous, origin_block_screen=false,
        sys=melitz_fixed_q_middle_constraint_system(...)) -> MelitzFixedQAProfileResult

Phase 2: the standalone experimental middle-loop solver. Minimizes `DeltaStar` over the free
A-block (`coordinate=:logA`, `x=A_free`) or its diagonal log-H reparameterization
(`coordinate=:logH`, `x=h_free`), holding `(g,q)` fixed at `(log(gamma_prime_j_fixed),
q_fixed)`, subject to the EXACT fixed-q ordering/same-bin linear constraints
(`melitz_fixed_q_middle_constraint_system`) registered as TRUE KNITRO linear rows (zero
per-iterate evaluation cost, exactly `nuisance_profile.jl`'s/`finite_delta_outer.jl`'s own
`:linear` cutoff-system pattern). `x_start` MUST already be exactly feasible for these linear
constraints (module header: "All starts must be converted to exact feasible middle
coordinates before KNITRO begins" -- this function does NOT project an infeasible start; see
`melitz_project_start_to_middle_constraints` for a caller-side projection helper).

Every inner evaluation is routed through `melitz_middle_objective_and_gradient!` (Phase 2
core) -- `FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified` only; `NumericalFailure`
propagates as a genuine KNITRO evaluation error (caught by KNITRO.jl, not silently absorbed).

`max_evals` bounds BOTH `cb_F!` and `cb_G!` calls combined (KNITRO's native `maxit`/eval-count
control is not separately exposed here; this function enforces the cap directly inside the
callbacks and throws once exceeded, which KNITRO.jl reports as a clean stop). `box`: a
symmetric `x_start .± box` bound in the CHOSEN coordinate's own units, purely for KNITRO
well-posedness (an artificial bound, matching `finite_delta_outer.jl`'s own `theta_box`
convention) -- default `2.0` (log units).
"""
function solve_melitz_fixed_q_A_profile(session::MelitzInnerSession, q_fixed::AbstractMatrix{Float64},
                                         gamma_prime_j_fixed::Float64, x_start::AbstractVector{Float64}, ctx;
                                         coordinate::Symbol=:logA,
                                         policy::MelitzInnerSolvePolicy=session.policy,
                                         max_evals::Int=150,
                                         box::Real=2.0,
                                         outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"),
                                         warm_start_source::Symbol=:previous,
                                         origin_block_screen::Bool=false,
                                         sys::Union{Nothing,MelitzFixedQMiddleConstraintSystem}=nothing,
                                         theta_fixed_q_for_constraints::Union{Nothing,AbstractVector}=nothing)
    coordinate in (:logA, :logH) || throw(ArgumentError("coordinate must be :logA or :logH"))
    t0 = time()
    n = length(x_start)

    if sys === nothing
        theta_fixed_q_for_constraints === nothing && throw(ArgumentError(
            "solve_melitz_fixed_q_A_profile: pass either `sys` or `theta_fixed_q_for_constraints`."))
        sys = melitz_fixed_q_middle_constraint_system(theta_fixed_q_for_constraints, ctx, session.obj)
    end
    rows = coordinate == :logH ? sys.rows_H : sys.rows_A
    rhs = coordinate == :logH ? sys.rhs_H : sys.rhs_A
    size(rows, 2) == n || throw(ArgumentError(
        "solve_melitz_fixed_q_A_profile: x_start has length $n, constraint system expects $(sys.n)"))

    n_fc_calls = Ref(0)
    n_ga_calls = Ref(0)
    n_finite = Ref(0)
    n_cap = Ref(0)
    n_inf = Ref(0)
    eval_log = MelitzMiddleEvalLogEntry[]

    function _eval(x::Vector{Float64}, kind::Symbol)
        t_eval0 = time()
        r = melitz_middle_objective_and_gradient!(session, x, q_fixed, gamma_prime_j_fixed, ctx;
            coordinate=coordinate, policy=policy, warm_start_source=warm_start_source,
            origin_block_screen=origin_block_screen)
        elapsed = time() - t_eval0
        cls = r.classification isa FiniteSolved ? (n_finite[] += 1; :FiniteSolved) :
              r.classification isa AboveEvaluationCap ? (n_cap[] += 1; :AboveEvaluationCap) :
              (n_inf[] += 1; :InfiniteDeltaCertified)
        push!(eval_log, MelitzMiddleEvalLogEntry(kind, length(eval_log) + 1, r.Delta, cls,
                                                   warm_start_source, elapsed))
        return r
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        x = collect(evalRequest.x)
        n_fc_calls[] += 1
        n_fc_calls[] + n_ga_calls[] > max_evals && throw(DomainError(x,
            "solve_melitz_fixed_q_A_profile: max_evals=$max_evals exceeded (Phase 2's own bounded-evaluations requirement)."))
        r = _eval(x, :fc)
        evalResult.obj[1] = r.Delta
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        x = collect(evalRequest.x)
        n_ga_calls[] += 1
        n_fc_calls[] + n_ga_calls[] > max_evals && throw(DomainError(x,
            "solve_melitz_fixed_q_A_profile: max_evals=$max_evals exceeded (Phase 2's own bounded-evaluations requirement)."))
        r = _eval(x, :ga)
        evalResult.objGrad .= r.grad_free
        return 0
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, outer_loop_opt)
    xIndices = melitz_kn_add_vars!(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, x_start .- box)
    KNITRO.KN_set_var_upbnds_all(kc, x_start .+ box)
    KNITRO.KN_set_var_primal_init_values_all(kc, x_start)

    m = length(rhs)
    if m > 0
        cIndices = melitz_kn_add_cons!(kc, m)
        for i in 1:m
            if sys.sense[i] == :eq
                KNITRO.KN_set_con_eqbnd(kc, cIndices[i], rhs[i])
            else
                KNITRO.KN_set_con_lobnd(kc, cIndices[i], rhs[i])
            end
        end
        nnz = m * n
        indexCons_lin = repeat(cIndices, inner=n)
        indexVars_lin = repeat(xIndices, outer=m)
        coefs_lin = vec(permutedims(rows))
        KNITRO.KN_add_con_linear_struct(kc, nnz, indexCons_lin, indexVars_lin, coefs_lin)
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!; nnzJ=0)
    melitz_apply_policy_to_knitro!(kc, policy)

    nStatus = -999
    x_final = copy(x_start)
    Delta_min = NaN
    try
        KNITRO.KN_solve(kc)
        nStatus_ref, objSol, xSol, _ = KNITRO.KN_get_solution(kc)
        nStatus = Int(nStatus_ref)
        Delta_min = Float64(objSol)
        x_final = collect(Float64.(xSol))
    catch e
        @warn "solve_melitz_fixed_q_A_profile: KN_solve terminated via exception (evaluation cap or genuine failure)" exception=(e, catch_backtrace())
    finally
        KNITRO.KN_free(kc)
    end

    A_free_final = coordinate == :logH ? melitz_A_free_from_h_free(x_final, ctx) : x_final
    theta_free_final = melitz_fixed_q_state_theta(A_free_final, q_fixed, gamma_prime_j_fixed, ctx)

    # Cold re-verification (fresh warm start, SAME policy) -- this codebase's own established
    # "the answer is the cold-reverified point, never the raw trajectory value" convention.
    session.obj.use_cached_x = false
    session.obj.x .= NaN
    r_final = solve_melitz_delta!(session, theta_free_final, policy; warm_start_source=:neutral,
                                   origin_block_screen=origin_block_screen)
    Delta_final_verified = r_final isa FiniteSolved ? r_final.Delta :
                            r_final isa AboveEvaluationCap ? r_final.certified_lower_bound :
                            r_final isa InfiniteDeltaCertified ? Inf : NaN

    return MelitzFixedQAProfileResult(nStatus, Delta_min, coordinate, collect(Float64.(x_start)), x_final,
        A_free_final, theta_free_final, r_final, Delta_final_verified,
        n_fc_calls[], n_ga_calls[], n_finite[], n_cap[], n_inf[], time() - t0, eval_log)
end

# ----------------------------------------------------------------------------------------
# Start-point projection: converts a caller-supplied A_free (e.g. an anchor's own A_free, or
# a cellwise-p*-compensated A from `lfd_preserving_state.jl`) into an EXACT feasible middle
# start by greedily reordering (never guessing/rescaling) tied/violating cells within each
# origin's own sorted-rank chain (module header: "All starts must be converted to exact
# feasible middle coordinates before KNITRO begins").
# ----------------------------------------------------------------------------------------

"""
    melitz_project_start_to_middle_constraints(A_free_start, sys, ctx) -> A_free_projected

For each origin's adjacent-pair chain (module header SCOPE #2), walks destinations in rank
order and, whenever a row's residual (`melitz_middle_constraint_residuals`, `:logA`) is
violated (or an equality row is off), moves ONLY the higher-cutoff-rank cell's `a_od` down
(or up, for equality) to exactly satisfy that one row, holding every earlier-processed cell
fixed -- a single forward pass per origin (not a global projection/QP), always yielding an
EXACT feasible point (residuals `~0` in EXACT arithmetic, verified in the tests) since the
pivot-adjoint-free ("other") cells are mutated directly and the map back to `A_free` is
identity outside the single pivot row (`ctx.A_pivot.pivot`'s own physical cell is skipped --
it is never a free coordinate to move directly; if a chain touches the pivot cell, the row is
instead satisfied by adjusting the OTHER (non-pivot) endpoint).
"""
function melitz_project_start_to_middle_constraints(A_free_start::AbstractVector{Float64},
                                                      sys::MelitzFixedQMiddleConstraintSystem, ctx)
    D = ctx.D
    A_free = copy(A_free_start)
    other = ctx.A_pivot.other
    pivot_lin = ctx.A_pivot.pivot
    idx_of_lin = Dict{Int,Int}(other[k] => k for k in eachindex(other))

    m = length(sys.rhs_A)
    for i in 1:m
        lin_lo = od2lin(sys.o_cell[i], sys.d_lo[i], D)
        lin_hi = od2lin(sys.o_cell[i], sys.d_hi[i], D)
        # row_af . A_free = a_hi - a_lo (by construction above); move whichever endpoint is a
        # free ("other") coordinate to satisfy the row exactly, preferring to move d_hi.
        cur = dot(sys.rows_A[i, :], A_free)
        resid = cur - sys.rhs_A[i]
        (sys.sense[i] == :ge && resid >= -1e-12) && continue
        (sys.sense[i] == :eq && abs(resid) < 1e-10) && continue
        need = sys.rhs_A[i] - cur   # amount a_hi must increase (== amount to add to a_hi, or subtract from a_lo)
        if lin_hi != pivot_lin && haskey(idx_of_lin, lin_hi)
            k = idx_of_lin[lin_hi]
            A_free[k] += need
        elseif lin_lo != pivot_lin && haskey(idx_of_lin, lin_lo)
            k = idx_of_lin[lin_lo]
            A_free[k] -= need
        else
            @warn "melitz_project_start_to_middle_constraints: row $i touches the pivot cell on both " *
                  "sides (should not happen -- pivot appears as both d_lo and d_hi target) -- left unprojected."
        end
    end
    return A_free
end
