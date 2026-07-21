# ============================================================================
# Production-candidate integration: exact fast infeasibility screens beyond
# zero-winner rejection.
#
# ADDITIVE ONLY, same discipline as infeasibility_screen.jl (Continuation 9)
# and winner_certificate.jl: does NOT modify the trusted dense constructor
# (moments_gammanorm.jl::EK_moments_gammanorm_directgp!), MinInd!, winners.jl,
# winners_v2.jl, compressed_moments.jl, lfix_incremental.jl, or
# infeasibility_screen.jl itself. Everything here is new; the only production
# integration points are `evaluate_fullA_screened_ranged` (a new opt-in
# entry point mirroring infeasibility_screen.jl::evaluate_fullA_screened) and
# `screen_hard_winners_ranged` (a new function, NOT an in-place edit of
# infeasibility_screen.jl::screen_hard_winners -- see that function's own
# "ADDITIVE ONLY" contract, which this respects by duplicating rather than
# modifying its trusted loop body, exactly as it itself duplicates
# compressed_moments.jl's winner-finding loop).
#
# WHY THIS IS A DIFFERENT FAILURE MODE than infeasibility_screen.jl's
# existing zero-winner screen: infeasibility_screen.jl proves "origin o can
# NEVER win destination d" (win COUNT = 0). This file proves a strictly
# weaker-premise, complementary failure: "origin o CAN win destination d on
# some draws, but even its best possible winning value can never reach the
# observed target trade share" (win count > 0, but win MAGNITUDE
# insufficient). Both are exact one-sided certificates of the SAME underlying
# fact (the primal moment problem is infeasible, minimum divergence = +inf);
# neither implies the other; a production caller wants both.
#
# ----------------------------------------------------------------------------
# SECTION 2 (brief): exact factorization, re-derived and cross-checked
# directly against the two independent authoritative sources in this repo --
# NOT re-derived from a schematic:
#
#   (a) compressed_moments.jl's own header comment (lines 15-38), which
#       states the exact hFunction!-matching formula for the raw bilateral
#       contribution:
#           r_{s,(o,d)} = pTsigma_{s,o,d} * 1{o = winner_{s,d}} - P_{od}*denom_d
#           pTsigma_{s,o,d} = constConsSigma_{o,d} / Usigma_{s,o}^{-mu}
#           constConsSigma_{o,d} = wHat_o^{1-sigma} * (AodPow_{o,d}*tau_{o,d})^{1-sigma}
#           AodPow_{o,d} = (Aod_{o,d}/cHat_{o,d})^{-mu},  Aod_{o,d} = Aod_theta_{o,d} * B(o,d)
#       Substituting Aod_{o,d} = Aod_theta_{o,d}*B(o,d) into AodPow then into
#       constConsSigma gives, after collecting every A_od_theta-independent
#       factor into a single per-cell constant K2(o,d):
#           constConsSigma_{o,d} = K2(o,d) * Aod_theta_{o,d}^{mu*(sigma-1)}
#       so the raw WINNING draw-level contribution factors EXACTLY as
#           h_{od,s}(A) = pTsigma_{s,o,d} = a_{od}(A) * x_{od,s},
#           a_{od}(A) = Aod_theta_{o,d}^{mu*(sigma-1)},   x_{od,s} = K2(o,d)/Usigma_{s,o}^{-mu}
#       with x_{od,s} COMPLETELY independent of A (data-only) IF AND ONLY IF
#       mu and sigma are themselves fixed across the outer loop (they enter
#       the exponent and Usigma_{s,o}^{-mu} too) -- verified below, not
#       assumed.
#   (b) context_real_d20.jl::d20_real_setup, lines 75-90 (this repo's real
#       D=20 production context builder, the SAME one c10_d20_production_
#       driver.jl / this integration branch's ported driver call): theta_lo[1]
#       == theta_hi[1] (mu) and theta_lo[2] == theta_hi[2] (sigma) are set
#       EQUAL, and neither index 1 nor 2 appears in `free_idx` -- mu, sigma
#       are literally box-constrained to a single point and never touched by
#       the outer optimizer. `free_idx = vcat(3+Dact, Aod_offset+1:Aod_offset+
#       Dact^2)` and `Aod_offset = 3+Dact`, `l_full = length(theta0_up)` where
#       `theta0_up = build_theta_gammanorm(...)` has length D^2+D+3 (mu,sigma,
#       D gammas, gamma_prime_focal, D^2 A_od) -- so `Aod_offset+Dact^2 ==
#       l_full` EXACTLY: A_od is confirmed (not assumed) to be the trailing
#       D^2 block of theta_full for every ctx this repo's real-data driver
#       builds. (Re-verify this equality at every ctx via the assertion in
#       `precompute_envelope` below -- do not hardcode the offset.)
#
#   usePMM / NormalizeMoments / SamplingWeights uniformity: grepped live
#   across every params NamedTuple in this repo that reaches
#   d20_real_setup/master_setup (AD_PARAMS and every other params tuple
#   found): `usePMM=0`, `NormalizeMoments=0`, `importanceSampling=0`
#   (=> SamplingWeights uniform) in EVERY config that exists in this
#   codebase today -- there is no live config with usePMM=1. The envelope
#   screen below still asserts `usePMM==0` explicitly at every
#   `precompute_envelope` call (raising `EnvelopeUnsupportedContext`, not
#   silently computing a wrong bound) rather than hardcoding that fact,
#   since a future config could set it. NormalizeMoments does NOT need a
#   guard: `nrm[j]` is a context-level (data-only, theta-independent)
#   constant regardless of its value, computed once here exactly as
#   compressed_moments.jl computes it -- confirmed by reading
#   compressed_moments.jl's own nrm construction (a function of
#   `gamma.sigma_Moments`/`gamma.moments_without_var`/`indicators.
#   NormalizeMoments`, none of which vary across the outer loop).
#
#   COMMON-MARGINAL CONTEXTS: `d20_real_setup`/`context_real_d20.jl` (the
#   context builder this whole file targets) has NO CM_L/CM_ENABLED wiring
#   at all -- common-marginals support lives only on a DIFFERENT, separately
#   unmerged branch (`integration/fullA-d20-common-marginals`) that has not
#   been reconciled with this branch's base commit. This file therefore does
#   NOT claim CM support: `precompute_envelope`/`range_screen_standalone`
#   below operate ONLY over inner-dual columns `1:ctx.obj.outer_constr_index-1`
#   exactly as the base context defines them (verified: no CM columns exist
#   in that range for this ctx). If/when CM lands on the same base as this
#   branch, the CM moments (theta-independent, appended after the bilateral
#   block per the CM wiring described in production-consolidation memory)
#   need their OWN range certificate derived and added here explicitly --
#   NOT assumed to be covered by the machinery in this file. See the
#   production integration doc's support/guard table.
# ============================================================================

isdefined(Main, :CompressedFactual) || include(joinpath(@__DIR__, "compressed_moments.jl"))
isdefined(Main, :precompute_pairwise_M) || include(joinpath(@__DIR__, "infeasibility_screen.jl"))
using SpecialFunctions: gamma as spgamma

# ============================================================================
# SECTION 3: pre-winner O(D^2), draw-free envelope certificate.
# ============================================================================

"Raised by `precompute_envelope` when the current ctx's config cannot be certified safe for the envelope screen (see file header). Callers must catch this ONCE at context-build time and disable the envelope screen (pass `envelope=nothing` downstream) -- never retry per outer point, and never apply a stale precomputation from a different config."
struct EnvelopeUnsupportedContext <: Exception
    reason::String
end

"""
    EnvelopePrecomp

One-time, context-level precomputation for `envelope_prewinner_screen` /
`screen_hard_winners_ranged`. Built ONCE per `ctx` (mu, sigma, and all other
data fields are fixed across the outer loop -- see file header) and reused
for every subsequent outer-point evaluation; O(D^2 + W*D) to build, O(D^2) to
query.

- `exponent = mu*(sigma-1)`: asserted > 0 at construction (mu>0, sigma>1 in
  every config this repo runs; if a future config violates this the
  monotonicity argument in `envelope_prewinner_screen`'s docstring fails and
  this screen must not be used -- the assertion catches that explicitly).
- `K2[o,d]`: the data-only constant s.t. `constConsSigma[o,d] =
  K2[o,d]*Aod_theta[o,d]^exponent` (re-derived from compressed_moments.jl's
  own AodPow/constConsSigma construction, see file header).
- `M[o,d] = max_s (K2[o,d]/UsigmaPow[s,o])`: the data-only upper envelope of
  `x_{od,s}` over all W draws.
- `b[o,d] = Pmat[o,d]*denom[d]`: the target (data-only).
- `gdiv`, `nrm`: the SAME per-column post-processing factors
  `compressed_moments.jl`/`range_screen_standalone` use.
- `SWmax`: sampling-weight max (conservative upper bound; exact when SW is
  uniform, which every live config in this repo is -- see file header).
"""
struct EnvelopePrecomp
    D::Int
    exponent::Float64
    K2::Matrix{Float64}
    M::Matrix{Float64}
    b::Matrix{Float64}
    gdiv::Vector{Float64}
    nrm::Vector{Float64}
    SWmax::Float64
    Pmat::Matrix{Float64}
end

"""
    precompute_envelope(ctx) -> EnvelopePrecomp

Builds the one-time envelope precomputation, or throws `EnvelopeUnsupportedContext`
if this ctx's config is not one the derivation above covers (currently:
`usePMM==1`, or mu/sigma not literally fixed to a point). Callers: catch this
ONCE at context-build time (see `evaluate_fullA_screened_ranged`'s own
handling), not per outer-point call.
"""
function precompute_envelope(ctx)
    γo = ctx.γ; D = ctx.D; U = ctx.U; W = size(U, 1)
    ctx.θ_lo[1] == ctx.θ_hi[1] || throw(EnvelopeUnsupportedContext("mu is not fixed to a point in this ctx (theta_lo[1] != theta_hi[1]) -- the a_od(A) factorization this screen relies on assumes mu is outer-loop-constant; re-derive before using this screen with mu free."))
    ctx.θ_lo[2] == ctx.θ_hi[2] || throw(EnvelopeUnsupportedContext("sigma is not fixed to a point in this ctx (theta_lo[2] != theta_hi[2]) -- same issue as mu, see above."))
    ind = γo.indicators
    ind.usePMM == 0 || throw(EnvelopeUnsupportedContext("usePMM==1 in this ctx: the envelope bound below omits the -usePMM*PMM[j] term compressed_moments.jl's own moment formula applies, so it would UNDERSTATE the true column value and could produce a false-positive infeasibility claim. Not derived/validated for usePMM==1 -- disabling rather than guessing."))

    μ = ctx.θ_lo[1]; σ = ctx.θ_lo[2]
    exponent = μ * (σ - 1)
    exponent > 0 || throw(EnvelopeUnsupportedContext("mu*(sigma-1) = $exponent is not > 0 (need mu>0, sigma>1) -- the nonnegativity/monotonicity argument this screen relies on does not hold; re-derive before using."))

    lambda = reshape(γo.P, (D, D))'
    B = γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    C1 = (B ./ γo.cHat) .^ (-μ)
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    K2 = [wPow[o] * γo.τ[o, d]^(1 - σ) * C1[o, d]^(1 - σ) for o in 1:D, d in 1:D]
    all(>=(0), K2) || throw(EnvelopeUnsupportedContext("K2(o,d) has a negative entry -- the raw-contribution nonnegativity argument this screen relies on does not hold for this ctx's data; re-derive before using."))

    UσPow = γo.Uσ .^ (-μ)
    Cinv = [1.0 / minimum(@view UσPow[:, o]) for o in 1:D]   # = max_s(1/UσPow[s,o])
    M = [K2[o, d] * Cinv[o] for o in 1:D, d in 1:D]

    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]
    Pmat = [γo.P[d + (o - 1) * D] for o in 1:D, d in 1:D]
    b = [Pmat[o, d] * denom[d] for o in 1:D, d in 1:D]
    all(>=(0), b) || throw(EnvelopeUnsupportedContext("target b(o,d) has a negative entry -- unexpected for this ctx's data; re-derive before using."))

    ctx.Aod_offset + D^2 == length(ctx.θ0_up) || throw(EnvelopeUnsupportedContext("A_od is not the trailing D^2 block of theta_full for this ctx (Aod_offset+D^2 != l_full) -- re-check the offset convention before using this screen."))
    oci = ctx.obj.outer_constr_index
    ncol = oci - 1
    gammafac = spgamma(μ * (1 - σ) + 1)
    gdiv = [j <= D^2 + 1 ? 1.0 / gammafac : 1.0 for j in 1:ncol]
    NM = ind.NormalizeMoments
    without = γo.moments_without_var
    nrm = [(NM == 1 && !(j in without)) ? 1.0 / γo.σ_Moments[j] : 1.0 for j in 1:ncol]
    SWmax = maximum(γo.SamplingWeights[1:W])

    return EnvelopePrecomp(D, exponent, K2, M, b, gdiv, nrm, SWmax, Pmat)
end

struct EnvelopeScreenResult
    status::Symbol   # :EXACT_INFEASIBLE_PREWINNER_ENVELOPE or :INCONCLUSIVE
    column::Int
    origin::Int
    destination::Int
    h_upper_bound::Float64
    target::Float64
    margin_normalized::Float64
    tol::Float64
end
_no_envelope_hit() = EnvelopeScreenResult(:INCONCLUSIVE, 0, 0, 0, NaN, NaN, NaN, NaN)

"""
    envelope_prewinner_screen(θ_full, ctx, ep::EnvelopePrecomp; safety_mult=50.0) -> EnvelopeScreenResult

O(D^2) SCALAR check, no draw access, no winner construction, no moment
construction: for every `(o,d)` with `Pmat[o,d]>0`, computes
`a_od = Aod_theta[o,d]^exponent`, then the fully-normalized centered upper
bound `g_upper = SWmax*nrm[j]*((a_od*M[o,d] - b[o,d])*gdiv[j])`. Because
`a_od*M[o,d] >= max_s x_{od,s}(A) = max_s h_{od,s}(A)` (an upper bound on the
best possible winning value, whether or not `o` actually wins any draw --
the true winning value can only be lower), `g_upper < -tol` is an EXACT
one-sided certificate that the true column max is also `< -tol`.

Only certifies the NEGATIVE-side violation (see file header's structural
nonnegativity argument, re-verified live via the `K2>=0`/`b>=0` assertions
in `precompute_envelope`) -- returns `:INCONCLUSIVE`, NEVER a feasibility
claim, if no cell is certified.
"""
function envelope_prewinner_screen(θ_full::AbstractVector, ctx, ep::EnvelopePrecomp; safety_mult::Float64 = 50.0)
    D = ep.D
    @assert ctx.Aod_offset + D^2 == length(θ_full) "envelope_prewinner_screen: A_od is not the trailing D^2 block of theta_full for this ctx -- re-check the offset convention before trusting this screen"
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    for d in 1:D, o in 1:D
        ep.Pmat[o, d] > 0 || continue
        j = d + (o - 1) * D
        a_od = Aod_θ[o, d]^ep.exponent
        h_upper = a_od * ep.M[o, d]
        g_upper = ep.SWmax * ep.nrm[j] * ((h_upper - ep.b[o, d]) * ep.gdiv[j])
        colscale = max(h_upper, ep.b[o, d], 1.0)
        tol = safety_mult * eps(Float64) * colscale
        if g_upper < -tol
            return EnvelopeScreenResult(:EXACT_INFEASIBLE_PREWINNER_ENVELOPE, j, o, d, h_upper, ep.b[o, d], -g_upper / colscale, tol)
        end
    end
    return _no_envelope_hit()
end

# ============================================================================
# SECTION 4: fused exact winning-maximum certificate inside hard-winner
# construction. Single pass: reproduces infeasibility_screen.jl::
# screen_hard_winners' own price-comparison formula and zero-win check
# (attributed reuse, same convention that function itself uses for
# compressed_moments.jl), PLUS tracks H^max_{od} in the SAME per-draw loop
# that already computes winner[s,d]/wval[s,d] -- no second winner scan.
# ============================================================================

struct WinnerRangeScreenResult
    feasible::Bool
    reject_kind::Symbol            # :none, :zero_winner, or :winning_range
    stage::Int
    failing_o::Int
    failing_d::Int
    Hmax_failing::Float64          # NaN unless reject_kind==:winning_range
    target_failing::Float64        # NaN unless reject_kind==:winning_range
    order::Vector{Int}
    winner::Union{Nothing,Matrix{Int}}
    wval::Union{Nothing,Matrix{Float64}}
    win_counts::Matrix{Int}
    Hmax::Matrix{Float64}          # D(origin) x D(destination); -Inf where origin never won (or unscanned)
end

"""
    screen_hard_winners_ranged(θ_full, ctx, Pmat, ep::EnvelopePrecomp; order=1:D, full_scan=false, safety_mult=50.0) -> WinnerRangeScreenResult

Destination-major hard-winner construction -- SAME price formula as
`infeasibility_screen.jl::screen_hard_winners` (direct, attributed reuse,
not re-derived). Within the SAME per-draw loop that already computes
`winner[s,d]`/`wval[s,d]` (no second scan), also tracks the running maximum
`Hmax[o,d] = max_{s: winner[s,d]==o} wval[s,d]` -- exactly
`H^max_{od} = max_{s:w_{sd}=o} h_{od,s}(A)`.

After EACH destination completes, checks (in order) for every origin `o`
with `Pmat[o,d]>0`:
  1. zero wins (`win_counts[o,d]==0`) -- SAME rejection as `screen_hard_
     winners` (`reject_kind=:zero_winner`);
  2. positive wins but `Hmax[o,d] < b_{od}` by more than a conservative
     normalized tolerance -- the NEW certificate this file adds
     (`reject_kind=:winning_range`, status `EXACT_INFEASIBLE_WINNING_RANGE`
     at the caller level).
Either rejection returns immediately without touching any later destination
in `order` (same early-exit discipline as `screen_hard_winners`).
"""
function screen_hard_winners_ranged(θ_full::AbstractVector, ctx, Pmat::AbstractMatrix, ep::EnvelopePrecomp;
        order::AbstractVector{Int} = 1:ctx.D, full_scan::Bool = false, safety_mult::Float64 = 50.0)
    γo = ctx.γ
    D = ctx.D; U = ctx.U; W = size(U, 1)
    μ = θ_full[1]; σ = θ_full[2]
    lambda = reshape(γo.P, (D, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    Aod = Aod_θ .* γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ γo.cHat) .^ (-μ)
    constCons = [γo.wHat[o] * AodPow[o, d] * γo.τ[o, d] for o in 1:D, d in 1:D]
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    constConsσ = [wPow[o] * (AodPow[o, d] * γo.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
    UPow = U .^ (-μ)
    UσPow = γo.Uσ .^ (-μ)
    denom = [γo.wHat[dd] * γo.L[dd] for dd in 1:D]

    winner = Matrix{Int}(undef, W, D)
    wval = Matrix{Float64}(undef, W, D)
    win_counts = zeros(Int, D, D)
    Hmax = fill(-Inf, D, D)

    for (stage, d) in enumerate(order)
        wc = zeros(Int, D)
        Hmax_d = fill(-Inf, D)
        @inbounds for s in 1:W
            best = constCons[1, d] / UPow[s, 1]; bo = 1
            for o in 2:D
                p = constCons[o, d] / UPow[s, o]
                if p < best
                    best = p; bo = o
                end
            end
            winner[s, d] = bo
            hval = constConsσ[bo, d] / UσPow[s, bo]
            wval[s, d] = hval
            # SINGLE-PASS fusion: track the running winning-branch maximum for the
            # winning origin, no separate scan of the draws.
            hval > Hmax_d[bo] && (Hmax_d[bo] = hval)
            for o in 1:D
                p = constCons[o, d] / UPow[s, o]
                p <= best && (wc[o] += 1)
            end
        end
        win_counts[:, d] .= wc
        Hmax[:, d] .= Hmax_d

        if !full_scan
            # 1) zero-winner rejection (same as screen_hard_winners)
            for o in 1:D
                if Pmat[o, d] > 0 && wc[o] == 0
                    return WinnerRangeScreenResult(false, :zero_winner, stage, o, d, NaN, NaN,
                                                    collect(order), nothing, nothing, win_counts, Hmax)
                end
            end
            # 2) winning-range rejection (new): positive wins, but the best winning
            # value can never reach the target -- conservative normalized tolerance,
            # same scale convention as envelope_prewinner_screen/range_screen_standalone.
            j0 = d  # column base, j = d + (o-1)*D
            for o in 1:D
                Pmat[o, d] > 0 || continue
                b_od = Pmat[o, d] * denom[d]
                j = d + (o - 1) * D
                colscale = max(Hmax_d[o], b_od, 1.0)
                tol = safety_mult * eps(Float64) * colscale
                if Hmax_d[o] < b_od - tol
                    return WinnerRangeScreenResult(false, :winning_range, stage, o, d, Hmax_d[o], b_od,
                                                    collect(order), nothing, nothing, win_counts, Hmax)
                end
            end
        end
    end

    if full_scan
        for d in 1:D, o in 1:D
            if Pmat[o, d] > 0 && win_counts[o, d] == 0
                return WinnerRangeScreenResult(false, :zero_winner, D, o, d, NaN, NaN,
                                                collect(order), winner, wval, win_counts, Hmax)
            end
        end
        for d in 1:D, o in 1:D
            Pmat[o, d] > 0 || continue
            b_od = denom[d] * Pmat[o, d]
            colscale = max(Hmax[o, d], b_od, 1.0)
            tol = safety_mult * eps(Float64) * colscale
            if Hmax[o, d] < b_od - tol
                return WinnerRangeScreenResult(false, :winning_range, D, o, d, Hmax[o, d], b_od,
                                                collect(order), winner, wval, win_counts, Hmax)
            end
        end
    end

    return WinnerRangeScreenResult(true, :none, D, 0, 0, NaN, NaN, collect(order), winner, wval, win_counts, Hmax)
end

# ============================================================================
# SECTION 5: general exact range-screen safety net -- operates ONLY on an
# already-built CompressedFactual (never rebuilds one; no separate O(W*D)
# winner pass beyond what building `cf` already required). Ported near-
# verbatim from the reviewed experimental branch (diag/fullA-d20-range-
# screen-review @ 0fd2bf8), which itself fixed a real O(W*D^2) rescan bug in
# an earlier draft -- kept as the true O(W*D) single-pass form here.
# ============================================================================

"""
    MomentRangeCertificate

Exact certificate for one strictly one-sided moment column (inner-dual
columns only, `1:cf.oci-1` -- the gravity moment at `obj.d` is a separate,
outer-loop-only column and is never touched by this screen, matching
`compressed_moments.jl`'s own `oci-1` scope).
"""
struct MomentRangeCertificate
    column::Int
    kind::Symbol           # :bilateral or :counterfactual
    origin::Int
    destination::Int
    sign::Symbol            # :positive or :negative
    min_val::Float64
    max_val::Float64
    margin::Float64
    margin_normalized::Float64
    tol::Float64
    draw_index::Int
end

struct RangeScreenResult
    status::Symbol          # :EXACT_INFEASIBLE_MOMENT_RANGE or :INCONCLUSIVE
    certificate::Union{Nothing,MomentRangeCertificate}
    all_certificates::Vector{MomentRangeCertificate}
    n_columns_checked::Int
    wall_time::Float64
end

_column_label(j::Int, D::Int) = j == D^2 + 1 ? (:counterfactual, 0, 0) : (:bilateral, div(j - 1, D) + 1, mod1(j, D))

function _certificate_from_range(j::Int, D::Int, min_val::Float64, max_val::Float64, tol::Float64, scale::Float64,
        min_idx::Int, max_idx::Int)
    kind, o, d = _column_label(j, D)
    if min_val > tol
        return MomentRangeCertificate(j, kind, o, d, :positive, min_val, max_val, min_val, min_val / max(scale, eps(Float64)), tol, min_idx)
    elseif max_val < -tol
        return MomentRangeCertificate(j, kind, o, d, :negative, min_val, max_val, -max_val, -max_val / max(scale, eps(Float64)), tol, max_idx)
    end
    return nothing
end

"First draw index where origin o does NOT win destination d (report-only, not on the hot path)."
function _first_nonwinner(cf::CompressedFactual, o::Int, d::Int, W::Int)
    @inbounds for s in 1:W
        cf.winner[s, d] != o && return s
    end
    return 0
end

"""
    range_screen_standalone(cf::CompressedFactual; safety_mult=50.0) -> RangeScreenResult

O(W*D) exact range screen over all `cf.oci-1` inner moment columns, operating
on an ALREADY-BUILT `CompressedFactual` -- never rebuilds `cf`, never adds a
separate winner pass beyond what `cf.winner`/`cf.wval` already give it. Safety
net alongside the pre-winner envelope / fused winner-range screen: covers
BOTH the negative-side bilateral case those already catch AND the positive-
side / counterfactual-column case they structurally do not (see file
header).
"""
function range_screen_standalone(cf::CompressedFactual; safety_mult::Float64 = 50.0)
    t0 = time()
    D = cf.D; W = cf.W; ncol = cf.oci - 1
    SWmin, SWmax = extrema(cf.SW)

    win_counts = zeros(Int, D, D)
    col_min = fill(Inf, ncol); col_max = fill(-Inf, ncol)
    col_min_idx = zeros(Int, ncol); col_max_idx = zeros(Int, ncol)

    @inbounds for d in 1:D
        for s in 1:W
            o = cf.winner[s, d]
            win_counts[o, d] += 1
            j = d + (o - 1) * D
            r = cf.wval[s, d] - cf.Pmat[o, d] * cf.denom[d]
            v = cf.SW[s] * cf.nrm[j] * (r * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
            if v < col_min[j]
                col_min[j] = v; col_min_idx[j] = s
            end
            if v > col_max[j]
                col_max[j] = v; col_max_idx[j] = s
            end
        end
    end

    @inbounds for d in 1:D, o in 1:D
        win_counts[o, d] == W && continue
        j = d + (o - 1) * D
        K = cf.nrm[j] * (-cf.Pmat[o, d] * cf.denom[d] * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        lo = K >= 0 ? SWmin * K : SWmax * K
        hi = K >= 0 ? SWmax * K : SWmin * K
        if lo < col_min[j]
            col_min[j] = lo
            col_min_idx[j] = _first_nonwinner(cf, o, d, W)
        end
        if hi > col_max[j]
            col_max[j] = hi
            col_max_idx[j] = _first_nonwinner(cf, o, d, W)
        end
    end

    if cf.cf_col > 0
        j = cf.cf_col
        lo = Inf; hi = -Inf; lo_i = 0; hi_i = 0
        @inbounds for s in 1:W
            v = cf.SW[s] * cf.nrm[j] * (cf.cf_raw[s] * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
            if v < lo; lo = v; lo_i = s; end
            if v > hi; hi = v; hi_i = s; end
        end
        col_min[j] = lo; col_max[j] = hi; col_min_idx[j] = lo_i; col_max_idx[j] = hi_i
    end

    scale = max(maximum(abs, cf.wval), maximum(abs, cf.Pmat .* cf.denom'))
    if cf.cf_col > 0 && !isempty(cf.cf_raw)
        scale = max(scale, maximum(abs, cf.cf_raw))
    end
    tol = safety_mult * eps(Float64) * scale

    certs = MomentRangeCertificate[]
    for j in 1:ncol
        c = _certificate_from_range(j, D, col_min[j], col_max[j], tol, scale, col_min_idx[j], col_max_idx[j])
        c !== nothing && push!(certs, c)
    end

    status = isempty(certs) ? :INCONCLUSIVE : :EXACT_INFEASIBLE_MOMENT_RANGE
    return RangeScreenResult(status, isempty(certs) ? nothing : certs[1], certs, ncol, time() - t0)
end

"""
    evaluate_fullA_screened_compressed_with_cf(x_free, θ_full, ctx, cf::CompressedFactual; warm=true, tag="") -> (result, prof_meta)

Byte-identical body to `infeasibility_screen.jl::evaluate_fullA_screened_compressed`
EXCEPT it takes an ALREADY-BUILT `cf` directly instead of calling
`compressed_factual_from_screen` internally -- lets a caller (this file's
`evaluate_fullA_screened_ranged`) build `cf` exactly ONCE and both (a) run
`range_screen_standalone` on it and (b) feed it to the real inner solve,
instead of building it twice. Attributed, near-verbatim duplication of that
function's post-cf-construction logic (same non-interference rationale that
function's own docstring states: this file does not modify infeasibility_
screen.jl).
"""
function evaluate_fullA_screened_compressed_with_cf(x_free::AbstractVector{Float64}, θ_full::AbstractVector,
        ctx, cf::CompressedFactual; warm::Bool = true, tag::String = "")
    obj = ctx.obj
    t_total0 = time()
    if !warm
        obj.x .= NaN
    end

    W = size(obj.U, 1)
    SW = ctx.γ.SamplingWeights[1:W]
    obj.H[:, 1] .= θ_full[3 + ctx.D] .* SW
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest
    grav_raw = compressed_gravity_raw(θ_full, ctx)

    st = CompressedCBState(obj, cf, grav_raw, false)
    t_inner0 = time()
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed(obj, st)
    t_inner = time() - t_inner0

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end
    inner_x = x
    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        K_hard = obj.H_save
    else
        obj.x .= NaN
        K_hard = -1e10
    end

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        elapsed = (total = time() - t_total0, inner = t_inner, post = 0.0)
        result = (x_free = collect(x_free), θ_full = θ_full,
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, ctx.D),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, moment_resid = Float64[], max_abs_moment_resid = NaN,
                  zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
                  weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
                  winner_hash = UInt64(0), inner_status = nStatus, inner_iters = inner_iters,
                  primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
                  elapsed = elapsed, error_reason = "inner solve failed: nStatus=$nStatus")
        prof_meta = (n_inner_solves = 1, n_inner_infeasible = 1, n_inner_iters = inner_iters,
                     n_fg_calls = n_fg, n_hess_calls = n_hess)
        return result, prof_meta
    end

    if !st.dense_materialized
        ncolI = st.cf.oci - 1
        materialize_dense_factual_structured!(@view(obj.H[:, 3:2+ncolI]), st.cf)
        fill_gravity_column!(obj, st.grav_raw)
        st.dense_materialized = true
    end

    K, G = (copy(@view(obj.H[:, 1])), copy(CS.select_G_from_H(obj, obj.H)))

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = obj(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    ζstar = inner_x[1]; λstar = inner_x[2:end]
    nkkt = min(length(λstar), size(G, 2))
    max_abs_moment_kkt_resid = kkt_residual_blas(G, m_weights, nkkt, W)

    gravity_raw = obj.outer_constr_index <= obj.d ? cbuf[2] : NaN
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D^2], ctx.D, ctx.D)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (ctx.D, ctx.D))'
    Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
    gravity_val = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)
    logA = -log.(AodPow)
    R_sum = sum(ctx.q_tilde .* logA)
    R_mean = R_sum / ctx.D^2
    R_beta = R_sum / sum(ctx.q_tilde .^ 2)

    moment_resid = moment_resid_blas(G, obj.d, W)
    max_abs_moment_resid = isempty(moment_resid) ? NaN : maximum(abs.(moment_resid))

    winner_hash = hash(cf.winner)

    t_total = time() - t_total0
    elapsed = (total = t_total, inner = t_inner, post = t_total - t_inner)

    result = (x_free = collect(x_free), θ_full = θ_full,
              gamma_focal_prime = θ_full[3+ctx.D], logA = logA,
              K_hard = K_hard, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              Delta_minus_delta = Delta_dual - obj.δ,
              gravity_raw = gravity_raw, gravity_value = gravity_val,
              gravity_R_sum = R_sum, gravity_R_mean = R_mean, gravity_R_beta = R_beta,
              moment_resid = moment_resid, max_abs_moment_resid = max_abs_moment_resid,
              zeta = ζstar, lambda = collect(λstar),
              m_mean = sum(m_weights)/W, m_min = minimum(m_weights), m_max = maximum(m_weights),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              winner_hash = winner_hash, inner_status = nStatus, inner_iters = inner_iters,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              cache_hit = false, warm_started = warm, tag = tag,
              elapsed = elapsed, error_reason = nothing)

    prof_meta = (n_inner_solves = 1, n_inner_infeasible = 0, n_inner_iters = inner_iters,
                 n_fg_calls = n_fg, n_hess_calls = n_hess)
    return result, prof_meta
end

"""
    dense_recheck_certificate(cert::MomentRangeCertificate, θ_full, ctx) -> (min_val, max_val, agrees::Bool)

Independent verification via the production DENSE `ctx.obj.moments!` path
(the same trusted function the dense `evaluate_fullA_fast` mode calls),
entirely independent of `compressed_moments.jl`'s winner-form derivation.
DIAGNOSTIC / validation use only -- O(W*ncol) cost, never on the production
hot path.
"""
function dense_recheck_certificate(cert::MomentRangeCertificate, θ_full::AbstractVector, ctx)
    obj = ctx.obj
    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, obj.d)
    obj.moments!(K, G, θ_full, ctx.U, obj)
    col = @view G[:, cert.column]
    min_val, max_val = extrema(col)
    agrees = isapprox(min_val, cert.min_val; rtol = 1e-9, atol = 1e-9) && isapprox(max_val, cert.max_val; rtol = 1e-9, atol = 1e-9)
    return min_val, max_val, agrees
end

# ============================================================================
# Top-level integration wrapper -- structured exact-infeasibility status,
# additive fall-through to the EXISTING evaluate_fullA_screened /
# evaluate_fullA_screened_compressed (infeasibility_screen.jl, unmodified)
# once every screen passes.
# ============================================================================

"""
    RangedScreenContext

Caller-built-once, reused-forever bundle of the extra structures this file
needs on top of what `d20_real_setup`/`infeasibility_screen.jl` already build
into `ctx` (`ctx.pairwise`, `ctx.witness`). `envelope === nothing` means this
ctx's config was found unsupported by `precompute_envelope` (see
`EnvelopeUnsupportedContext`) -- the envelope/fused-winning-range screens are
then SKIPPED entirely (never applied stale), and only the existing zero-
winner screen + the general `range_screen_standalone` safety net run.
"""
struct RangedScreenContext
    envelope::Union{Nothing,EnvelopePrecomp}
    unsupported_reason::Union{Nothing,String}
end

"""
    build_ranged_screen_context(ctx) -> RangedScreenContext

Call ONCE per ctx (e.g. right after `d20_real_setup`), reuse for every
subsequent outer-point call. Catches `EnvelopeUnsupportedContext` here, at
context-build time, exactly once -- not per outer-point call.
"""
function build_ranged_screen_context(ctx)
    try
        ep = precompute_envelope(ctx)
        return RangedScreenContext(ep, nothing)
    catch e
        e isa EnvelopeUnsupportedContext || rethrow()
        return RangedScreenContext(nothing, e.reason)
    end
end

"""
    infeasible_result_ranged(x_free, theta_full, ctx, screen_status, failing_o, failing_d, stage, t_screen, tag, warm; extra...)

Same field-compatible result shape as `infeasibility_screen.jl::infeasible_result`
(so generic downstream code keeps working) with two new sentinel `inner_status`
codes for the two new certificate kinds this file adds.
"""
function infeasible_result_ranged(x_free, θ_full, ctx, screen_status::Symbol, failing_o::Int, failing_d::Int,
        stage::Int, t_screen::Float64, tag::String, warm::Bool; Hmax_failing::Float64 = NaN, target_failing::Float64 = NaN)
    D = ctx.D
    sentinel = screen_status === :EXACT_INFEASIBLE_PREWINNER_ENVELOPE ? -9004 :
               screen_status === :EXACT_INFEASIBLE_WINNING_RANGE      ? -9005 :
               screen_status === :EXACT_INFEASIBLE_MOMENT_RANGE       ? -9006 : -9000
    elapsed = (total = t_screen, inner = 0.0, post = 0.0)
    extra_msg = isnan(Hmax_failing) ? "" : ", best-possible winning value=$Hmax_failing vs target=$target_failing"
    return (x_free = collect(x_free), θ_full = θ_full,
            gamma_focal_prime = θ_full[3+D], logA = fill(NaN, D, D),
            K_hard = NaN, Delta_dual = Inf, Delta_primal = Inf, Delta_minus_delta = Inf,
            gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
            gravity_R_beta = NaN, moment_resid = Float64[], max_abs_moment_resid = NaN,
            zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
            weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
            winner_hash = UInt64(0), inner_status = sentinel, inner_iters = missing,
            primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
            elapsed = elapsed,
            error_reason = "exact_infeasible ($(screen_status)): origin $failing_o, destination $failing_d" * extra_msg *
                            ", stage=$stage/$D destinations scanned before rejection",
            screen_status = screen_status, screen_failing_o = failing_o, screen_failing_d = failing_d,
            screen_stage = stage)
end

"""
    evaluate_fullA_screened_ranged(x_free, ctx, rsc::RangedScreenContext;
        moment_representation=:compressed, cache=nothing, use_cache=true,
        use_witness=false, use_general_range_safety_net=true, mode=:hard,
        warm=true, tag="") -> (result, screen_meta)

Production integration point. Screening order (cheapest, most-certain-to-fire
first):
  1. `infeasibility_screen.jl`'s existing pairwise never-wins certificate
     (unchanged, O(D^2), draw-free).
  2. NEW: `envelope_prewinner_screen` (O(D^2), draw-free) -- SKIPPED if
     `rsc.envelope === nothing` (unsupported context, see above).
  3. optional `use_witness` extreme-draw witness (unchanged, existing).
  4. NEW: `screen_hard_winners_ranged` (single fused O(W*D)-ish pass:
     zero-winner check == existing `screen_hard_winners`'s check, PLUS the
     new winning-range check) if `rsc.envelope !== nothing`; otherwise falls
     back to the existing `screen_hard_winners` (zero-winner check only).
  5. on screen-pass: delegates to the EXISTING `evaluate_fullA_fast`
     (`:dense`) or `evaluate_fullA_screened_compressed` (`:compressed`,
     reusing the fused winner/wval, same as `evaluate_fullA_screened`) for
     the real evaluation.
  6. optional (`use_general_range_safety_net`, default true): AFTER a
     `:compressed` evaluation builds `cf` (or after any successful compressed
     evaluation), runs `range_screen_standalone(cf)` -- reuses the ALREADY-
     BUILT `cf`, no rebuild, no separate winner pass. This is a pure
     diagnostic/insurance check on an already-feasible-per-CC-solve point;
     see the production integration doc for why a positive hit here would be
     a genuine anomaly worth investigating, not treated as authoritative
     over the real inner solve's own result.
"""
function evaluate_fullA_screened_ranged(x_free::AbstractVector{Float64}, ctx, rsc::RangedScreenContext;
        moment_representation::Symbol = :compressed,
        cache = nothing, use_cache::Bool = true,
        use_witness::Bool = false, use_general_range_safety_net::Bool = true,
        mode::Symbol = :hard, warm::Bool = true, tag::String = "",
        pairwise::Union{Nothing,PairwiseCertificate} = nothing,
        witness::Union{Nothing,ExtremeDrawWitness} = nothing,
        safety_mult::Float64 = 50.0)

    mode == :hard || error("evaluate_fullA_screened_ranged: mode=:$mode not implemented (matches infeasibility_screen.jl)")
    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode)

    if cache !== nothing && use_cache
        hit = _cache_lookup(cache, key)
        if hit !== nothing
            return merge(hit, (cache_hit = true, tag = tag)),
                   (screen_status = get(hit, :screen_status, :cache_hit), elapsed = 0.0)
        end
    end

    t0 = time()
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    Pmat = target_shares(ctx)

    # ---- 1. existing pairwise never-wins certificate (unchanged) ----
    a = compute_a_od(θ_full, ctx)
    pc = pairwise === nothing ? (ctx.pairwise === nothing ? precompute_pairwise_M(ctx) : ctx.pairwise) : pairwise
    pres = pairwise_certificate(a, pc, Pmat)
    if pres.infeasible
        t_screen = time() - t0
        result = infeasible_result(x_free, θ_full, ctx, :pairwise_certified_infeasible,
                                    pres.worst_o, pres.worst_d, 0, t_screen, tag, warm)
        cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
        return result, (screen_status = :pairwise_certified_infeasible, worst_o = pres.worst_o,
                         worst_d = pres.worst_d, elapsed = t_screen)
    end

    # ---- 2. NEW: pre-winner envelope certificate ----
    if rsc.envelope !== nothing
        eres = envelope_prewinner_screen(θ_full, ctx, rsc.envelope; safety_mult = safety_mult)
        if eres.status === :EXACT_INFEASIBLE_PREWINNER_ENVELOPE
            t_screen = time() - t0
            result = infeasible_result_ranged(x_free, θ_full, ctx, :EXACT_INFEASIBLE_PREWINNER_ENVELOPE,
                                               eres.origin, eres.destination, 0, t_screen, tag, warm;
                                               Hmax_failing = eres.h_upper_bound, target_failing = eres.target)
            cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
            return result, (screen_status = :EXACT_INFEASIBLE_PREWINNER_ENVELOPE, worst_o = eres.origin,
                             worst_d = eres.destination, upper_bound = eres.h_upper_bound, target = eres.target,
                             margin_normalized = eres.margin_normalized, elapsed = t_screen)
        end
    end

    # ---- 3. optional witness (unchanged) ----
    if use_witness
        B = hard_score_B(ctx)
        wt = witness === nothing ? (ctx.witness === nothing ? build_extreme_draw_witness(ctx) : ctx.witness) : witness
        for d in 1:ctx.D, o in 1:ctx.D
            Pmat[o, d] > 0 || continue
            exists, s, ntested, csize = query_witness(o, d, a, B, wt)
            if !exists
                t_screen = time() - t0
                result = infeasible_result(x_free, θ_full, ctx, :witness_certified_infeasible, o, d, 0, t_screen, tag, warm)
                cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
                return result, (screen_status = :witness_certified_infeasible, worst_o = o, worst_d = d, elapsed = t_screen)
            end
        end
    end

    # ---- 4. fused hard-winner construction (zero-winner + NEW winning-range) ----
    order = order_destinations(pres, ctx.D)
    if rsc.envelope !== nothing
        wres = screen_hard_winners_ranged(θ_full, ctx, Pmat, rsc.envelope; order = order, safety_mult = safety_mult)
        if !wres.feasible
            t_screen = time() - t0
            status = wres.reject_kind === :zero_winner ? :winner_scan_infeasible : :EXACT_INFEASIBLE_WINNING_RANGE
            result = infeasible_result_ranged(x_free, θ_full, ctx, status, wres.failing_o, wres.failing_d,
                                               wres.stage, t_screen, tag, warm;
                                               Hmax_failing = wres.Hmax_failing, target_failing = wres.target_failing)
            cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
            return result, (screen_status = status, worst_o = wres.failing_o, worst_d = wres.failing_d,
                             stage = wres.stage, elapsed = t_screen)
        end
        winner_reuse, wval_reuse = wres.winner, wres.wval
    else
        wres0 = screen_hard_winners(θ_full, ctx, Pmat; order = order)
        if !wres0.feasible
            t_screen = time() - t0
            result = infeasible_result(x_free, θ_full, ctx, :winner_scan_infeasible, wres0.failing_o,
                                        wres0.failing_d, wres0.stage, t_screen, tag, warm)
            cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
            return result, (screen_status = :winner_scan_infeasible, worst_o = wres0.failing_o,
                             worst_d = wres0.failing_d, stage = wres0.stage, elapsed = t_screen)
        end
        winner_reuse, wval_reuse = wres0.winner, wres0.wval
    end
    t_screen_passed = time() - t0

    # ---- 5. real evaluation (delegates to the EXISTING, unmodified paths) ----
    if moment_representation === :dense
        result, prof_meta = evaluate_fullA_fast(x_free, ctx; cache = nothing, use_cache = false,
                                                  mode = mode, warm = warm, tag = tag)
        result = merge(result, (screen_status = :screen_passed,))
        cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
        return result, (screen_status = :screen_passed, screen_elapsed = t_screen_passed, prof_meta...)
    elseif moment_representation === :compressed
        # Build cf EXACTLY ONCE (from the winner/wval the fused screen already computed --
        # this is the cheap O(D^2)+O(W) packaging step, NOT a winner rescan) and reuse it for
        # both the general range-screen safety net (step 6) and the real inner solve (step 5),
        # rather than building it twice.
        wres_dummy = WinnerScreenResult(true, ctx.D, 0, 0, collect(order), winner_reuse, wval_reuse, zeros(Int, ctx.D, ctx.D))
        cf = compressed_factual_from_screen(θ_full, ctx, wres_dummy)

        safety_hit = nothing
        if use_general_range_safety_net
            rres = range_screen_standalone(cf; safety_mult = safety_mult)
            if rres.status === :EXACT_INFEASIBLE_MOMENT_RANGE
                safety_hit = rres.certificate
                t_screen = time() - t0
                result = infeasible_result_ranged(x_free, θ_full, ctx, :EXACT_INFEASIBLE_MOMENT_RANGE,
                                                   safety_hit.origin, safety_hit.destination, ctx.D, t_screen, tag, warm)
                result = merge(result, (safety_net_certificate = safety_hit,))
                cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
                return result, (screen_status = :EXACT_INFEASIBLE_MOMENT_RANGE, certificate = safety_hit, elapsed = t_screen)
            end
        end

        result, prof_meta = evaluate_fullA_screened_compressed_with_cf(x_free, θ_full, ctx, cf; warm = warm, tag = tag)
        result = merge(result, (screen_status = :screen_passed, safety_net_checked = use_general_range_safety_net))
        cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
        return result, (screen_status = :screen_passed, screen_elapsed = t_screen_passed, prof_meta...)
    else
        error("evaluate_fullA_screened_ranged: moment_representation=:$moment_representation not implemented (only :dense, :compressed)")
    end
end
