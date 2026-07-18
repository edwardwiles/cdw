# ============================================================================
# Phase 2 (continuation 3): block-local AND genuinely incremental exact
# evaluators for L_fix, per the user's explicit design (base cache of
# q_s^* = -zeta*-lambda*'G_s(theta0) decomposed into PER-DESTINATION
# contributions; a perturbation updates ONLY the 1-2 affected destination
# blocks' contribution and reassembles q_s from cached pieces -- never
# rebuilds the full G matrix).
#
# DEPENDENCY GRAPH, verified directly from moments/hFunction.jl and
# full_aod_diag/moments_gammanorm.jl (not assumed):
#   - Perturbing Aod_theta[o,d] (one entry of the D x D free matrix) changes
#     ONLY the D moment columns of hFunction!'s destination-d block
#     (d1 = d + (o'-1)*D for o' in 1:D, ALL D origins at that destination --
#     because MinInd! picks a winner across all D origins for fixed d, so
#     the winner CAN switch even though only one origin's underlying price
#     changed). If (o,d)==(baseIndex,baseIndex), it ALSO changes the single
#     counterfactual column (index D^2+1) via hFunctionCounter!'s
#     constConsσ[baseIndex,baseIndex].
#   - In the pivot-reduced coordinates (gravity_elimination.jl), ONE z_free
#     coordinate maps to exactly ONE direct Aod entry PLUS the pivot Aod
#     entry (pivot_expand's affine combination) -- i.e. up to 2 affected
#     destinations, up to 2 changed (o,d) cells total.
#   - gamma'_focal (w[1]) changes ONLY the counterfactual column; zero A_od
#     entries change; hFunctionCounter!'s counterType==1 branch never calls
#     MinInd!, so this coordinate is smooth (no winner-switching machinery
#     needed at all).
#
# CLOSED-FORM SIMPLIFICATION (derived, then verified empirically against the
# trusted fixed_dual_L below): within ANY destination-d block, a "loser"
# column's raw value is `-P[d1]*denom[d]`, a FIXED DATA CONSTANT independent
# of which origin lost or who won; only the WINNING origin's column carries
# theta-dependence (`pricesTempσ[winner] - P[d1(winner)]*denom[d]`). This
# means the lambda*-weighted contribution of an ENTIRE destination-d block to
# q_s collapses to:
#
#   contrib[s,d] = (SW[s]/gammafac) * ( CONST_d[d] + lambda*[d1(winner(s,d))] * pTsigma(winner(s,d), s, d) )
#
# where CONST_d[d] = -denom[d] * sum_o lambda*[d+(o-1)D]*P[d+(o-1)D] is a
# PURE DATA+lambda* constant (zero draws-loop cost, computed ONCE). A
# perturbation therefore only needs the NEW winner(s,d) and their pTsigma
# value, per draw -- exactly the incremental target.
# ============================================================================
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
using SpecialFunctions: gamma as spgamma
using LinearAlgebra: dot

lin_to_od(lin::Int, D::Int) = (mod1(lin, D), div(lin - 1, D) + 1)   # (o, d) from column-major linear index

"""
    min_secondmin_with_idx(col) -> (m1, idx1, m2, idx2)

Same two-pass scan as `winners_v2.jl::min_and_secondmin`, additionally
returning the runner-up's INDEX (needed for the O(1) incremental winner
update below, which must know WHO the runner-up is, not just its value).
"""
function min_secondmin_with_idx(col)
    n = length(col)
    m1 = col[1]; idx1 = 1
    m2 = oftype(m1, Inf); idx2 = 0
    @inbounds for i in 2:n
        v = col[i]
        if isless(v, m1)
            m2 = m1; idx2 = idx1
            m1 = v; idx1 = i
        elseif isless(v, m2)
            m2 = v; idx2 = i
        end
    end
    return m1, idx1, m2, idx2
end

"""
    min_secondthirdmin_with_idx(col) -> (m1, idx1, m2, idx2, m3, idx3)

Continuation 8 addition (additive -- extends, does not replace,
`min_secondmin_with_idx` above): same first-occurrence `isless` scan, extended
to rank-3. `idx3`/`m3` are `0`/`Inf` if `D<3`. Matches
`winner_certificate.jl::top3_scan`'s semantics exactly (same tie-break
convention), duplicated here rather than `include`d from that file to keep
`lfix_incremental.jl` self-contained (this file predates `winner_certificate.jl`
and other files depend on its current include order) -- equivalence with
`top3_scan` is verified in `test_winner_top3_equivalence.jl`.
"""
function min_secondthirdmin_with_idx(col)
    n = length(col)
    m1 = col[1]; idx1 = 1
    m2 = oftype(m1, Inf); idx2 = 0
    m3 = oftype(m1, Inf); idx3 = 0
    @inbounds for i in 2:n
        v = col[i]
        if isless(v, m1)
            m3 = m2; idx3 = idx2
            m2 = m1; idx2 = idx1
            m1 = v; idx1 = i
        elseif isless(v, m2)
            m3 = m2; idx3 = idx2
            m2 = v; idx2 = i
        elseif isless(v, m3)
            m3 = v; idx3 = i
        end
    end
    return m1, idx1, m2, idx2, m3, idx3
end

"""
    update_winner_o1(price_wo, wo, price_ro, ro, o_changed, new_price) -> (new_wo, new_price_wo, new_ro, new_price_ro)

TRUE O(1) incremental winner update for ONE draw: given the CACHED winner
(`wo`, value `price_wo`) and runner-up (`ro`, value `price_ro`) among D
competitors, and that origin `o_changed`'s price changed to `new_price`
(all OTHER origins unchanged), returns the new winner/runner-up without
touching any of the other D-2 unchanged competitors. Proof this is exact
(not an approximation): every unchanged competitor already satisfies
`price >= price_wo` (by definition of `wo` being the prior global min) and,
except for `ro` itself, `price >= price_ro` too. Case analysis:
  - `o_changed == wo` (the winner's own price moved):
      - `new_price <= price_ro`: winner unchanged (still <= everyone, since
        every unchanged competitor is >= price_ro >= new_price by hypothesis
        the case `new_price<=price_ro`... wait -- unchanged competitors are
        >= price_wo_OLD, not necessarily >= new_price if new_price rose above
        price_wo_OLD; but they ARE >= price_ro when >= second place, and
        price_ro>=price_wo_OLD always) -- if `new_price <= price_ro` the
        winner stays `wo` (its new price is still <= the former runner-up,
        which was <= every other unchanged competitor).
      - `new_price > price_ro`: `ro` becomes the new global min (it is <=
        every unchanged competitor by definition, and now < o_changed too);
        the new runner-up is `min(new_price, ro)` among {o_changed's new
        price} vs the BEST of the remaining D-2 -- which we do NOT have
        cached (this is the one case where an O(1) update cannot recover
        the EXACT new runner-up without a fallback). Since this function is
        used ONLY to determine the WINNER for the L_fix contribution
        formula (the runner-up value is never used downstream), this case
        returns `new_ro = o_changed`, `new_price_ro = new_price` as a
        DELIBERATELY INEXACT placeholder for the runner-up alone, clearly
        flagged via the `ro_exact` return -- callers that only need the
        winner (the L_fix use case) are unaffected; a caller needing the
        exact runner-up in this branch must fall back to a full rescan.
  - `o_changed != wo`:
      - `new_price >= price_wo`: winner unchanged. Runner-up: if
        `o_changed == ro`, the exact new runner-up is unrecoverable without
        the true third-place (same caveat as above, same placeholder);
        otherwise (o_changed was neither winner nor runner-up) the runner-up
        is unchanged UNLESS `new_price < price_ro`, in which case `o_changed`
        becomes the new (exact) runner-up.
      - `new_price < price_wo`: `o_changed` becomes the new winner (proof:
        every unchanged competitor is >= price_wo > new_price, and the old
        `wo` is now demoted). New runner-up = old `wo` EXACTLY (old winner's
        price is unchanged and was <= every unchanged competitor).
"""
function update_winner_o1(price_wo::Float64, wo::Int, price_ro::Float64, ro::Int, o_changed::Int, new_price::Float64)
    if o_changed == wo
        if new_price <= price_ro
            return wo, new_price, ro, price_ro, true
        else
            return ro, price_ro, o_changed, new_price, false   # runner-up inexact (flagged)
        end
    else
        if new_price < price_wo
            return o_changed, new_price, wo, price_wo, true
        elseif o_changed == ro
            if new_price < price_wo   # unreachable given outer branch, kept for clarity/symmetry
                return o_changed, new_price, wo, price_wo, true
            end
            return wo, price_wo, o_changed, new_price, false   # runner-up inexact (flagged)
        else
            new_ro = new_price < price_ro ? o_changed : ro
            new_price_ro = new_price < price_ro ? new_price : price_ro
            return wo, price_wo, new_ro, new_price_ro, true
        end
    end
end

"""
aod_level_cell(theta_full, ctx, o, d) -- the LEVEL Aod[o,d] (gravity_tariff.jl's own level-conversion
formula), single cell, O(1). NOTE: `lambda = reshape(P,(D,D))'`, so `lambda[o,d] = P[d+(o-1)*D]`
(the SAME d1=d+(o-1)*D linear-index convention used throughout hFunction.jl/winners.jl) -- verified
against `winners.jl::factual_prices`'s own (already-validated) full-matrix formula, not re-derived
from scratch a second time.
"""
function aod_level_cell(θ_full::AbstractVector, ctx, o::Int, d::Int)
    γo = ctx.γ
    μ = θ_full[1]
    D = ctx.D
    lambda_od = γo.P[d + (o - 1) * D]; lambda_1d = γo.P[d]   # lambda[1,d] = P[d+(1-1)*D] = P[d]
    Aod_θ_od = θ_full[ctx.Aod_offset + o + (d - 1) * D]
    return Aod_θ_od * γo.cHat[o, d] * ((γo.wHat[o] * γo.τ[o, d]) / (γo.wHat[1, 1] * γo.τ[1, d]))^(1 / μ) * (lambda_od / lambda_1d)
end

"""
aod_pow_cell(theta_full, ctx, o, d, mu) -- AodPow[o,d] = (Aod_level[o,d]/cHat[o,d])^(-mu). This, NOT
the level Aod, is what hFunction!/hFunctionCounter! actually receive as their (locally-named) `Aod`
argument -- both call sites pass `AodPow`, confirmed from moments_gammanorm.jl's call:
`hFunction!(..., AodPow, ...)` / `hFunctionCounter!(..., AodPow, ...)`. Conflating the level with
AodPow here was the root cause of an earlier self-validation failure in this file.
"""
function aod_pow_cell(θ_full::AbstractVector, ctx, o::Int, d::Int)
    μ = θ_full[1]
    lvl = aod_level_cell(θ_full, ctx, o, d)
    return (lvl / ctx.γ.cHat[o, d])^(-μ)
end

"""
    price_and_pTsigma_cell(θ_full, ctx, o, d) -> (price::Vector{W}, pTσ::Vector{W})

O(W) single-(o,d)-cell recompute of hFunction!'s `pricesTemp[o]`/`pricesTempσ[o]`
formulas for destination d (UoModel==1: o1=o).
"""
function price_and_pTsigma_cell(θ_full::AbstractVector, ctx, o::Int, d::Int)
    γo = ctx.γ; σ = θ_full[2]; μ = θ_full[1]
    AodPow = aod_pow_cell(θ_full, ctx, o, d)
    constCons_od = γo.wHat[o] * AodPow * γo.τ[o, d]
    wPow_o = γo.wHat[o]^(1 - σ)
    constConsσ_od = wPow_o * (AodPow * γo.τ[o, d])^(1 - σ)
    U = ctx.U
    # NOTE: hFunction! divides by UPow/UσPow = U.^(-mu)/Uσ.^(-mu), NOT raw U/Uσ -- these power
    # transforms are applied by the CALLER (EK_moments_gammanorm_directgp!) before hFunction! ever
    # sees them; matched here exactly, not omitted.
    price = constCons_od ./ (@view(U[:, o]) .^ (-μ))
    pTσ = constConsσ_od ./ (@view(γo.Uσ[:, o]) .^ (-μ))
    return price, pTσ
end

struct LFixBaseCache
    D::Int; oci::Int; W::Int; μ::Float64; σ::Float64; baseIndex::Int
    gammafac::Float64
    SW::Vector{Float64}
    denom::Vector{Float64}          # length D
    CONST_d::Vector{Float64}        # length D
    price0::Array{Float64,3}        # W x D x D  (levels, winner-finding)
    pTσ0::Array{Float64,3}          # W x D x D  (sigma-transformed)
    winner0::Matrix{Int}            # W x D
    winner_price0::Matrix{Float64}  # W x D, price LEVEL of the cached winner
    runnerup0::Matrix{Int}          # W x D, origin index of the cached runner-up
    runnerup_price0::Matrix{Float64}  # W x D, price LEVEL of the cached runner-up
    # ---- Continuation 8 addition (additive; every field above is UNCHANGED in
    # meaning/values -- old code reading only the fields above continues to work
    # exactly as before). Third-place cache, needed by `coord_winner_update!`-style
    # exact O(1) top-3 updates for the same-destination-two-changed-origins case
    # (see count_winner_flips_multi_top3 / dest_contrib_incremental_top3). Computed
    # essentially for FREE from the ALREADY-BUILT dense price0/pTσ0 below (an extra
    # O(W*D^2) PASS over data already in memory, not an extra O(W*D^2) RECOMPUTE) ----
    third0::Matrix{Int}             # W x D, origin index of the cached third-place (0 if D<3)
    third_price0::Matrix{Float64}   # W x D, price LEVEL of the cached third-place (Inf if D<3)
    third_pTσ0::Matrix{Float64}     # W x D, sigma-transformed value of the cached third-place
    contrib0::Matrix{Float64}       # W x D, cached per-destination contribution to q0
    λstar::Vector{Float64}
    ζstar::Float64
    q0::Vector{Float64}
    wPrime_bi::Float64; τPrime_bi::Float64; LPrime_bi::Float64
    Uσ_bi::Vector{Float64}
    λ_cf::Float64
    cf_contrib0::Vector{Float64}
end

"""
    TiedWinnerError

Continuation 6 finding: `build_lfix_base_cache`'s self-validation was previously mislabeled as
"closed-form derivation has a bug" for a specific, understood, and DIFFERENT condition -- an exact
(bit-for-bit) price TIE between two or more origins at some (draw, destination) pair. `MinInd!`
(`misc/smoothMinIndNew!.jl`), which `hFunction!` actually calls, sets `xInd[i]=1` for EVERY origin
satisfying `x[i] <= xMin` (not `>`) -- i.e. on an exact tie, ALL tied origins get winner-share
credit simultaneously (an economically sensible market-split convention). This cache's winner-finding
(`min_secondmin_with_idx`'s `isless` scan, used throughout `lfix_incremental.jl`'s O(1)/O(D) tiers)
assumes a UNIQUE arg-min winner per (draw,destination), by design -- the entire incremental-update
case analysis in `update_winner_o1` is built on that assumption and is not economical to extend to
N-way ties (correctness-critical code, not extended lightly). Root-caused this continuation via a
direct per-(draw,destination) contrib0-vs-true-G scan: exactly 1/32000 (draw,destination) pairs had a
bit-exact tie at the specific (decoupled, non-jointly-optimized) point that first exposed this;
`price_and_pTsigma_cell`/`aod_pow_cell`/`CONST_d`'s own formulas were independently verified EXACT
(0.0 diff) at every checked (draw,destination) pair once the tie was accounted for -- this is not a
formula bug, it is a genuine unhandled edge case, now DETECTED EXPLICITLY (see `detect_price_ties`)
rather than surfacing as a confusing self-validation failure.

Ties are a probability-zero event for GENERIC continuous draws -- this has never been observed at any
of this investigation's validated candidate points (upper/lower incumbents, poll neighbors, D=6 pilot
start) and is not expected in a live KNITRO trajectory (a converged outer point's own A_od values are
not chosen to create exact coincidences). It DOES occur when code (e.g. a gamma-profile driver)
evaluates the composite gradient at ARBITRARY/decoupled (gamma',A) pairs, including degenerate ones
inherited from an earlier failed optimization step. Callers should catch `TiedWinnerError` specifically
(distinct from a genuine correctness bug) and fall back to a full-rebuild gradient (`fixed_dual_L`-based
central FD, which correctly reflects `MinInd!`'s true tie-splitting behavior since it always rebuilds
the complete moment matrix) for that one outer point, per this file's own established "correctness over
speed" precedent for rare edge cases (see the 2-changed-origins fallback in `dest_contrib_incremental`).
"""
struct TiedWinnerError <: Exception
    n_tied_pairs::Int
    examples::Vector{Tuple{Int,Int}}   # (ω, d) pairs, first few only
end
Base.showerror(io::IO, e::TiedWinnerError) = print(io,
    "TiedWinnerError: $(e.n_tied_pairs) (draw,destination) pair(s) have an EXACT price tie between " *
    "2+ origins -- build_lfix_base_cache's unique-winner assumption does not hold here (NOT a " *
    "derivation bug, see TiedWinnerError's docstring). First examples (ω,d): $(e.examples)")

"""
    detect_price_ties(price0::Array{Float64,3}, D::Int, W::Int; tol=0.0) -> Vector{Tuple{Int,Int}}

Scans the cached `price0` (W x D x D) for (draw,destination) pairs where 2+ origins are tied at the
row minimum (within `tol`, default EXACT bit-equality). Returns the list of tied `(ω,d)` pairs
(empty if none) -- called BEFORE the self-validation so a tie is diagnosed precisely, not confused
with a generic derivation bug.
"""
function detect_price_ties(price0::Array{Float64,3}, D::Int, W::Int; tol::Float64 = 0.0)
    tied = Tuple{Int,Int}[]
    @inbounds for d in 1:D, ω in 1:W
        mn = price0[ω, 1, d]
        for o in 2:D
            price0[ω, o, d] < mn && (mn = price0[ω, o, d])
        end
        n_at_min = 0
        for o in 1:D
            price0[ω, o, d] <= mn + tol && (n_at_min += 1)
        end
        n_at_min > 1 && push!(tied, (ω, d))
    end
    return tied
end

"""
    build_lfix_base_cache(x_free0, ctx, base::BaseDualState) -> LFixBaseCache

Builds the per-destination-contribution cache at the base point. Self-
validates internally: reconstructs q0 from the cache pieces and asserts it
matches `-base.ζstar - lambda*'G0[s,1:oci-1]` computed directly (not merely
assumed to match by construction) -- errors loudly if the closed-form
derivation above has a sign/indexing bug, rather than silently producing a
wrong cache. Throws `TiedWinnerError` (see above), NOT the generic error,
when the failure is a detected exact price tie rather than a derivation bug.
"""
function build_lfix_base_cache(x_free0::AbstractVector, ctx, base::BaseDualState)
    obj = ctx.obj
    D = ctx.D; W = size(obj.U, 1); oci = obj.outer_constr_index
    μ = base.θ_full0[1]; σ = ctx.σ; bi = ctx.bi
    γo = ctx.γ
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = γo.SamplingWeights[1:W]
    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]   # gamma[d]==1 factual side, always
    λstar = base.λstar

    price0 = Array{Float64}(undef, W, D, D)
    pTσ0 = Array{Float64}(undef, W, D, D)
    for d in 1:D, o in 1:D
        p, ps = price_and_pTsigma_cell(base.θ_full0, ctx, o, d)
        price0[:, o, d] .= p; pTσ0[:, o, d] .= ps
    end

    # ---- tie check BEFORE anything else (Continuation 6 fix, see TiedWinnerError's docstring): an
    # exact price tie between 2+ origins means MinInd!'s real winner-splitting behavior cannot be
    # represented by this cache's unique-arg-min machinery. Detected explicitly and distinctly from a
    # genuine derivation bug, rather than surfacing only as a confusing self-validation failure. ----
    tied_pairs = detect_price_ties(price0, D, W)
    isempty(tied_pairs) || throw(TiedWinnerError(length(tied_pairs), tied_pairs[1:min(5, end)]))

    winner0 = Matrix{Int}(undef, W, D)
    winner_price0 = Matrix{Float64}(undef, W, D)
    runnerup0 = Matrix{Int}(undef, W, D)
    runnerup_price0 = Matrix{Float64}(undef, W, D)
    # Continuation 8: extended to rank-3 (additive -- winner0/winner_price0/runnerup0/
    # runnerup_price0 are computed IDENTICALLY to before, min_secondthirdmin_with_idx's
    # first two return values match min_secondmin_with_idx's exactly by construction,
    # verified in test_winner_top3_equivalence.jl). third0/third_price0 are NEW.
    third0 = Matrix{Int}(undef, W, D)
    third_price0 = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, ω in 1:W
        m1, idx1, m2, idx2, m3, idx3 = min_secondthirdmin_with_idx(@view(price0[ω, :, d]))
        winner0[ω, d] = idx1; winner_price0[ω, d] = m1
        runnerup0[ω, d] = idx2; runnerup_price0[ω, d] = m2
        third0[ω, d] = idx3; third_price0[ω, d] = m3
    end
    # third_pTσ0: sigma-transformed value of the cached third-place, needed by
    # dest_contrib_incremental_top3 (mirrors how pTσ0 is already stored densely for
    # every origin -- third_pTσ0 just indexes it at the third-place origin, O(W*D),
    # free given pTσ0 is already fully materialized above).
    third_pTσ0 = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, ω in 1:W
        t = third0[ω, d]
        third_pTσ0[ω, d] = t == 0 ? Inf : pTσ0[ω, t, d]
    end

    CONST_d = zeros(D)
    for d in 1:D
        s = 0.0
        for o in 1:D
            d1 = d + (o - 1) * D
            s += λstar[d1] * (-γo.P[d1] * denom[d])
        end
        CONST_d[d] = s
    end

    contrib0 = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, ω in 1:W
        wo = winner0[ω, d]
        d1w = d + (wo - 1) * D
        contrib0[ω, d] = (SW[ω] / gammafac) * (CONST_d[d] + λstar[d1w] * pTσ0[ω, wo, d])
    end

    # ---- counterfactual column pieces ----
    wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
    wPrime_bi = wPrime[bi]   # == 1.0 by construction, kept symbolic for clarity/robustness
    τPrime_bi = γo.τPrime[bi, bi]
    LPrime_bi = γo.LPrime[bi]
    # hFunctionCounter!'s own 4th positional arg (named Uσ in its body) is bound to UσPow = Uσ.^(-mu)
    # by its caller (EK_moments_gammanorm_directgp!) -- matched exactly, mu fixed so precomputed once.
    Uσ_bi = γo.Uσ[:, bi] .^ (-μ)     # UoModel==1: o1 = baseIndex
    d1_cf = D^2 + 1
    λ_cf = oci - 1 >= d1_cf ? λstar[d1_cf] : 0.0

    AodPow_bibi0 = aod_pow_cell(base.θ_full0, ctx, bi, bi)
    γ_prime_bi0 = base.θ_full0[3+D]
    constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi0 * τPrime_bi)^(1 - σ)
    denom_cf0 = γ_prime_bi0^σ * wPrime_bi_gdp(wPrime_bi, LPrime_bi)
    raw_cf0 = constConsσ_bibi ./ Uσ_bi .- denom_cf0
    cf_contrib0 = λ_cf .* (raw_cf0 ./ gammafac .* SW)

    # ---- self-validation: reconstruct q0 from cache, compare against direct computation ----
    K = zeros(W); Gfull = zeros(W, obj.d)
    obj.moments!(K, Gfull, base.θ_full0, obj.U, obj)
    q0_true = [-base.ζstar - dot(λstar, @view(Gfull[s, 1:oci-1])) for s in 1:W]
    q0_cache = [-base.ζstar - sum(@view(contrib0[s, :])) - cf_contrib0[s] for s in 1:W]
    maxerr = maximum(abs.(q0_true .- q0_cache))
    maxerr < 1e-8 || error("build_lfix_base_cache: self-validation FAILED, max|q0_true-q0_cache|=$maxerr -- closed-form derivation has a bug, not a numerical-tolerance issue")

    return LFixBaseCache(D, oci, W, μ, σ, bi, gammafac, SW, denom, CONST_d, price0, pTσ0,
        winner0, winner_price0, runnerup0, runnerup_price0,
        third0, third_price0, third_pTσ0, contrib0,
        λstar, base.ζstar, q0_true, wPrime_bi, τPrime_bi, LPrime_bi, Uσ_bi, λ_cf, cf_contrib0)
end

"gdp used inside hFunctionCounter! for baseIndex: wPrime[bi]*LPrime[bi] (wPrime[bi]==1 always, kept explicit)."
wPrime_bi_gdp(wPrime_bi, LPrime_bi) = wPrime_bi * LPrime_bi

"""
    cf_contrib_at(cache, θ_full) -> Vector{W}

Recomputes the counterfactual column's lambda*-weighted contribution at an
arbitrary theta (only Aod[bi,bi] and gamma'_focal matter) -- O(W), no draws
loop beyond a single broadcast.
"""
function cf_contrib_at(cache::LFixBaseCache, θ_full::AbstractVector, ctx)
    bi = cache.baseIndex; σ = cache.σ
    AodPow_bibi = aod_pow_cell(θ_full, ctx, bi, bi)
    γ_prime_bi = θ_full[3+ctx.D]
    constConsσ_bibi = cache.wPrime_bi^(1 - σ) * (AodPow_bibi * cache.τPrime_bi)^(1 - σ)
    denom_cf = γ_prime_bi^σ * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi)
    raw_cf = constConsσ_bibi ./ cache.Uσ_bi .- denom_cf
    return cache.λ_cf .* (raw_cf ./ cache.gammafac .* cache.SW)
end

"""
    dest_contrib_block_local(cache, ctx, θ_full, d) -> Vector{W}

Tier 1 (block-local): FULL recompute of destination d's price/pTsigma/winner
for ALL D origins (not just the changed one), then the same closed-form
contribution formula. Correct but does not exploit that D-1 origins are
usually unchanged -- Tier 2 below does.
"""
function dest_contrib_block_local(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int)
    D = cache.D; W = cache.W
    price_d = Matrix{Float64}(undef, W, D); pTσ_d = Matrix{Float64}(undef, W, D)
    for o in 1:D
        p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
        price_d[:, o] .= p; pTσ_d[:, o] .= ps
    end
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        _, wo, _ = min_and_secondmin(@view(price_d[ω, :]))
        d1w = d + (wo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_d[ω, wo])
    end
    return contrib
end

"""
    dest_contrib_incremental(cache, ctx, θ_full, d, changed_origins) -> Vector{W}

Tier 2 (genuinely incremental): recomputes price/pTsigma ONLY for the origins
in `changed_origins` (1 or 2 of them, per the dependency graph above);
combines with CACHED `cache.price0[:,o,d]`/`cache.pTσ0[:,o,d]` for every
other origin, then a fresh min/secondmin rescan (the scan itself is
inherently O(D) per draw -- cannot be avoided without a fancier order-
statistic structure -- but the PRICE FORMULA EVALUATION is O(|changed|) not
O(D)).
"""
function dest_contrib_incremental(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    new_price = Dict{Int,Vector{Float64}}(); new_pTσ = Dict{Int,Vector{Float64}}()
    for o in changed_origins
        p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
        new_price[o] = p; new_pTσ[o] = ps
    end
    contrib = Vector{Float64}(undef, W)
    col = Vector{Float64}(undef, D)
    @inbounds for ω in 1:W
        for o in 1:D
            col[o] = haskey(new_price, o) ? new_price[o][ω] : cache.price0[ω, o, d]
        end
        _, wo, _ = min_and_secondmin(col)
        pTσ_wo = haskey(new_pTσ, wo) ? new_pTσ[wo][ω] : cache.pTσ0[ω, wo, d]
        d1w = d + (wo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"""
    dest_contrib_incremental_top3(cache, ctx, θ_full, d, changed_origins) -> Vector{W}

Continuation 8: O(1)-per-draw replacement for `dest_contrib_incremental_o1`'s
2-changed-origin fallback (which previously ALWAYS called `dest_contrib_incremental`,
an O(D) rescan using the full `cache.price0[:,o,d]` background array for every draw).
Uses the SAME exact top-3-cache argument as
`winner_certificate.jl::coord_winner_update!` (reused, not re-derived): with <=2
changed origins in one destination, the best surviving UNCHANGED origin is at worst
rank 3, so the exact new winner is argmin over {changed origins' NEW (price,pTsigma)}
union {first of (winner0,runnerup0,third0) not in changed_origins}. Falls back to
`dest_contrib_incremental`'s O(D) rescan only for the never-happens-for-a-single-
reduced-coordinate |changed_origins|>2 case (safety net, matches this file's own
established fallback discipline). Equivalence with `dest_contrib_incremental` is
verified exactly (not just to tolerance) in `test_winner_top3_equivalence.jl`,
including synthetic forced-2-changed-origin cases beyond what a real D=4 run
naturally exercises.
"""
function dest_contrib_incremental_top3(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    length(changed_origins) > 2 && return dest_contrib_incremental(cache, ctx, θ_full, d, changed_origins)

    D = cache.D; W = cache.W
    Cd = changed_origins
    new_price = Dict{Int,Vector{Float64}}(); new_pTσ = Dict{Int,Vector{Float64}}()
    for o in Cd
        p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
        new_price[o] = p; new_pTσ[o] = ps
    end
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_p = Inf; best_pTσ = Inf
        if !(r1 in Cd)
            best_o = r1; best_p = cache.winner_price0[ω, d]; best_pTσ = cache.pTσ0[ω, r1, d]
        elseif !(r2 in Cd)
            best_o = r2; best_p = cache.runnerup_price0[ω, d]; best_pTσ = cache.pTσ0[ω, r2, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_p = cache.third_price0[ω, d]; best_pTσ = cache.third_pTσ0[ω, d]
        end
        bo = best_o; bp = best_p; bpTσ = best_pTσ
        for o in Cd
            v = new_price[o][ω]
            if v < bp
                bp = v; bo = o; bpTσ = new_pTσ[o][ω]
            end
        end
        if bo == 0
            # extremely defensive: all of top-3 were changed (needs D<=3 & |Cd|>=3);
            # unreachable for |Cd|<=2 with D>=3 -- same defensive fallback as
            # coord_winner_update!'s own docstring. Full O(D) column rescan.
            col = Vector{Float64}(undef, D)
            for o in 1:D
                col[o] = haskey(new_price, o) ? new_price[o][ω] : cache.price0[ω, o, d]
            end
            _, bo, _ = min_and_secondmin(col)
            bpTσ = haskey(new_pTσ, bo) ? new_pTσ[bo][ω] : cache.pTσ0[ω, bo, d]
        end
        d1w = d + (bo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * bpTσ)
    end
    return contrib
end

"""
    dest_contrib_incremental_o1(cache, ctx, θ_full, d, changed_origins; multi_method=:top3) -> Vector{W}

Tier 3 (TRUE O(1) winner update): for the common case of exactly ONE changed
origin in destination d, determines the new winner via `update_winner_o1`
using ONLY the cached (winner, winner_price, runnerup, runnerup_price) --
NO rescan of the other D-1 competitors at all (proven exact for the WINNER
in every branch, see `update_winner_o1`'s docstring). For the rarer case of
TWO changed origins in the SAME destination (both the direct coordinate AND
the gravity pivot land in the same destination), chaining two O(1) updates
can propagate an INEXACT cached runner-up into the second update's decision
(traced through explicitly in `update_winner_o1`'s docstring) -- rather than
risk a silent winner error in that narrow case, this now DEFAULTS
(`multi_method=:top3`) to the exact O(1) top-3-cache update
`dest_contrib_incremental_top3` (Continuation 8, closes the same fallback
`count_winner_flips_multi_top3` closes in composite_gradient.jl). The ORIGINAL
O(D) rescan (`dest_contrib_incremental`) is kept, reachable via
`multi_method=:generic`, as the documented reference/fallback -- correctness
first, and both are verified exactly equal in `test_winner_top3_equivalence.jl`.
"""
function dest_contrib_incremental_o1(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int}; multi_method::Symbol = :top3)
    if length(changed_origins) != 1
        multi_method === :top3 && return dest_contrib_incremental_top3(cache, ctx, θ_full, d, changed_origins)
        multi_method === :generic && return dest_contrib_incremental(cache, ctx, θ_full, d, changed_origins)
        error("dest_contrib_incremental_o1: multi_method must be :top3 or :generic, got $multi_method")
    end

    D = cache.D; W = cache.W
    o = changed_origins[1]
    new_price, new_pTσ = price_and_pTsigma_cell(θ_full, ctx, o, d)
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        wo, price_wo, ro, price_ro, _exact = update_winner_o1(
            cache.winner_price0[ω, d], cache.winner0[ω, d],
            cache.runnerup_price0[ω, d], cache.runnerup0[ω, d],
            o, new_price[ω])
        pTσ_wo = wo == o ? new_pTσ[ω] : cache.pTσ0[ω, wo, d]
        d1w = d + (wo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"lfix_from_q(q, ζstar) -> Float64 -- the final scalar, matching fixed_dual_L's own formula exactly."
function lfix_from_q(q::AbstractVector, ζstar::Float64)
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return -(sum(Psi_q) / length(q) + ζstar)
end

"""
    affected_cells(pe, coord_idx) -> Vector{Tuple{Int,Int}}

`coord_idx` in 1:D^2 (1-indexed into the reduced w vector: 1=gamma'_focal,
2:D^2 = z_free[1:D^2-1]). Returns the list of (o,d) Aod cells that change
when coord_idx is perturbed (empty for coord_idx==1 -- gamma only touches
the counterfactual column, no A_od cell).
"""
function affected_cells(pe, coord_idx::Int)
    coord_idx == 1 && return Tuple{Int,Int}[]
    k = coord_idx - 1   # index into z_free
    dir_lin = pe.other_idx[k]
    piv_lin = pe.pivot_lin
    return [lin_to_od(dir_lin, pe.D), lin_to_od(piv_lin, pe.D)]
end

"""
    lfix_incremental_at(cache, ctx, pe, w0, coord_idx, new_val; tier=:incremental) -> Float64

The main entry point: `w0` is the cache's base reduced coordinate vector,
`coord_idx`/`new_val` specify a SINGLE-coordinate perturbation (matching how
a central-FD gradient probes one coordinate at a time). `tier` selects
`:incremental` (Tier 2, default), `:block_local` (Tier 1), or `:incremental_o1`
(Tier 3, the TRUE O(1)-winner-update version) -- for the separate
profiling/equivalence comparison the task requires).
"""
function lfix_incremental_at(cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64; tier::Symbol = :incremental, multi_method::Symbol = :top3)
    D = cache.D
    w = copy(w0); w[coord_idx] = new_val
    z = pivot_expand(w[2:end], pe)
    Aod_theta = exp.(z)
    x_free = vcat(w[1], vec(Aod_theta))
    θ_full = CS.reconstruct_full(x_free, ctx.m)

    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))   # destinations touched
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache.baseIndex && d == cache.baseIndex, cells)

    q = copy(cache.q0)
    for d in affected_dests
        old_contrib = @view cache.contrib0[:, d]
        new_contrib = if tier == :block_local
            dest_contrib_block_local(cache, ctx, θ_full, d)
        elseif tier == :incremental
            origins_here = [o for (o, dd) in cells if dd == d]
            dest_contrib_incremental(cache, ctx, θ_full, d, origins_here)
        elseif tier == :incremental_o1
            origins_here = [o for (o, dd) in cells if dd == d]
            dest_contrib_incremental_o1(cache, ctx, θ_full, d, origins_here; multi_method = multi_method)
        else
            error("lfix_incremental_at: unknown tier=$tier")
        end
        # q_s = -zeta* - sum_d contrib[s,d] - cf_contrib[s] (contrib is defined WITHOUT the leading
        # minus, matching contrib_true_mat's own validated definition) -- an INCREASE in a
        # destination's contribution DECREASES q, hence subtract the delta, not add it. (Sign bug
        # caught by the self-validation-passing-but-perturbation-failing pattern: q0 alone was right,
        # only perturbation deltas had the flip.)
        q .-= new_contrib .- old_contrib
    end
    if cf_touched
        new_cf = cf_contrib_at(cache, θ_full, ctx)
        q .-= new_cf .- cache.cf_contrib0
    end

    return lfix_from_q(q, cache.ζstar)
end
