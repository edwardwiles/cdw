# ============================================================================
# Compressed winner-form representation of the FACTUAL bilateral moments.
#
# ADDITIVE / DIAGNOSTIC ONLY. Does NOT modify the trusted dense constructor
# (full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp!), the
# tie-breaking rule (misc/smoothMinIndNew!.jl::MinInd!), lfix_incremental.jl,
# or composite_gradient_fast.jl. It re-derives -- in a single self-contained
# module -- the same winner-form contraction that lfix_incremental.jl's
# `build_lfix_base_cache`/`contrib0` already exploits (and self-validates), but
# packaged as a standalone compressed representation + dual contraction with
# the full column-normalization bookkeeping (SamplingWeights / NormalizeMoments
# / usePMM / gammafac) made explicit so it is provably correct beyond the one
# specific ctx config, and with exact-tie detection (reusing lfix's
# `detect_price_ties`/`TiedWinnerError`) built in from the start.
#
# EXACT FACTUAL FORMULA (verified line-by-line against hFunction.jl +
# moments_gammanorm.jl, UoModel==1 branch, counterType==1 autarky):
#
#   Raw hFunction! output, bilateral column d1 = d + (o-1)*D:
#       r_{s,(o,d)} = pTσ_{s,o,d} * 1{o = winner_{s,d}}  -  P_{(o,d)} * denom_d
#     winner_{s,d} = argmin_o  price_{s,o,d},   price = constCons_{o,d} / U_{s,o}^{-μ}
#     pTσ_{s,o,d}  = constConsσ_{o,d} / Uσ_{s,o}^{-μ}          (the "winning CES value" v)
#     constCons_{o,d}  = wHat_o * AodPow_{o,d} * τ_{o,d}
#     constConsσ_{o,d} = wHat_o^{1-σ} * (AodPow_{o,d} * τ_{o,d})^{1-σ}
#     denom_d = γ_d^σ * (wHat_d * L_d) = wHat_d * L_d      (γ_d ≡ 1 normalization)
#     P = observed bilateral shares λ̂  (ctx.γ.P, reshaped)
#
#   Post-processing applied by EK_moments_gammanorm_directgp! (in order):
#     (1) cols 1..D^2+1 divided by gammafac = Γ(μ(1-σ)+1)
#     (2) if usePMM:  col j -= PMM_j     (j = 1..numMomentsSimple)
#     (3) if NormalizeMoments: col j *= 1/σ_Moments_j   (j ∉ moments_without_var)
#     (4) col j *= SamplingWeights_s   (all j)
#   =>  G_{s,j} = SW_s * nrm_j * ( r_{s,j} * gdiv_j  -  usePMM*PMM_j )
#       gdiv_j = 1/gammafac if j ≤ D^2+1 else 1 ;  nrm_j = 1/σ_M_j (or 1)
#
#   NOTE (correction to the user's schematic G_{·d,s}=v_{sd}(e_{w}-λ̂_{·d})):
#   the centering term is DRAW-INDEPENDENT (−P_{od}·denom_d), NOT scaled by the
#   per-draw winner value v_{sd}. So the exact fixed-dual contraction over the
#   bilateral block is
#       Σ_{o,d} β_{od} G_{s,(o,d)}
#         = SW_s * [ Σ_d κ_{win_sd,d} v_{s,d}  +  Σ_d C_d  −  usePMM·<β,nrm·PMM> ]
#   with κ_{o,d} = β_{o,d}·nrm_{o,d}·gdiv_{o,d}  and  C_d = −denom_d Σ_o κ_{o,d} P_{o,d}
#   -- a per-draw O(D) sum (winner pick per destination) plus draw-independent
#   constants, i.e. O(W·D) total, vs O(W·D^2) for the dense mat-vec.
#
#   The counterfactual price-index column (d1 = D^2+1) is a single extra column,
#   r_{s} = constConsσ'_{bi,bi} / Uσ_{s,bi}^{-μ} − denom'_{bi}  (primed quantities),
#   handled exactly (O(W)) alongside the bilateral block.
# ============================================================================

using LinearAlgebra: dot
using SpecialFunctions: gamma as spgamma

# core-moment-cache-benchmark task (2026-07-24): defensive self-include, matching this codebase's
# own convention (oracle_fast.jl:60, winner_certificate.jl:48) of each file pulling in its own
# @prof/PROF_ENABLED dependency rather than relying on include order -- this file is included
# (c10_d20_production_driver.jl) BEFORE oracle_fast.jl, so @prof would otherwise be undefined here.
include(joinpath(@__DIR__, "instrumentation.jl"))

# Canonical winner engine (2026-07-24 unification task): defensive self-include (see
# fast_range_screen.jl's identical `isdefined(Main, :CompressedFactual) || include(...)` pattern)
# -- this file loads before winner_certificate.jl's normal load point in the driver's include
# chain, so canonical_price_precompute/canonical_winner_argmin must be pulled in explicitly here
# for build_compressed_factual to use.
isdefined(Main, :WinnerRefCache) || include(joinpath(@__DIR__, "winner_certificate.jl"))

# ============================================================================
# Shared economic moment-state builder runtime counters (2026-07-27 task).
# Defined HERE (not compressed_factual_buffer_reuse.jl) so they exist for every one of the
# ~50 existing bench/test scripts that `include(compressed_moments.jl)` WITHOUT also including
# compressed_factual_buffer_reuse.jl -- those scripts still call the allocating
# `build_compressed_factual` below, and it must be able to increment
# ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS unconditionally without an UndefVarError.
# compressed_factual_buffer_reuse.jl's in-place machinery (included after this file in every
# production driver) increments the remaining five counters.
# ============================================================================
"defined-hot-path allocating call count; must be 0 across an ordinary production driver run"
const ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS = Ref(0)
"in-place, non-allocating build_compressed_factual! call count"
const INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS = Ref(0)
"CompressedFactualWorkspace object creations (build_compressed_factual_workspace calls); should be 1 per live production context after warm-up"
const ECONOMIC_WORKSPACE_ALLOCATIONS = Ref(0)
"in-place refills of an existing workspace at a genuinely new outer point (θ_full changed since the workspace's last fill)"
const ECONOMIC_WORKSPACE_REFILLS = Ref(0)
"workspace rebuilds triggered by a genuine (D,Ddest,W) shape change on an ALREADY-attached workspace; must be 0 after warm-up"
const ECONOMIC_WORKSPACE_RESIZES = Ref(0)
"in-place builds whose θ_full is bit-identical to the immediately preceding build on the SAME workspace -- i.e. the economic state was reconstructed twice for what is functionally the same outer point"
const DUPLICATE_ECONOMIC_STATE_BUILDS = Ref(0)

"Resets all six shared economic moment-state builder runtime counters to zero. Call once at the start of a measurement window (a gate script, a fresh driver run) -- these are process-global Refs, not reset automatically."
function reset_economic_moment_state_counters!()
    ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[] = 0
    INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS[] = 0
    ECONOMIC_WORKSPACE_ALLOCATIONS[] = 0
    ECONOMIC_WORKSPACE_REFILLS[] = 0
    ECONOMIC_WORKSPACE_RESIZES[] = 0
    DUPLICATE_ECONOMIC_STATE_BUILDS[] = 0
    return nothing
end

"Snapshot of all six counters as a NamedTuple, for logging in a public driver's final verdict block."
economic_moment_state_counters() = (
    allocating_build_compressed_factual_calls = ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[],
    inplace_build_compressed_factual_calls = INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS[],
    economic_workspace_allocations = ECONOMIC_WORKSPACE_ALLOCATIONS[],
    economic_workspace_refills = ECONOMIC_WORKSPACE_REFILLS[],
    economic_workspace_resizes = ECONOMIC_WORKSPACE_RESIZES[],
    duplicate_economic_state_builds = DUPLICATE_ECONOMIC_STATE_BUILDS[],
)

"""
    CompressedFactual

Compressed winner-form representation of the factual moment matrix over the
inner-dual columns 1..oci-1 (= D*D_dest bilateral + 1 counterfactual price
index). Stores O(W·D_dest) + O(D·D_dest) data, NOT the dense O(W·D·D_dest)
matrix. Everything needed to reproduce `dot(β, G[s,1:oci-1])` exactly for any
dual β.

RECTANGULAR (exclude-ROW-destination unrestricted-core release, 2026-07-24):
`D` = number of origins (always the full country count, origins are never
restricted); `D_dest` = number of ACTIVE destinations (`D_dest == D` under
`destination_sample=:all_legacy`, `D_dest == D-1` under `:exclude_row`).
Bilateral arrays are `D x D_dest` (or `W x D_dest`), never `D x D`. The
flattened bilateral-column index is the destination-fast convention
`j = s + (o-1)*D_dest` (`s` = LOCAL active-destination slot 1..D_dest, `o` =
global origin 1..D) -- see `cc_algo/active_layout.jl`'s
`active_cell_index`/MEMORY moments-vs-aod-linear-index-convention. When
`D_dest == D` (square/`:all_legacy`) this collapses to the pre-existing
`j = d + (o-1)*D` legacy formula bit-for-bit.
"""
struct CompressedFactual
    D::Int
    D_dest::Int
    W::Int
    oci::Int
    # --- winner-form bilateral block ---
    winner::Matrix{Int}         # W x D_dest : argmin origin per (draw, active-destination slot)
    wval::Matrix{Float64}       # W x D_dest : v_{s,slot} = pTσ of the winning origin
    # --- fixed (draw-independent) data used by the contraction ---
    Pmat::Matrix{Float64}       # D x D_dest : observed bilateral shares, Pmat[o,slot]=P[slot+(o-1)D_dest]
    denom::Vector{Float64}      # D_dest
    gdiv::Vector{Float64}       # length oci-1 : 1/gammafac (cols ≤ D*D_dest+1) else 1
    nrm::Vector{Float64}        # length oci-1 : NormalizeMoments factor (or 1)
    PMM::Vector{Float64}        # length oci-1 : per-moment PMM (used only if usePMM==1)
    usePMM::Int
    SW::Vector{Float64}         # W : sampling weights
    gammafac::Float64
    # --- counterfactual price-index column (col D*D_dest+1) ---
    cf_raw::Vector{Float64}     # W : raw hFunctionCounter! value (before post-proc)
    cf_col::Int                 # = D*D_dest+1 (0 if this column is not an inner-dual column)
    # --- tie bookkeeping ---
    n_tied::Int
    tied_examples::Vector{Tuple{Int,Int}}
end

"""
    build_compressed_factual(θ_full, ctx; check_ties=true) -> CompressedFactual

Build the compressed winner-form representation at `θ_full`. Winner-finding is
O(W·D^2) (irreducible: MinInd! compares D origins per (draw,destination)), but
only the WINNING origin's σ-value is evaluated per (draw,destination) (O(W·D)),
vs the dense path evaluating all D and zeroing losers.

If `check_ties` and any (draw,destination) has 2+ origins bit-exactly tied at
the row-min price, throws `TiedWinnerError` (reusing lfix_incremental.jl's type)
-- the one-winner assumption does not hold there (see MEMORY tie-bug note).

ALLOCATING REFERENCE IMPLEMENTATION (shared economic moment-state builder task,
2026-07-27): this is the freshly-heap-allocating correctness reference / test helper /
one-off diagnostic convenience wrapper. It must NOT be called on any repeated production
hot path -- use `build_economic_moment_state!`/`cf_build` (compressed_factual_buffer_reuse.jl),
which dispatch to the in-place, non-allocating `build_compressed_factual!` whenever `ctx`
carries an attached `cf_workspace`. See `ALLOCATING_COMPRESSED_FACTUAL_CALLSITE_AUDIT_2026-07-27.md`
for the full call-site classification. Every call here increments the runtime counter
`ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS` (defined below) so a production driver run can be
audited for accidental use of this allocating path.
"""
function build_compressed_factual(θ_full::AbstractVector, ctx; check_ties::Bool = true)
    ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[] += 1
    γo = ctx.γ
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    U = ctx.U; W = size(U, 1)
    μ = θ_full[1]; σ = θ_full[2]
    ind = γo.indicators
    oci = ctx.obj.outer_constr_index

    # ---- draw-independent per-cell constants (D x D_dest) ----
    # Canonical winner engine (2026-07-24): shared precompute, replacing this function's own
    # hand-derived Aod/AodPow/constCons/constConsσ/UPow/UσPow block -- byte-identical formula to
    # canonical_price_precompute (winner_certificate.jl), now the single shared source also used by
    # screen_hard_winners/screen_hard_winners_ranged. (lambda/Aod_θ reshape conventions unchanged:
    # D_dest-fast for γo.P/lambda, the "moments" stride; D-fast for the Aod parameter block -- see
    # MEMORY moments-vs-aod-linear-index-convention.)
    pp = canonical_price_precompute(θ_full, ctx)
    constCons = pp.constCons; constConsσ = pp.constConsσ; logCC = pp.logCC
    mulU = pp.mulU; UPow = pp.UPow; UσPow = pp.UσPow; AodPow = pp.AodPow
    # denom is a per-COUNTRY (γo.wHat/γo.L are GLOBAL, length-D, country-indexed) quantity, so the
    # active-destination slot s must be mapped to its GLOBAL country index -- global_destination(ctx,s)
    # -- NOT used as a global index directly (that would be wrong whenever the omitted destination
    # is not the last global index; harmless-but-fragile no-op in today's production ROW-is-last
    # layout, real bug in general -- see cc_algo/active_layout.jl).
    denom = [γo.wHat[global_destination(ctx, s)] * γo.L[global_destination(ctx, s)] for s in 1:Ddest]  # γ_d ≡ 1
    Pmat = [γo.P[s + (o - 1) * Ddest] for o in 1:D, s in 1:Ddest]  # Pmat[o,s]=P[a(o,s)], already slot-local

    winner = Matrix{Int}(undef, W, Ddest)
    wval = Matrix{Float64}(undef, W, Ddest)
    tied = Tuple{Int,Int}[]
    n_tied = 0
    @inbounds for s in 1:Ddest
        for w in 1:W
            # Winner IDENTIFICATION via the shared fast log-additive argmin (canonical_winner_argmin)
            # -- see fast_range_screen.jl::screen_hard_winners_ranged's identical swap for the full
            # rationale/correctness argument. `check_ties` below is UNCHANGED (still price-space).
            bo, _ = canonical_winner_argmin(logCC, mulU, w, s, D)
            best = constCons[bo, s] / UPow[w, bo]
            winner[w, s] = bo
            wval[w, s] = constConsσ[bo, s] / UσPow[w, bo]
            if check_ties
                # MinInd! sets xInd[o]=1 for EVERY o with price <= min: count them.
                c = 0
                for o in 1:D
                    (constCons[o, s] / UPow[w, o] <= best) && (c += 1)
                end
                if c > 1
                    n_tied += 1
                    length(tied) < 5 && push!(tied, (w, s))
                end
            end
        end
    end
    if check_ties && n_tied > 0
        throw(TiedWinnerError(n_tied, tied))
    end

    # ---- normalization / post-processing vectors over inner-dual columns ----
    gammafac = spgamma(μ * (1 - σ) + 1)
    ncol = oci - 1
    ncell = D * Ddest
    gdiv = [j <= ncell + 1 ? 1.0 / gammafac : 1.0 for j in 1:ncol]
    NM = ind.NormalizeMoments
    without = γo.moments_without_var
    nrm = [(NM == 1 && !(j in without)) ? 1.0 / γo.σ_Moments[j] : 1.0 for j in 1:ncol]
    usePMM = ind.usePMM
    PMMv = usePMM == 1 ? Float64[γo.PMM[j] for j in 1:ncol] : zeros(ncol)
    SW = γo.SamplingWeights[1:W]

    # ---- counterfactual price-index column (D*D_dest+1), if it is an inner-dual col ----
    cf_col = ncell + 1
    cf_raw = zeros(W)
    if cf_col <= ncol
        bi = ctx.bi
        wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
        wPrime_bi = wPrime[bi]                       # ==1
        τPrime_bi = γo.τPrime[bi, bi]
        LPrime_bi = γo.LPrime[bi]
        # AodPow's SECOND axis is now the LOCAL active-destination slot, not a global country
        # index -- bi (a global index) must be translated via dest_slot(ctx,bi) before indexing.
        # (the focal-country==ROW guard in context_real_d20.jl guarantees bi is always an active
        # destination, so this never errors in production).
        AodPow_bibi = AodPow[bi, dest_slot(ctx, bi)]  # same factual AodPow (hFunctionCounter! is passed AodPow)
        γ_prime_bi = θ_full[3 + D]
        constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi * τPrime_bi)^(1 - σ)
        denom_cf = γ_prime_bi^σ * (wPrime_bi * LPrime_bi)
        UσPow_bi = @view(γo.Uσ[:, bi]) .^ (-μ)
        @. cf_raw = constConsσ_bibi / UσPow_bi - denom_cf
    else
        cf_col = 0
    end

    return CompressedFactual(D, Ddest, W, oci, winner, wval, Pmat, denom, gdiv, nrm,
        PMMv, usePMM, SW, gammafac, cf_raw, cf_col, 0, Tuple{Int,Int}[])
end

"""
    compressed_dual_contraction(β, cf::CompressedFactual) -> Vector{W}

Exact compressed evaluation of `t_s = Σ_{j=1}^{oci-1} β_j · G_{s,j}` for every
draw s, WITHOUT materializing the dense G. β has length oci-1 (= D*D_dest
bilateral followed by the counterfactual column). O(W·D_dest) work, O(D·D_dest)
setup.
"""
function compressed_dual_contraction(β::AbstractVector, cf::CompressedFactual)
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    length(β) == cf.oci - 1 || error("β length $(length(β)) != oci-1 = $(cf.oci-1)")

    # κ_{o,slot} = β_{(o,slot)} · nrm · gdiv   (bilateral cols)
    κ = Matrix{Float64}(undef, D, Ddest)
    C = zeros(Ddest)                               # C_slot = −denom_slot Σ_o κ_{o,slot} P_{o,slot}
    @inbounds for slot in 1:Ddest
        acc = 0.0
        for o in 1:D
            j = slot + (o - 1) * Ddest
            k = β[j] * cf.nrm[j] * cf.gdiv[j]
            κ[o, slot] = k
            acc += k * cf.Pmat[o, slot]
        end
        C[slot] = -cf.denom[slot] * acc
    end
    Csum = sum(C)

    # counterfactual column coefficient
    κ_cf = cf.cf_col > 0 ? β[cf.cf_col] * cf.nrm[cf.cf_col] * cf.gdiv[cf.cf_col] : 0.0

    # PMM constant term  usePMM·Σ_j β_j nrm_j PMM_j
    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for j in 1:(cf.oci - 1)
            pmmterm += β[j] * cf.nrm[j] * cf.PMM[j]
        end
    end

    t = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        acc = Csum
        for slot in 1:Ddest
            acc += κ[cf.winner[w, slot], slot] * cf.wval[w, slot]
        end
        acc += κ_cf * cf.cf_raw[w]
        t[w] = cf.SW[w] * (acc - pmmterm)
    end
    return t
end

"""
    compressed_dual_contraction!(t, β, cf, κ, C) -> t

Addendum Part A remediation (2026-07-26): in-place analogue of `compressed_dual_contraction`,
writing into caller-supplied `t` (length W) using persistent `κ` (D x Ddest) / `C` (Ddest) scratch
instead of allocating fresh each call. Identical math/order of operations -- see the allocating
original's own docstring for the full derivation. The allocating original is UNCHANGED and kept
(still used by `dual_bank.jl`'s cheap scorer, `theta_cplus.jl`, and various benchmark/test
scripts, none of them the hot per-FG-callback path this exists to fix).
"""
function compressed_dual_contraction!(t::AbstractVector{Float64}, β::AbstractVector, cf::CompressedFactual,
                                       κ::AbstractMatrix{Float64}, C::AbstractVector{Float64})
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    length(β) == cf.oci - 1 || error("β length $(length(β)) != oci-1 = $(cf.oci-1)")

    @inbounds for slot in 1:Ddest
        acc = 0.0
        for o in 1:D
            j = slot + (o - 1) * Ddest
            k = β[j] * cf.nrm[j] * cf.gdiv[j]
            κ[o, slot] = k
            acc += k * cf.Pmat[o, slot]
        end
        C[slot] = -cf.denom[slot] * acc
    end
    Csum = sum(C)

    κ_cf = cf.cf_col > 0 ? β[cf.cf_col] * cf.nrm[cf.cf_col] * cf.gdiv[cf.cf_col] : 0.0

    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for j in 1:(cf.oci - 1)
            pmmterm += β[j] * cf.nrm[j] * cf.PMM[j]
        end
    end

    @inbounds for w in 1:W
        acc = Csum
        for slot in 1:Ddest
            acc += κ[cf.winner[w, slot], slot] * cf.wval[w, slot]
        end
        acc += κ_cf * cf.cf_raw[w]
        t[w] = cf.SW[w] * (acc - pmmterm)
    end
    return t
end

"""
    materialize_dense_factual(cf::CompressedFactual) -> Matrix{Float64}

DIAGNOSTIC: reconstruct the dense W x (oci-1) factual moment matrix from the
compressed representation (bilateral + counterfactual columns), applying the
exact post-processing. For equivalence checks against obj.moments!'s G[:,1:oci-1].
"""
function materialize_dense_factual(cf::CompressedFactual)
    D = cf.D; Ddest = cf.D_dest; W = cf.W; ncol = cf.oci - 1
    G = zeros(W, ncol)
    @inbounds for slot in 1:Ddest, w in 1:W
        wo = cf.winner[w, slot]
        v = cf.wval[w, slot]
        for o in 1:D
            j = slot + (o - 1) * Ddest
            r = (o == wo ? v : 0.0) - cf.Pmat[o, slot] * cf.denom[slot]
            G[w, j] = cf.SW[w] * cf.nrm[j] * (r * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    if cf.cf_col > 0
        j = cf.cf_col
        @inbounds for w in 1:W
            G[w, j] = cf.SW[w] * cf.nrm[j] * (cf.cf_raw[w] * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    return G
end

"""
    materialize_dense_factual!(Gview, cf::CompressedFactual)

ADDITIVE (continuation 8, live-integration): in-place variant of
`materialize_dense_factual`, writing into a caller-supplied `W x (oci-1)`
view/matrix instead of allocating a fresh one. Used by the live compressed
Hessian-callback adapter (`compressed_live.jl`) to fill `obj.H`'s existing
dense G columns from an already-built `CompressedFactual` -- this is cheaper
than a from-scratch dense `moments!` call because `cf.winner`/`cf.wval`
(the expensive part: winner search + per-winner sigma-value) are already
computed; this is pure O(W*D) broadcast-equivalent write, no search, no
sigma-value evaluation for losers. Identical formula to
`materialize_dense_factual`, just avoiding the allocation -- not separately
re-derived.
"""
function materialize_dense_factual!(Gview::AbstractMatrix, cf::CompressedFactual)
    D = cf.D; Ddest = cf.D_dest; W = cf.W; ncol = cf.oci - 1
    size(Gview) == (W, ncol) || error("materialize_dense_factual!: size(Gview)=$(size(Gview)) != (W,oci-1)=($W,$ncol)")
    @inbounds for slot in 1:Ddest, w in 1:W
        wo = cf.winner[w, slot]
        v = cf.wval[w, slot]
        for o in 1:D
            j = slot + (o - 1) * Ddest
            r = (o == wo ? v : 0.0) - cf.Pmat[o, slot] * cf.denom[slot]
            Gview[w, j] = cf.SW[w] * cf.nrm[j] * (r * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    if cf.cf_col > 0
        j = cf.cf_col
        @inbounds for s in 1:W
            Gview[s, j] = cf.SW[s] * cf.nrm[j] * (cf.cf_raw[s] * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    return Gview
end
