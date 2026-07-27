# ============================================================================
# Phase B (final-operator-stack-release, 2026-07-27): winner-aware economic x restriction cross
# Hessian, H_ER = Q'SR - pi*(nu'SR), for the CM-grid restriction block (flexible-CM's H_EC, and
# the CM block shared by CM+ZC/common-Frechet).
#
# DERIVATION (cross-checked against the already-validated H_EE winner-pair kernel,
# core_exact_hessian.jl::winner_pair_hessian!, not re-derived from a blank page): that function's
# own internal consistency pins down E's exact decomposition. Its GRADIENT-shaped accumulator
# (`u[j] += Snu[w]*y[w,slot]`, `Snu[w] = S[w]*nu[w]`, single power of nu) is the linear-in-E
# contraction Sigma_w S[w]*E[w,j]*1; its HESSIAN cross-correction accumulator (`r[j] +=
# Snu2[w]*y[w,slot]`, `Snu2[w] = S[w]*nu[w]^2`, double power of nu) is the bilinear-in-E*E
# contraction Sigma_w S[w]*E[w,i]*E[w,j] restricted to E's OWN two factors. A cross term against
# an EXTERNAL restriction column R[w,l] (not itself of the "nu*(...)" form E's rows carry) only
# ever contracts ONE factor of E against R -- so it uses the SAME single-nu-power `Snu` weight the
# gradient accumulator already uses, not `Snu2`. This is verified, not assumed: matching the
# dense H_EC formula (cm_hessian_architectures.jl, `S[x,j,bx] += ws*E[s,j]` with `ws=S[w]` alone
# and E[w,j] = nu[w]*(y[w,slot]*1{winner=j} - pi[j])) shows `ws*E[w,j] = Snu[w]*y*1{winner=j} -
# Snu[w]*pi[j]` exactly -- the decomposition this file implements.
#
# COMPLEXITY: replaces the dense build_bin_tables!'s O(W*D*NCORE) S-table fill (materializing
# obj.H's dense economic G columns) with O(W*D*Ddest) winner-bin accumulation (no dense E read at
# all) -- a D-fold reduction (NCORE = D*Ddest), and the whole point of Phase B's "eliminate dense
# economic moment columns" ask for this block.
# ============================================================================

isdefined(Main, :WinnerPairHessCtx) || include(joinpath(@__DIR__, "core_exact_hessian.jl"))
# Winner-aware H_ER phase (2026-07-27), §7: cross-Hessian backend-use counters live in the shared
# no_dense_g_counters.jl (NO_DENSE_G_COUNTERS), not a separate Ref here -- see that file's own
# record_winner_cross_hessian_call!/record_dense_cross_hessian_call! (added there this phase).
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))

"""
    WinnerBinCrossScratch

Persistent scratch for `winner_pair_cross_hessian_cm!`: `QTab[j,x,k] = Sigma_{w: winner(w,slot(j))=o(j)} Snu[w]*y[w,slot(j)]*1{bin(U[w,x])=k}`
(size `ncolI x D x (L+1)`) and `NuTab[x,k] = Sigma_w Snu[w]*1{bin(U[w,x])=k}` (size `D x (L+1)`,
independent of the economic column -- the `ν'SR` term is a SINGLE vector shared by every
economic-column row, per `H_ER = Q'SR - π(ν'SR)`'s own rank-1 structure). Cumulative
(prefix-summed over k<=l) twins `QCScum`/`NuCScum` follow the SAME `CS_x(j,l) = Σ_{k<=l}` naming
convention as `cm_hessian_architectures.jl`'s own `CScum`.
"""
mutable struct WinnerBinCrossScratch
    ncolI::Int
    D::Int
    L::Int
    QTab::Array{Float64,3}     # ncolI x D x (L+1)
    NuTab::Matrix{Float64}     # D x (L+1)
    SOnlyTab::Matrix{Float64}  # D x (L+1) -- row-1 ("ones"/zeta-paired H column) accumulator, S-only (no nu)
    QCfTab::Matrix{Float64}    # D x (L+1) -- the "cf"/common-factor column (cf.cf_col>0 only), accumulated
    # over EVERY sample (not winner-conditioned like QTab's regular economic columns -- cf.cf_raw is a
    # plain per-sample value, not a winner-selected one), weighted by Snu[w]*cf_raw_scaled[w] mirroring
    # `core_exact_hessian.jl`'s own `uu += Snu[w]*crs[w]` gradient-shaped accumulator for this column.
    QCScum::Array{Float64,3}   # ncolI x D x L
    NuCScum::Matrix{Float64}   # D x L
    SOnlyCScum::Matrix{Float64}  # D x L
    QCfCScum::Matrix{Float64}  # D x L
    # Common-Fréchet winner-aware H_ER phase (2026-07-27), Part B: UN-binned (no threshold/bin
    # dimension) per-economic-column accumulator `EsumEcon[j] = sum_w Snu[w]*y[w,slot(j)]*
    # 1{winner(w,slot(j))=o(j)}` (length ncolI, `wctx`'s own 1:ncolI numbering, NOT NCORE-offset).
    # Needed ONLY by common-Fréchet's H_E,level block (`winner_pair_cross_hessian_esum!` below) --
    # flexible-CM's own H_EC/H_CC blocks have no un-binned economic-column-sum term, so this field
    # is unused (but harmlessly filled) for a plain flexible-CM caller of this same scratch struct.
    EsumEcon::Vector{Float64}  # ncolI
end

function WinnerBinCrossScratch(ncolI::Int, D::Int, L::Int)
    return WinnerBinCrossScratch(ncolI, D, L,
        zeros(ncolI, D, L + 1), zeros(D, L + 1), zeros(D, L + 1), zeros(D, L + 1),
        zeros(ncolI, D, L), zeros(D, L), zeros(D, L), zeros(D, L),
        zeros(ncolI))
end

"Rebuild (or reuse, if already the right size) `ws` for the current `(ncolI, D, L)` -- mirrors this codebase's own `resize_*_if_needed!` idiom."
function ensure_winner_bin_cross_scratch!(ws_ref::Base.RefValue{Union{Nothing,WinnerBinCrossScratch}}, ncolI::Int, D::Int, L::Int)
    ws = ws_ref[]
    if ws === nothing || ws.ncolI != ncolI || ws.D != D || ws.L != L
        ws_ref[] = WinnerBinCrossScratch(ncolI, D, L)
    end
    return ws_ref[]
end

"""
    winner_pair_cross_hessian_cm!(Hraw_EC, obj, wctx::WinnerPairHessCtx, ws::WinnerBinCrossScratch,
        Bidx::AbstractMatrix{<:Integer}, L::Int, origins::Vector{Int}, refIndex1::Int) -> Hraw_EC

Fills `Hraw_EC` (`ncolI x nO`, ONE threshold block `l` at a time is NOT how this is organized --
unlike the dense per-`l` loop, this fills the FULL `ncolI x (nO*L)` raw cross block in one pass,
caller slices per `l` exactly as `hessian_cm_structured!`'s own `l`-loop already does when
applying the optional `R`-congruence and writing into `Hfull`) -- see `winner_pair_cross_hessian_cm_block!`
below for the per-`l` convenience wrapper matching that call site's own loop shape exactly.

Requires `obj.arg0` to already reflect the CURRENT (zeta,lambda) (same precondition as
`winner_pair_hessian!`) -- recomputes `ddPsi!` internally, does not require a fresh `obj.arg2`.
"""
function winner_pair_cross_hessian_fill!(wctx::WinnerPairHessCtx, ws::WinnerBinCrossScratch,
        obj, Bidx::AbstractMatrix{<:Integer})
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2
    Ddest = wctx.Ddest; W = wctx.W
    nu = wctx.nu; y = wctx.y; winner = wctx.winner
    D = ws.D; L = ws.L; nbins = L + 1

    QTab = ws.QTab; NuTab = ws.NuTab; SOnlyTab = ws.SOnlyTab; QCfTab = ws.QCfTab; EsumEcon = ws.EsumEcon
    fill!(QTab, 0.0); fill!(NuTab, 0.0); fill!(SOnlyTab, 0.0); fill!(QCfTab, 0.0); fill!(EsumEcon, 0.0)

    has_cf = wctx.has_cf
    crs = wctx.cf_raw_scaled
    @inbounds for w in 1:W
        Sw = S[w]; nuw = nu[w]
        snu = Sw * nuw
        snucf = has_cf ? snu * crs[w] : 0.0
        for x in 1:D
            b = Bidx[w, x]
            NuTab[x, b] += snu
            # row-1 ("ones" H column, obj.H[:,2].=1.0 -- compressed_live.jl) uses S alone, no nu:
            # that column is a literal constant-1 moment, not part of the winner/nu-weighted
            # economic-column family (matches winner_pair_hessian!'s own S_sum = Sigma_w S[w],
            # used unmodified for the (zeta,zeta) Hessian entry).
            SOnlyTab[x, b] += Sw
            has_cf && (QCfTab[x, b] += snucf)
        end
    end
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = winner[w, slot]
            j = slot + (o - 1) * Ddest
            snuy = (S[w] * nu[w]) * y[w, slot]
            # Common-Fréchet Part B: UN-binned accumulation (no x/Bidx loop) alongside the existing
            # per-bin QTab fill -- O(W*Ddest) additional work, negligible next to QTab's own
            # O(W*Ddest*D). See EsumEcon's own field docstring above.
            EsumEcon[j] += snuy
            for x in 1:D
                QTab[j, x, Bidx[w, x]] += snuy
            end
        end
    end

    QCScum = ws.QCScum; NuCScum = ws.NuCScum; SOnlyCScum = ws.SOnlyCScum; QCfCScum = ws.QCfCScum
    @inbounds for x in 1:D
        acc = 0.0; accS = 0.0; accCf = 0.0
        for l in 1:L
            acc += NuTab[x, l]
            NuCScum[x, l] = acc
            accS += SOnlyTab[x, l]
            SOnlyCScum[x, l] = accS
            accCf += QCfTab[x, l]
            QCfCScum[x, l] = accCf
        end
    end
    @inbounds for x in 1:D, j in 1:ws.ncolI
        acc = 0.0
        for l in 1:L
            acc += QTab[j, x, l]
            QCScum[j, x, l] = acc
        end
    end
    return ws
end

"""
    winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, ws, l, origins, refIndex1, M) -> Hraw_EC

Per-threshold-block (`l`) raw `H_EC` slab, `NCORE x nO` where `NCORE = wctx.ncolI + 1` (row 1 =
the "ones"/zeta-paired `H[:,2]` column, S-only weighted, no `pi_vec` correction since that column
has no entry in `pi_vec`; rows 2:NCORE = the `ncolI` real economic-lambda columns, row `j+1`
corresponding to `wctx`'s own column `j`) -- this `+1` row offset matches `cm_hessian_
architectures.jl`'s own `E = @view H[:, 2:1+NCORE]` slicing EXACTLY (`E`'s first column is the
ones column, not an economic one), so this drops into that call site as a straight replacement
for the dense `CS_`-table read at the SAME row indices. Caller must call
`winner_pair_cross_hessian_fill!` ONCE per Hessian callback first (builds `QCScum`/`NuCScum`/
`SOnlyCScum` for ALL `l` at once), then this per-`l` slice is O(NCORE*nO), matching the dense
version's own per-`l` cost.
"""
function winner_pair_cross_hessian_cm_block!(Hraw_EC::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerBinCrossScratch, l::Int, origins::Vector{Int}, refIndex1::Int, M)
    QCScum = ws.QCScum; NuCScum = ws.NuCScum; SOnlyCScum = ws.SOnlyCScum; QCfCScum = ws.QCfCScum
    pi_vec = wctx.pi_vec
    invM = 1.0 / M
    has_cf = wctx.has_cf
    jcf = wctx.ncolI   # the cf column's index WITHIN wctx's own 1:ncolI numbering (cf.cf_col)
    @inbounds for (oi, o) in enumerate(origins)
        Hraw_EC[1, oi] = (SOnlyCScum[o, l] - SOnlyCScum[refIndex1, l]) * invM
        nu_diff = NuCScum[o, l] - NuCScum[refIndex1, l]
        for j in 1:wctx.ncolI
            q_diff = QCScum[j, o, l] - QCScum[j, refIndex1, l]
            Hraw_EC[j + 1, oi] = (q_diff - pi_vec[j] * nu_diff) * invM
        end
        # The "cf"/common-factor column (if present) is NOT winner-conditioned like the regular
        # economic columns above -- QTab[jcf,:,:] was left at zero by the main slot-loop (no
        # sample's `winner[w,slot]` ever equals it, it isn't a bilateral (slot,origin) pair at
        # all), so overwrite that one row here with its own dedicated accumulation.
        if has_cf
            qcf_diff = QCfCScum[o, l] - QCfCScum[refIndex1, l]
            Hraw_EC[jcf + 1, oi] = (qcf_diff - pi_vec[jcf] * nu_diff) * invM
        end
    end
    return Hraw_EC
end

# ============================================================================
# Common-Fréchet winner-aware H_ER phase (2026-07-27), Part B: the level-anchor block's H_E,level
# needs `sum_{o=1}^D CS_[o,j,l]` (a SUM over all D origins, unlike H_EC's own (o,ref)-DIFFERENCE)
# and the UN-binned column sum `Esum[j] = sum_s w[s]*E[s,j]` -- neither is provided by
# `winner_pair_cross_hessian_cm_block!` above, which only ever produces per-threshold DIFFERENCES.
# Both are cheap, O(D) and O(1) respectively per (j,l)/j, reusing the SAME cumulative tables
# `winner_pair_cross_hessian_fill!` already builds (plus the new `EsumEcon` field above for the
# UN-binned half) -- no second O(W) pass. See docs/COMMON_FRECHET_WINNER_AWARE_HER_RELEASE_2026-07-27.md
# for the full derivation and the cross-check against the dense `CS_`/`Esum` formulas this replaces.
# ============================================================================

"""
    winner_pair_cross_hessian_colsum!(colsum, wctx, ws, l) -> colsum

`colsum[j] = sum_{o=1}^D CS_[o,j,l]` for `j = 1:NCORE` (`NCORE = wctx.ncolI + 1`, SAME row
convention as `winner_pair_cross_hessian_cm_block!`: row 1 = the ones/zeta column, S-only; rows
2:NCORE = the `ncolI` economic-lambda columns, row `j+1` <-> `wctx`'s own column `j`) -- the
SUM-over-all-origins counterpart to that function's own (o,ref)-DIFFERENCE, needed by common-
Fréchet's level restriction (a sum over all D origins with weight `1/sqrt(D)`, not a CM-style
difference against `refIndex1`). `O(D)` per row, `O(D*NCORE)` total per `l` -- same complexity class
as the dense `for o in 1:D; acc += CS_[o,j,l]; end` loop this replaces. Caller must call
`winner_pair_cross_hessian_fill!` once per Hessian callback first (same precondition as
`winner_pair_cross_hessian_cm_block!`).
"""
function winner_pair_cross_hessian_colsum!(colsum::AbstractVector{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerBinCrossScratch, l::Int)
    QCScum = ws.QCScum; NuCScum = ws.NuCScum; SOnlyCScum = ws.SOnlyCScum; QCfCScum = ws.QCfCScum
    pi_vec = wctx.pi_vec
    has_cf = wctx.has_cf
    jcf = wctx.ncolI   # the cf column's index WITHIN wctx's own 1:ncolI numbering (cf.cf_col)
    D = ws.D

    sumNu = 0.0
    @inbounds for o in 1:D
        sumNu += NuCScum[o, l]
    end

    sumS = 0.0
    @inbounds for o in 1:D
        sumS += SOnlyCScum[o, l]
    end
    colsum[1] = sumS

    @inbounds for j in 1:wctx.ncolI
        sumQ = 0.0
        for o in 1:D
            sumQ += QCScum[j, o, l]
        end
        colsum[j + 1] = sumQ - pi_vec[j] * sumNu
    end

    # Same cf-column override as winner_pair_cross_hessian_cm_block! -- QCScum[jcf,:,:] was left at
    # zero by the main slot-loop (the cf column is not a (slot,origin) pair), so overwrite that one
    # entry with its own dedicated accumulation.
    if has_cf
        sumQCf = 0.0
        @inbounds for o in 1:D
            sumQCf += QCfCScum[o, l]
        end
        colsum[jcf + 1] = sumQCf - pi_vec[jcf] * sumNu
    end
    return colsum
end

"""
    winner_pair_cross_hessian_esum!(Esum, wctx, ws, w, Wtot) -> Esum

`Esum[j] = sum_s w[s]*E[s,j]` for `j = 1:NCORE` -- the UN-binned (no threshold dependence) column
sum common-Fréchet's `H_E,level` block needs for its nonzero-target correction term (the level
feature, unlike CM's own zero-target raw features, subtracts `target[l]` -- see
`cm_frechet_hessian.jl`'s own header derivation). `j=1` (the ones/zeta column, `E[:,1]≡1`) is
`Wtot = sum(w)` exactly, passed in rather than recomputed (callers already have it for the
UNCHANGED `H_level,level` block). `j=2:NCORE` uses `ws.EsumEcon[j-1] - wctx.pi_vec[j-1]*t0`,
`t0 = sum_w S[w]*nu[w]` (`S` is `w` here -- `winner_pair_cross_hessian_fill!`'s own precondition is
that `obj.arg2` already reflects the current weights, same `w` this function receives). Derivation:
`w[s]*E[s,j] = S[w]*nu[w]*(y[w,slot]*1{winner=j} - pi[j]) = Snu[w]*y*1{winner=j} - Snu[w]*pi[j]`
(the SAME decomposition `winner_pair_cross_hessian_fill!`'s own header comment already establishes
for the binned case) -- summing over `s`/`w` and using `EsumEcon`'s own un-binned accumulation gives
this formula directly. Caller must call `winner_pair_cross_hessian_fill!` once per Hessian callback
first (builds `EsumEcon`).

BUGFIX (found via this family's own D=4 wiring gate, 2026-07-27): the "cf"/common-factor column
(`wctx.has_cf`, index `wctx.ncolI` within `wctx`'s own numbering) is NOT a (slot,origin) pair, so
(exactly like `QTab[jcf,:,:]` in the binned case, see `winner_pair_cross_hessian_cm_block!`'s own
cf override) `EsumEcon[jcf]` is left at zero by `winner_pair_cross_hessian_fill!`'s slot loop --
using it unconditionally for `Esum[jcf+1]` silently dropped the entire cf-column contribution,
undercounting `Esum[jcf+1]` by exactly `ecf` below. Overwritten here with the correct dedicated
accumulation (`ecf = sum_w S[w]*nu[w]*cf_raw_scaled[w]`, mirroring `winner_pair_cross_hessian_fill!`'s
own `snucf = snu*crs[w]` weighting for `QCfTab`), computed in the SAME O(W) pass as `t0` (no second
traversal).
"""
function winner_pair_cross_hessian_esum!(Esum::AbstractVector{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerBinCrossScratch, w::AbstractVector{Float64}, Wtot::Float64)
    Esum[1] = Wtot
    nu = wctx.nu
    has_cf = wctx.has_cf
    crs = wctx.cf_raw_scaled
    t0 = 0.0
    ecf = 0.0
    if has_cf
        @inbounds for s in eachindex(w)
            snu = w[s] * nu[s]
            t0 += snu
            ecf += snu * crs[s]
        end
    else
        @inbounds for s in eachindex(w)
            t0 += w[s] * nu[s]
        end
    end
    pi_vec = wctx.pi_vec
    EsumEcon = ws.EsumEcon
    @inbounds for j in 1:wctx.ncolI
        Esum[j + 1] = EsumEcon[j] - pi_vec[j] * t0
    end
    if has_cf
        jcf = wctx.ncolI
        Esum[jcf + 1] = ecf - pi_vec[jcf] * t0
    end
    return Esum
end
