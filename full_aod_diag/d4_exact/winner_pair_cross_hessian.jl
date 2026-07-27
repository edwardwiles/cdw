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
end

function WinnerBinCrossScratch(ncolI::Int, D::Int, L::Int)
    return WinnerBinCrossScratch(ncolI, D, L,
        zeros(ncolI, D, L + 1), zeros(D, L + 1), zeros(D, L + 1), zeros(D, L + 1),
        zeros(ncolI, D, L), zeros(D, L), zeros(D, L), zeros(D, L))
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

    QTab = ws.QTab; NuTab = ws.NuTab; SOnlyTab = ws.SOnlyTab; QCfTab = ws.QCfTab
    fill!(QTab, 0.0); fill!(NuTab, 0.0); fill!(SOnlyTab, 0.0); fill!(QCfTab, 0.0)

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
# Winner-aware H_ER phase (2026-07-27), Sections 4/5: winner-aware economic x mean/pairwise-ZC
# cross Hessian, H_EZ = E'SZ, Z = Phi - 1*t' (the mean/pair restriction block, already-centered --
# `zc_restriction_operator.jl`'s own `Φ - 1t'` decomposition). ONE shared primitive, used by BOTH
# CM+ZC (`cm_meanzc_production.jl`) and origin-ZC-only (`cm_originzc_production.jl`) call sites --
# task brief is explicit these two families must share this cross function, not have two
# independent derivations.
#
# DERIVATION: H_EZ = E'S*Phi - (E'S*1)*t' = (E'S*Phi) - Esum*t' (task brief's own decomposition,
# Esum = E'S*1). This function does NOT thread `Phi`/`t`/`Esum` separately -- it takes the
# ALREADY-CENTERED `Z = Phi - 1*t'` directly, which is algebraically IDENTICAL
# (`E'S*(Phi - 1t') = E'S*Phi - (E'S*1)*t'`, the same quantity) and is what BOTH production call
# sites already have sitting in `obj.H`/`E` for free every Hessian callback regardless of backend
# (`moments!` writes `dest = Z - νtargets'` into H's mean/pair columns every call, independent of
# which backend fills H_EE) -- reading it costs nothing extra and is NOT part of the "no dense G"
# invariant this port is about, which targets ONLY the winner-conditioned ECONOMIC (E) columns,
# never the cheap-by-construction restriction columns (task brief: "the goal ... is NOT
# necessarily fewer FLOPs than the dense E'S*Phi gemm ... it's eliminating the dense read of
# obj.H's economic columns (E)"). Centering `Z` before this function sees it also avoids threading
# the per-outer-point target-layout machinery (`mean_targets`/`pair_targets`/`νfull`) through the
# Hessian callback at all -- neither wiring call site has convenient access to the current outer
# point's `νfull` at Hessian-build time, while both already have `Z` (centered) sitting in a local
# dense view. The standalone gate below validates this decomposition is exact by comparing against
# a fully independent dense `E'*diag(S)*Z` reference built the "obvious slow way".
#
# Per the SAME row convention `winner_pair_cross_hessian_cm_block!` already establishes: row 1 =
# the "ones"/zeta-paired H column (S-only, no nu, no pi_vec correction -- it isn't part of
# `wctx.ncolI`'s own pi_vec-indexed family), rows 2:ncolI+1 = wctx's own `ncolI` economic columns
# (bilateral (slot,origin) pairs, PLUS the "cf"/gravity common-factor column at row `ncolI+1` when
# `wctx.has_cf` -- confirmed by cross-reading `cm_originzc_moments.jl::wrap_moments_with_originzc`:
# the "gravity" column there is `G_tmp[:,end]`, i.e. the SAME core-object last column
# `build_winner_pair_ctx`/`cf.cf_col` already treats as the "cf" common-factor column for H_EE --
# no separate gravity-specific logic is needed here, it is automatically covered by `wctx.has_cf`).
#
# UNLIKE the CM-grid cross primitive above, there is no `L`-threshold binning here at all (Z's
# columns are raw/continuous, not step functions of a bin index) -- so there is no separate
# "fill once, per-l block many times" split; ALL of `H_EZ` (every mean+pair column, across every
# level, concatenated in the SAME column order `wrap_moments_with_originzc`/
# `wrap_moments_with_cm_meanzc`'s own G-layout already uses) is filled in ONE call, O(W*Ddest*n_x)
# for the winner-conditioned bilateral rows (mirrors the CM-grid primitive's own per-slot loop
# structure, task brief candidate (b), with the loop nest ordered `slot -> x -> w` so both `Z[:,x]`
# and `winner[:,slot]` are read column-contiguous) plus O(W*n_x) BLAS gemv's for row 1 and the cf
# row (which are NOT winner-conditioned).
# ============================================================================

"""
    WinnerZCCrossScratch

Persistent scratch for `winner_pair_cross_hessian_zc_block!`: `Snu[w] = S[w]*nu[w]` (refreshed
once per Hessian callback via `winner_pair_cross_hessian_zc_prep!`, shared across every
mean/pair-level call within that same callback -- Snu does not depend on which level/column is
being filled), `crs_buf[w] = Snu[w]*wctx.cf_raw_scaled[w]` (only meaningful when `wctx.has_cf`,
likewise level-independent), `v` (length-`W` per-slot scratch, `v[w] = Snu[w]*y[w,slot]`), and
`NuZ_buf` (length->=`max_nx` scratch for `NuZ[x] = sum_w Snu[w]*Z[w,x]`, the pi_vec-correction
term shared by every bilateral/cf row of a single call). `max_nx` should be sized to the WIDEST
single `winner_pair_cross_hessian_zc_block!` call a caller will ever make (e.g. `n_mean+n_pair`
for a combined single-call fill, matching `build_cm_meanzc_bin_ctx`'s own "size scratch once,
reuse forever" discipline for `cross_scratch`/`core_ws`).
"""
mutable struct WinnerZCCrossScratch
    W::Int
    max_nx::Int
    Snu::Vector{Float64}
    crs_buf::Vector{Float64}
    v::Vector{Float64}
    NuZ_buf::Vector{Float64}
end

WinnerZCCrossScratch(W::Int, max_nx::Int) =
    WinnerZCCrossScratch(W, max_nx, zeros(W), zeros(W), zeros(W), zeros(max_nx))

"Rebuild (or reuse, if already the right size) `ws` for the current `(W, max_nx)` -- mirrors this file's own `ensure_winner_bin_cross_scratch!` idiom."
function ensure_winner_zc_cross_scratch!(ws_ref::Base.RefValue{Union{Nothing,WinnerZCCrossScratch}}, W::Int, max_nx::Int)
    ws = ws_ref[]
    if ws === nothing || ws.W != W || ws.max_nx < max_nx
        ws_ref[] = WinnerZCCrossScratch(W, max_nx)
    end
    return ws_ref[]
end

"""
    winner_pair_cross_hessian_zc_prep!(ws::WinnerZCCrossScratch, wctx::WinnerPairHessCtx, S::AbstractVector{Float64}) -> ws

Refresh `ws.Snu`/`ws.crs_buf` for the CURRENT Hessian callback's `S` (`= obj.arg2`, caller-supplied
-- unlike `winner_pair_cross_hessian_fill!`, this does NOT recompute `ddPsi!` itself, since both
wiring call sites already have a fresh `S`/`w` in hand by the time they reach the ZC cross block;
callers must ensure `S` reflects the current dual point). Call ONCE per Hessian callback, before
any `winner_pair_cross_hessian_zc_block!` call(s) for that callback.
"""
function winner_pair_cross_hessian_zc_prep!(ws::WinnerZCCrossScratch, wctx::WinnerPairHessCtx, S::AbstractVector{Float64})
    W = wctx.W
    length(S) == W || error("winner_pair_cross_hessian_zc_prep!: length(S)=$(length(S)) != wctx.W=$W")
    nu = wctx.nu
    Snu = ws.Snu
    @inbounds for w in 1:W
        Snu[w] = S[w] * nu[w]
    end
    if wctx.has_cf
        crs = wctx.cf_raw_scaled
        crsbuf = ws.crs_buf
        @inbounds for w in 1:W
            crsbuf[w] = Snu[w] * crs[w]
        end
    end
    return ws
end

"""
    winner_pair_cross_hessian_zc_block!(HEZ, wctx, ws, S, Z, M) -> HEZ

Fills `HEZ` (`(wctx.ncolI+1) x size(Z,2)`) = `(1/M) * E'*diag(S)*Z` for an arbitrary
ALREADY-CENTERED restriction feature matrix `Z` (`W x n_x`, e.g. the full concatenated
mean+pair block, or a single level's slice -- this function is level-agnostic, see the file
header). Requires `winner_pair_cross_hessian_zc_prep!(ws, wctx, S)` to have been called already
THIS Hessian callback (does not recompute `Snu`/`crs_buf` itself, so it is safe/cheap to call this
function more than once per callback against the SAME `ws`/`S` -- e.g. once per level -- without
redoing the O(W) prep work).
"""
function winner_pair_cross_hessian_zc_block!(HEZ::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerZCCrossScratch, S::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, M::Real)
    W = wctx.W; Ddest = wctx.Ddest
    ncolI = wctx.ncolI
    nx = size(Z, 2)
    size(HEZ) == (ncolI + 1, nx) || error("winner_pair_cross_hessian_zc_block!: size(HEZ)=$(size(HEZ)) != ($(ncolI + 1), $nx)")
    length(S) == W || error("winner_pair_cross_hessian_zc_block!: length(S)=$(length(S)) != wctx.W=$W")
    size(Z, 1) == W || error("winner_pair_cross_hessian_zc_block!: size(Z,1)=$(size(Z, 1)) != wctx.W=$W")
    nx <= ws.max_nx || error("winner_pair_cross_hessian_zc_block!: nx=$nx exceeds ws.max_nx=$(ws.max_nx) -- rebuild scratch via ensure_winner_zc_cross_scratch!")

    y = wctx.y; winner = wctx.winner; pi_vec = wctx.pi_vec
    has_cf = wctx.has_cf; jcf = ncolI
    Snu = ws.Snu; v = ws.v
    invM = 1.0 / M

    # row 1 ("ones"/zeta-paired H column): S-only weighted, no nu, no pi_vec correction -- plain gemv.
    row1 = @view HEZ[1, :]
    BLAS.gemv!('T', invM, Z, S, 0.0, row1)

    nbilateral = has_cf ? ncolI - 1 : ncolI
    bilateral_block = @view HEZ[2:1+nbilateral, :]
    fill!(bilateral_block, 0.0)

    # Winner-conditioned scatter accumulation, RAW (uncorrected) into HEZ[j+1,x] -- mirrors
    # winner_pair_cross_hessian_fill!'s own slot loop, generalized from bin-membership (Bidx) to a
    # direct continuous feature value Z[w,x]. Loop order slot -> x -> w keeps both Z[:,x] and
    # winner[:,slot] column-contiguous.
    @inbounds for slot in 1:Ddest
        for w in 1:W
            v[w] = Snu[w] * y[w, slot]
        end
        wcol = @view winner[:, slot]
        for x in 1:nx
            Zx = @view Z[:, x]
            for w in 1:W
                o = wcol[w]
                j = slot + (o - 1) * Ddest
                HEZ[j+1, x] += v[w] * Zx[w]
            end
        end
    end

    # NuZ[x] = sum_w Snu[w]*Z[w,x] -- the pi_vec-correction term shared by every bilateral/cf row
    # (same rank-1 structure winner_pair_cross_hessian_fill!'s own NuTab/NuCScum plays for the
    # CM-grid block, here un-binned since Z is continuous).
    NuZ = @view ws.NuZ_buf[1:nx]
    BLAS.gemv!('T', 1.0, Z, Snu, 0.0, NuZ)

    @inbounds for j in 1:nbilateral
        pij = pi_vec[j]
        for x in 1:nx
            HEZ[j+1, x] = invM * (HEZ[j+1, x] - pij * NuZ[x])
        end
    end

    if has_cf
        row_cf = @view HEZ[jcf+1, :]
        BLAS.gemv!('T', invM, Z, ws.crs_buf, 0.0, row_cf)
        pij = pi_vec[jcf]
        @inbounds for x in 1:nx
            row_cf[x] -= invM * pij * NuZ[x]
        end
    end

    return HEZ
end
