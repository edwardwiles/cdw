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
    Ddest::Int   # profiled economic block port (2026-08-01): needed to size MTab/MCScum/T0_slot below
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
    # Profiled economic block port (2026-08-01): the destination-specific target-correction total
    # T^R_{d,k} the mission spec requires, built from cf.wval (raw, winner-INDEPENDENT per-draw
    # destination value -- see PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md §1) rather than from
    # winner-conditioned QTab/QCScum, so it needs no separate "anchor still wins sometimes" side
    # table: wval is defined on every draw regardless of which origin wins, including an omitted
    # anchor. MTab[slot,x,k] = Sigma_{w: bin(U[w,x])=k} Snu[w]*wval[w,slot] (Ddest x D x (L+1),
    # filled in the SAME existing slot/w loop that fills QTab -- one extra multiply-add per (w,x),
    # no new O(W) pass), MCScum its (x,l)-cumulative twin (Ddest x D x L, same CS_x(j,l) prefix-sum
    # convention as QCScum/NuCScum). T0_slot[slot] = Sigma_w Snu[w]*wval[w,slot] (Ddest, the
    # UN-binned total, computed directly rather than read off MCScum[:, :, L] to avoid depending on
    # whether bin L is inclusive of every draw -- see design doc §1 for why this is computed
    # independently). Filled UNCONDITIONALLY (cheap, same complexity class as the existing QTab
    # fill) but only ever READ by the new `use_profiled_correction=true` path in
    # `winner_pair_cross_hessian_cm_block!`/`_colsum!`/`_esum!` -- every existing
    # `use_profiled_correction=false` (default) caller is unaffected, byte-for-byte.
    MTab::Array{Float64,3}     # Ddest x D x (L+1)
    MCScum::Array{Float64,3}   # Ddest x D x L
    T0_slot::Vector{Float64}   # Ddest
    # Profiled economic block port (2026-08-01): H_EF's colsum!/esum! analog of H_EC's
    # per-(o,refIndex1) MCScum DIFFERENCE -- colsum! needs a SUM over all D CM-grid origins x
    # (common-Fréchet's level restriction has no reference-contrast structure, see
    # winner_pair_cross_hessian_colsum!'s own docstring), so MSumX[d,l] = Σ_x MCScum[d,x,l]
    # (Ddest x L, computed once per callback, O(Ddest*D*L), reused by every l/j pair -- no new
    # O(W) pass). Only read by the new use_profiled_correction=true path.
    MSumX::Matrix{Float64}     # Ddest x L
    # optimize/structured-cross-hessian-ZC-CM-2026-07-28: persistent `Threads.@spawn` task buffer
    # for `winner_pair_cross_hessian_fill_threaded!` (threaded_cross_hessian.jl), sized to
    # `Threads.nthreads()` at construction so no `Vector{Task}` is allocated per Hessian callback.
    # Shared by BOTH raw-table passes in that function (origin-owned and slot-owned) -- neither
    # holds a reference into the other's iteration, so reusing the same buffer sequentially (fill
    # + fetch all, THEN reuse for the second pass) is safe.
    tasks_ec::Vector{Task}
    # Common-Fréchet winner-aware H_ER phase (2026-07-27), Part B: UN-binned (no threshold/bin
    # dimension) per-economic-column accumulator `EsumEcon[j] = sum_w Snu[w]*y[w,slot(j)]*
    # 1{winner(w,slot(j))=o(j)}` (length ncolI, `wctx`'s own 1:ncolI numbering, NOT NCORE-offset).
    # Needed ONLY by common-Fréchet's H_E,level block (`winner_pair_cross_hessian_esum!` below) --
    # flexible-CM's own H_EC/H_CC blocks have no un-binned economic-column-sum term, so this field
    # is unused (but harmlessly filled) for a plain flexible-CM caller of this same scratch struct.
    EsumEcon::Vector{Float64}  # ncolI
end

function WinnerBinCrossScratch(ncolI::Int, D::Int, L::Int, Ddest::Int = cld(ncolI, D))
    # BUGFIX (2026-07-28, root-caused independently by both the D=4 cm_meanzc gate and the D=20
    # flexible_cm/common_frechet profiling task): the last two positional args here were swapped
    # relative to the struct's OWN declared field order (`tasks_ec::Vector{Task}` THEN
    # `EsumEcon::Vector{Float64}`) -- passing `zeros(ncolI)::Vector{Float64}` into the
    # `tasks_ec::Vector{Task}` slot and `Vector{Task}(...)` into the `EsumEcon::Vector{Float64}`
    # slot. Julia's default memberwise constructor tries to `convert` each positional arg to its
    # field's declared type, so this raised `MethodError: Cannot convert Float64 to Task` (via
    # `unsafe_copyto!`) the FIRST time this constructor was ever reached with a genuinely fresh
    # `(ncolI,D,L)` (i.e. the first Hessian callback for a context whose `core_cf_ref[]` is an
    # actual `CompressedFactual`, engaging the `:winner_bin` H_EC cross-Hessian backend) -- a
    # real, pre-existing latent bug from when `tasks_ec` was inserted ahead of the pre-existing
    # `EsumEcon` field without updating this positional call, not something either task's own
    # profiling/gate edits caused. See `docs/ZC_CENTERING_LIFECYCLE_RELEASE_2026-07-28.md` and
    # `docs/FLEXCM_FRECHET_D20_PROFILE_AND_GATES_2026-07-28.md` for the two independent
    # reproductions (D=4 direct call outside KNITRO with a full stack trace; D=20 through the real
    # public driver with a `git stash` control run).
    return WinnerBinCrossScratch(ncolI, D, L, Ddest,
        zeros(ncolI, D, L + 1), zeros(D, L + 1), zeros(D, L + 1), zeros(D, L + 1),
        zeros(ncolI, D, L), zeros(D, L), zeros(D, L), zeros(D, L),
        zeros(Ddest, D, L + 1), zeros(Ddest, D, L), zeros(Ddest), zeros(Ddest, L),
        Vector{Task}(undef, Threads.nthreads()), zeros(ncolI))
end

"Rebuild (or reuse, if already the right size) `ws` for the current `(ncolI, D, L)` -- mirrors this codebase's own `resize_*_if_needed!` idiom. `Ddest` defaults to a size-only-correct-when-no-cf-column fallback for callers that don't yet pass it explicitly (harmless -- MTab/MCScum/T0_slot are unread on any path that doesn't opt into `use_profiled_correction=true`)."
function ensure_winner_bin_cross_scratch!(ws_ref::Base.RefValue{Union{Nothing,WinnerBinCrossScratch}}, ncolI::Int, D::Int, L::Int, Ddest::Int = cld(ncolI, D))
    ws = ws_ref[]
    if ws === nothing || ws.ncolI != ncolI || ws.D != D || ws.L != L || ws.Ddest != Ddest
        ws_ref[] = WinnerBinCrossScratch(ncolI, D, L, Ddest)
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
    nu = wctx.nu; y = wctx.y; winner = wctx.winner; wval = wctx.wval
    D = ws.D; L = ws.L; nbins = L + 1

    QTab = ws.QTab; NuTab = ws.NuTab; SOnlyTab = ws.SOnlyTab; QCfTab = ws.QCfTab; EsumEcon = ws.EsumEcon
    MTab = ws.MTab; T0_slot = ws.T0_slot
    fill!(QTab, 0.0); fill!(NuTab, 0.0); fill!(SOnlyTab, 0.0); fill!(QCfTab, 0.0); fill!(EsumEcon, 0.0)
    fill!(MTab, 0.0); fill!(T0_slot, 0.0)

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
            snu_w = S[w] * nu[w]
            snuy = snu_w * y[w, slot]
            # Common-Fréchet Part B: UN-binned accumulation (no x/Bidx loop) alongside the existing
            # per-bin QTab fill -- O(W*Ddest) additional work, negligible next to QTab's own
            # O(W*Ddest*D). See EsumEcon's own field docstring above.
            EsumEcon[j] += snuy
            # Profiled economic block port (2026-08-01): T0_slot/MTab, winner-UNCONDITIONAL (uses
            # wval, not y -- see MTab's own field docstring above), filled in this SAME slot/w loop
            # so no new O(W) pass is added.
            mv = snu_w * wval[w, slot]
            T0_slot[slot] += mv
            for x in 1:D
                QTab[j, x, Bidx[w, x]] += snuy
                MTab[slot, x, Bidx[w, x]] += mv
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
    MCScum = ws.MCScum
    @inbounds for slot in 1:Ddest, x in 1:D
        acc = 0.0
        for l in 1:L
            acc += MTab[slot, x, l]
            MCScum[slot, x, l] = acc
        end
    end
    # H_EF's colsum!/esum! sum-over-x analog of H_EC's per-x difference -- see MSumX's own field
    # docstring on WinnerBinCrossScratch. O(Ddest*D*L), cheap next to the O(W*Ddest*D) fill above.
    MSumX = ws.MSumX
    @inbounds for slot in 1:Ddest
        for l in 1:L
            acc = 0.0
            for x in 1:D
                acc += MCScum[slot, x, l]
            end
            MSumX[slot, l] = acc
        end
    end
    return ws
end

"""
    winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, ws, l, origins, refIndex1, M;
        use_profiled_correction=false) -> Hraw_EC

`use_profiled_correction=false` (default): byte-for-byte the original OLD-formulation behavior
(destination-independent `nu_diff` target correction, `Σ_w S_w R_k(w)` in
`PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md`'s notation) -- every existing caller (flexible CM,
common Fréchet, CM+ZC) is unaffected by this method's existence.

`use_profiled_correction=true`: replaces `nu_diff` with the destination-specific
`T_diff = MCScum[d,o,l] - MCScum[d,refIndex1,l]`, `d = wctx.target_slot[j]` -- the mission's
`T^R_{d,k} = Σ_o' W^R_{o'd,k}` (design doc §2), correct for a reduced/anchor-omitting economic
layout because `MCScum` is built from `wctx.wval` (always defined, including on draws where an
omitted anchor wins), not from the winner-conditioned `QCScum` buckets. The France/cf row is
DELIBERATELY EXCLUDED from this switch (kept on `nu_diff` even when `use_profiled_correction=true`)
-- `wctx.target_slot[cf.cf_col]` is a sentinel `0` because the true France destination slot needs
`ctx.bi`, unavailable to `build_winner_pair_ctx(cf::CompressedFactual)`; see design doc §2 and
`core_exact_hessian.jl`'s own `target_slot` field docstring. This is a documented scope boundary,
not a silent gap: requesting the profiled correction with `wctx.has_cf=true` still produces a
CORRECT (if not-yet-profiled) France row, it does not produce a wrong one.
"""
function winner_pair_cross_hessian_cm_block!(Hraw_EC::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerBinCrossScratch, l::Int, origins::Vector{Int}, refIndex1::Int, M;
        use_profiled_correction::Bool = false)
    QCScum = ws.QCScum; NuCScum = ws.NuCScum; SOnlyCScum = ws.SOnlyCScum; QCfCScum = ws.QCfCScum
    MCScum = ws.MCScum
    pi_vec = wctx.pi_vec
    target_slot = wctx.target_slot
    invM = 1.0 / M
    has_cf = wctx.has_cf
    jcf = wctx.ncolI   # the cf column's index WITHIN wctx's own 1:ncolI numbering (cf.cf_col)
    @inbounds for (oi, o) in enumerate(origins)
        Hraw_EC[1, oi] = (SOnlyCScum[o, l] - SOnlyCScum[refIndex1, l]) * invM
        nu_diff = NuCScum[o, l] - NuCScum[refIndex1, l]
        for j in 1:wctx.ncolI
            q_diff = QCScum[j, o, l] - QCScum[j, refIndex1, l]
            corr = if use_profiled_correction
                d = target_slot[j]
                MCScum[d, o, l] - MCScum[d, refIndex1, l]
            else
                nu_diff
            end
            Hraw_EC[j + 1, oi] = (q_diff - pi_vec[j] * corr) * invM
        end
        # The "cf"/common-factor column (if present) is NOT winner-conditioned like the regular
        # economic columns above -- QTab[jcf,:,:] was left at zero by the main slot-loop (no
        # sample's `winner[w,slot]` ever equals it, it isn't a bilateral (slot,origin) pair at
        # all), so overwrite that one row here with its own dedicated accumulation. Always uses
        # nu_diff (see docstring: France row profiling is an explicit follow-on, not done here).
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
    Ddest::Int   # profiled economic block port (2026-08-01): needed to size SnuWval/TZ_buf below
    Snu::Vector{Float64}
    crs_buf::Vector{Float64}
    v::Vector{Float64}
    NuZ_buf::Vector{Float64}
    # Profiled economic block port (2026-08-01): the H_EZ analog of H_EC's MCScum
    # (PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md §3) -- `SnuWval[w,d] = Snu[w]*wctx.wval[w,d]`
    # (W x Ddest, filled once per callback in `winner_pair_cross_hessian_zc_prep!`, same lifecycle
    # as `Snu`) and `TZ_buf[d,x] = Σ_w SnuWval[w,d]*Z[w,x]` (Ddest x max_nx, ONE BLAS.gemm! call
    # per `winner_pair_cross_hessian_zc_block!` invocation, replacing the old single-vector `NuZ`
    # correction with a per-destination one). Filled/read only by the new
    # `use_profiled_correction=true` path -- unread by every existing caller.
    SnuWval::Matrix{Float64}   # W x Ddest
    TZ_buf::Matrix{Float64}    # Ddest x max_nx
    # optimize/structured-cross-hessian-ZC-CM-2026-07-28: persistent scratch for
    # `winner_pair_cross_hessian_zc_block_threaded!` (threaded_cross_hessian.jl) -- `tasks_ez`
    # (sized to `Threads.nthreads()`) avoids a per-callback `Vector{Task}` allocation;
    # `thread_scratch_ez[k]` is a dedicated length-`W` buffer for worker-slot `k`'s `v[w] =
    # Snu[w]*y[w,slot]` scratch (one per worker slot, not one per `slot` value, so at most
    # `Threads.nthreads()` buffers regardless of `Ddest`) -- avoids the serial version's single
    # shared `ws.v` field, which would race if reused directly across concurrent worker tasks.
    tasks_ez::Vector{Task}
    thread_scratch_ez::Vector{Vector{Float64}}
end

"`Ddest` defaults to 1 (a deliberately conservative, deliberately-too-small-to-silently-misuse
default) for callers that don't pass it explicitly -- SnuWval/TZ_buf are unread on any path that
doesn't opt into `use_profiled_correction=true`, so no existing caller needs to change."
WinnerZCCrossScratch(W::Int, max_nx::Int, Ddest::Int = 1) =
    WinnerZCCrossScratch(W, max_nx, Ddest, zeros(W), zeros(W), zeros(W), zeros(max_nx),
        zeros(W, Ddest), zeros(Ddest, max_nx),
        Vector{Task}(undef, Threads.nthreads()),
        [zeros(W) for _ in 1:Threads.nthreads()])

"Rebuild (or reuse, if already the right size) `ws` for the current `(W, max_nx)` -- mirrors this file's own `ensure_winner_bin_cross_scratch!` idiom."
function ensure_winner_zc_cross_scratch!(ws_ref::Base.RefValue{Union{Nothing,WinnerZCCrossScratch}}, W::Int, max_nx::Int, Ddest::Int = 1)
    ws = ws_ref[]
    if ws === nothing || ws.W != W || ws.max_nx < max_nx || ws.Ddest != Ddest
        ws_ref[] = WinnerZCCrossScratch(W, max_nx, Ddest)
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
    # Profiled economic block port (2026-08-01): SnuWval, same lifecycle as Snu -- see
    # WinnerZCCrossScratch's own field docstring. Cheap/harmless when ws.Ddest==1 (the
    # not-yet-profiled-caller default, still correct-if-unused since it's simply never read);
    # ws.Ddest must never exceed wctx.Ddest (indexes into wctx.wval's own Ddest dimension below).
    Ddest_ws = ws.Ddest
    Ddest_ws <= wctx.Ddest || error("winner_pair_cross_hessian_zc_prep!: ws.Ddest=$Ddest_ws exceeds wctx.Ddest=$(wctx.Ddest) -- rebuild scratch via ensure_winner_zc_cross_scratch! with the correct Ddest")
    wval = wctx.wval
    SnuWval = ws.SnuWval
    @inbounds for d in 1:Ddest_ws, w in 1:W
        SnuWval[w, d] = Snu[w] * wval[w, d]
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

`use_profiled_correction=false` (default): byte-for-byte the original OLD-formulation behavior
(the single global `NuZ` correction) -- every existing caller (CM+ZC, ZC-only) is unaffected.
`use_profiled_correction=true`: replaces `NuZ[x]` (destination-independent) with the
destination-specific `TZ[d,x] = Σ_w Snu[w]*wctx.wval[w,d]*Z[w,x]`, `d = wctx.target_slot[j]`,
computed as ONE `BLAS.gemm!` call (`TZ = SnuWval' * Z`) -- see
PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md §3. Same France/cf-row scope boundary as
`winner_pair_cross_hessian_cm_block!` (kept on `NuZ`, see that function's own docstring).
"""
function winner_pair_cross_hessian_zc_block!(HEZ::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerZCCrossScratch, S::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, M::Real;
        use_profiled_correction::Bool = false)
    W = wctx.W; Ddest = wctx.Ddest
    ncolI = wctx.ncolI
    nx = size(Z, 2)
    size(HEZ) == (ncolI + 1, nx) || error("winner_pair_cross_hessian_zc_block!: size(HEZ)=$(size(HEZ)) != ($(ncolI + 1), $nx)")
    length(S) == W || error("winner_pair_cross_hessian_zc_block!: length(S)=$(length(S)) != wctx.W=$W")
    size(Z, 1) == W || error("winner_pair_cross_hessian_zc_block!: size(Z,1)=$(size(Z, 1)) != wctx.W=$W")
    nx <= ws.max_nx || error("winner_pair_cross_hessian_zc_block!: nx=$nx exceeds ws.max_nx=$(ws.max_nx) -- rebuild scratch via ensure_winner_zc_cross_scratch!")
    use_profiled_correction && ws.Ddest != Ddest &&
        error("winner_pair_cross_hessian_zc_block!: use_profiled_correction=true requires ws.Ddest=$(ws.Ddest) == wctx.Ddest=$Ddest -- rebuild scratch via ensure_winner_zc_cross_scratch!(...; Ddest)")

    y = wctx.y; winner = wctx.winner; pi_vec = wctx.pi_vec; target_slot = wctx.target_slot
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

    # Profiled economic block port (2026-08-01): TZ[d,x] = Σ_w SnuWval[w,d]*Z[w,x], ONE BLAS.gemm!
    # call, only computed when actually needed (use_profiled_correction=true) -- see
    # PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md §3.
    TZ = nothing
    if use_profiled_correction
        TZ = @view ws.TZ_buf[:, 1:nx]
        BLAS.gemm!('T', 'N', 1.0, ws.SnuWval, Z, 0.0, TZ)
    end

    @inbounds for j in 1:nbilateral
        pij = pi_vec[j]
        if use_profiled_correction
            d = target_slot[j]
            for x in 1:nx
                HEZ[j+1, x] = invM * (HEZ[j+1, x] - pij * TZ[d, x])
            end
        else
            for x in 1:nx
                HEZ[j+1, x] = invM * (HEZ[j+1, x] - pij * NuZ[x])
            end
        end
    end

    if has_cf
        row_cf = @view HEZ[jcf+1, :]
        BLAS.gemv!('T', invM, Z, ws.crs_buf, 0.0, row_cf)
        pij = pi_vec[jcf]
        # France/cf row deliberately excluded from the profiled path -- always NuZ, see docstring.
        @inbounds for x in 1:nx
            row_cf[x] -= invM * pij * NuZ[x]
        end
    end

    return HEZ
end

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
    winner_pair_cross_hessian_colsum!(colsum, wctx, ws, l; use_profiled_correction=false) -> colsum

`colsum[j] = sum_{o=1}^D CS_[o,j,l]` for `j = 1:NCORE` (`NCORE = wctx.ncolI + 1`, SAME row
convention as `winner_pair_cross_hessian_cm_block!`: row 1 = the ones/zeta column, S-only; rows
2:NCORE = the `ncolI` economic-lambda columns, row `j+1` <-> `wctx`'s own column `j`) -- the
SUM-over-all-origins counterpart to that function's own (o,ref)-DIFFERENCE, needed by common-
Fréchet's level restriction (a sum over all D origins with weight `1/sqrt(D)`, not a CM-style
difference against `refIndex1`). `O(D)` per row, `O(D*NCORE)` total per `l` -- same complexity class
as the dense `for o in 1:D; acc += CS_[o,j,l]; end` loop this replaces. Caller must call
`winner_pair_cross_hessian_fill!` once per Hessian callback first (same precondition as
`winner_pair_cross_hessian_cm_block!`).

`use_profiled_correction=false` (default): byte-for-byte the original behavior (single global
`sumNu` correction, same for every economic row `j`). `use_profiled_correction=true`: replaces
`sumNu` with the destination-specific `ws.MSumX[wctx.target_slot[j], l]` (see that field's own
docstring on `WinnerBinCrossScratch`) -- the mission's `T^F_{d,k}` for H_EF, per
PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md §4. Same France/cf-row scope boundary as H_EC/H_EZ
(kept on `sumNu`).
"""
function winner_pair_cross_hessian_colsum!(colsum::AbstractVector{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerBinCrossScratch, l::Int; use_profiled_correction::Bool = false)
    QCScum = ws.QCScum; NuCScum = ws.NuCScum; SOnlyCScum = ws.SOnlyCScum; QCfCScum = ws.QCfCScum
    MSumX = ws.MSumX
    pi_vec = wctx.pi_vec
    target_slot = wctx.target_slot
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
        corr = use_profiled_correction ? MSumX[target_slot[j], l] : sumNu
        colsum[j + 1] = sumQ - pi_vec[j] * corr
    end

    # Same cf-column override as winner_pair_cross_hessian_cm_block! -- QCScum[jcf,:,:] was left at
    # zero by the main slot-loop (the cf column is not a (slot,origin) pair), so overwrite that one
    # entry with its own dedicated accumulation. Always sumNu (France row excluded, see docstring).
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
    winner_pair_cross_hessian_esum!(Esum, wctx, ws, w, Wtot; use_profiled_correction=false) -> Esum

`use_profiled_correction=false` (default): byte-for-byte the original behavior (single global `t0`
correction). `use_profiled_correction=true`: replaces `t0` with the destination-specific
`ws.T0_slot[wctx.target_slot[j]]` -- the UN-BINNED counterpart of `winner_pair_cross_hessian_colsum!`'s
`MSumX`, already built by `winner_pair_cross_hessian_fill!` (see `T0_slot`'s own field docstring).
Same France/cf-row scope boundary as the other amended cross-block functions.

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
        ws::WinnerBinCrossScratch, w::AbstractVector{Float64}, Wtot::Float64; use_profiled_correction::Bool = false)
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
    target_slot = wctx.target_slot
    T0_slot = ws.T0_slot
    EsumEcon = ws.EsumEcon
    @inbounds for j in 1:wctx.ncolI
        corr = use_profiled_correction ? T0_slot[target_slot[j]] : t0
        Esum[j + 1] = EsumEcon[j] - pi_vec[j] * corr
    end
    if has_cf
        jcf = wctx.ncolI
        Esum[jcf + 1] = ecf - pi_vec[jcf] * t0
    end
    return Esum
end

# ============================================================================
# CM+ZC E/C/Z block-partition + H_CZ release (2026-07-27): H_CZ = C'SZ, the CM-grid (common-
# marginals bin/threshold restriction, `C`) x mean/pairwise-ZC-restriction (`Z`) cross Hessian.
# Bin-index-keyed analogue of `winner_pair_cross_hessian_cm_block!` above, with the ZC-restriction's
# ALREADY-CENTERED, ALREADY-S-weighted `ZcS` (`zc_restriction_operator.jl::ZCCenteredScratch`,
# built directly from the family's own raw `Zraw_all`/`Zpairraw_all` feature matrices, never from
# `obj.H`) taking the role the winner-selected economic column `Q`/`QTab` plays in that function --
# there is no winner-selection here at all (`Z`'s columns are plain per-draw feature values, not a
# winner-argmin outcome), only a bin-membership test, so this is considerably simpler: a single
# `SBinTab`-style accumulation, no separate `pi_vec`/nu-power correction term (that correction was
# needed there because `E`'s own columns are `nu*(y*1{winner}-pi)`-shaped; `ZcS` here is already the
# raw, already-centered, already-weighted quantity to sum).
#
# CM+ZC's H_EC/H_CZ split (`cm_hessian_architectures.jl::hessian_cm_structured!`): once `E`'s
# widened NCORE columns are split into the TRUE economic sub-block (`1:ncore_core`, filled via the
# EXISTING `winner_pair_cross_hessian_cm_block!`, using `wctx` which is ALREADY exactly
# `ncore_core`-wide, unaffected by CM+ZC's Z-widening) and the Z sub-block (`ncore_core+1:NCORE`,
# filled via THIS new primitive), the two together reconstruct the FULL widened `H_EC` block the
# dense `CS_`-table path used to build in one pass -- WITHOUT ever reading the dense economic (E)
# columns of `obj.H`, satisfying this phase's own "H_CZ ... NOT via dense CM columns read from
# obj.H" requirement (the CM-grid restriction `C` itself is reconstructed from `Bidx`/bin
# membership directly, exactly as the dense path already did, never from a materialized `obj.H`
# CM-grid block, which under `:cm_lookup`/`:operator` inner FG backends may not even be filled).
# ============================================================================

"""
    BinZCrossScratch

Persistent scratch for `bin_zc_cross_hessian_fill!`/`_block!`: `ZBinTab[x,j,k] =
Σ_{w: bin(U[w,x])=k} ZcS[w,j]` (`D x nz x (L+1)`, `nz = n_restriction(op)`), cumulative
`ZBinCScum[x,j,l] = Σ_{k<=l} ZBinTab[x,j,k]` (`D x nz x L`) -- same `CS_x(j,l)` prefix-sum
convention `cm_hessian_architectures.jl`'s own `CScum`/this file's own `QCScum` already use.
"""
mutable struct BinZCrossScratch
    D::Int
    L::Int
    nz::Int
    ZBinTab::Array{Float64,3}
    ZBinCScum::Array{Float64,3}
    # optimize/structured-cross-hessian-ZC-CM-2026-07-28: persistent `Threads.@spawn` task buffer
    # for `bin_zc_cross_hessian_fill_threaded!` (threaded_cross_hessian.jl), sized to
    # `Threads.nthreads()` at construction -- no per-callback `Vector{Task}` allocation.
    tasks_cz::Vector{Task}
end
BinZCrossScratch(D::Int, L::Int, nz::Int) =
    BinZCrossScratch(D, L, nz, zeros(D, nz, L + 1), zeros(D, nz, L), Vector{Task}(undef, Threads.nthreads()))

"Rebuild (or reuse, if already the right size) `ws` for the current `(D,L,nz)` -- mirrors this file's own `ensure_*_scratch!` idiom."
function ensure_bin_zc_cross_scratch!(ws::Union{Nothing,BinZCrossScratch}, D::Int, L::Int, nz::Int)
    if ws === nothing || ws.D != D || ws.L != L || ws.nz != nz
        return BinZCrossScratch(D, L, nz)
    end
    return ws
end

"""
    bin_zc_cross_hessian_fill!(ws::BinZCrossScratch, Bidx, ZcS) -> ws

Fills `ws.ZBinTab`/`ws.ZBinCScum` from the CURRENT Hessian callback's `ZcS`
(`zc_restriction_operator.jl::ZCCenteredScratch`'s own `ZcS` view, `W x nz`, already
`S`-weighted and already target-centered -- built by `refresh_zc_centered!`, called once per
callback before this function). `Bidx` is the SAME `W x D` bin-index matrix `cctx.Bidx` already
carries (theta-independent, precomputed once per campaign).
"""
function bin_zc_cross_hessian_fill!(ws::BinZCrossScratch, Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64})
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")
    ZBinTab = ws.ZBinTab
    fill!(ZBinTab, 0.0)
    @inbounds for w in 1:W
        for x in 1:D
            b = Bidx[w, x]
            for j in 1:nz
                ZBinTab[x, j, b] += ZcS[w, j]
            end
        end
    end
    ZBinCScum = ws.ZBinCScum
    @inbounds for x in 1:D, j in 1:nz
        acc = 0.0
        for l in 1:L
            acc += ZBinTab[x, j, l]
            ZBinCScum[x, j, l] = acc
        end
    end
    return ws
end

"""
    bin_zc_cross_hessian_block!(HCZ, ws::BinZCrossScratch, l, origins, refIndex1, M) -> HCZ

Per-threshold-block (`l`) raw `H_CZ` slab, `nz x nO` (`nz = ws.nz`, `nO = length(origins)`) --
same `(o,ref)`-DIFFERENCE convention `winner_pair_cross_hessian_cm_block!` already establishes for
`H_EC`'s own raw block, so this drops into `hessian_cm_structured!`'s SAME per-`l` loop as a
direct sibling call (see that file's own `Hraw_EC`/`block_ec` handling -- H_CZ gets the SAME
optional `R`-congruence treatment before being written into `Hfull`). Caller must call
`bin_zc_cross_hessian_fill!` ONCE per Hessian callback first.
"""
function bin_zc_cross_hessian_block!(HCZ::AbstractMatrix{Float64}, ws::BinZCrossScratch,
        l::Int, origins::Vector{Int}, refIndex1::Int, M)
    size(HCZ) == (ws.nz, length(origins)) || error("bin_zc_cross_hessian_block!: size(HCZ)=$(size(HCZ)) != ($(ws.nz), $(length(origins)))")
    ZBinCScum = ws.ZBinCScum
    invM = 1.0 / M
    @inbounds for (oi, o) in enumerate(origins)
        for j in 1:ws.nz
            HCZ[j, oi] = invM * (ZBinCScum[o, j, l] - ZBinCScum[refIndex1, j, l])
        end
    end
    return HCZ
end
