# D=20 profiling task (flexible_cm/common_frechet, 2026-07-28): self-include the opt-in
# `@cmhess_prof` sub-block timing macro's defining file if not already loaded -- see the identical
# guard/rationale in cm_hessian_architectures.jl (this file is used together with that one in every
# existing caller, but this guard is repeated here defensively/idempotently).
isdefined(Main, :CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED) || include(joinpath(@__DIR__, "cm_hessian_subblock_profiling.jl"))

# Continuation 14 (integration/fullA-cm-parallel-production), Task 2: combined CM-Hessian
# benchmark support. PORTED (not re-invented) from diag/fullA-inner-blas-threading
# (`git show ecc820d:full_aod_diag/d4_exact/cm_hessian_threaded.jl`), which built and validated
# (to ~1.6-1.8e-14 vs serial, `c13_validate_threaded_archC.jl`) draw-chunk Julia-thread-parallel
# weighted-bin-contingency accumulation for the Architecture-C structured CM Hessian. That branch's
# own report (docs/fullA_inner_blas_threading_report.md sec 12) found only 1.04x END-TO-END on an
# EASY ~2s cold calibration-adjacent solve and explicitly did NOT merge it as a hard default. This
# session's job (per the plan) is to re-test the SAME validated component on a genuinely HARD
# 20-35s / 7-10-Hessian-callback CM point (see c14_find_hard_cm_point.jl), where the Hessian
# callback fires many more times relative to the one-time moment build -- a scenario explicitly
# flagged as untested there.
#
# Changes made during this port (disclosed, not silent):
#   1. BUG FIX: the source branch's `archC_hess_cb_builder_threaded` never incremented
#      `_INNER_CALL_COUNTERS[].n_hess_calls` (unlike the serial `archC_hess_cb_builder` in
#      cm_hessian_architectures.jl, which does). Harmless for that branch's own report (it never
#      relied on the counter), but this session's benchmark needs accurate n_hess_calls for every
#      cell -- fixed here.
#   2. `BLAS.syrk!` for the H_EE block (brief task 2d): H_EE = (1/M) Ews' * Ews is a genuine
#      Gram matrix (Ews = E .* sqrt.(w)) -- symmetric by construction, mathematically valid for
#      syrk. The ORIGINAL `hessian_cm_structured!` (cm_hessian_architectures.jl, production,
#      untouched by this file) still uses `BLAS.gemm!`; this file's variants use `BLAS.syrk!`
#      instead (~1.4-1.8x faster per the source branch's own isolated-kernel measurement, sec 1).
#   3. Unified into ONE hessian-construction function parameterized by (threaded_bins::Bool,
#      use_syrk::Bool) rather than two near-duplicate copies (serial vs threaded), to cut the risk
#      of the two copies silently drifting -- the H_EC/H_CC/packing tail (unchanged from the
#      original) is written ONCE, not twice.
#   4. No atomics anywhere (verified: the source branch's own header already established this
#      invariant via a fixed-thread-index-order reduction after `Threads.@threads :static`; kept
#      unchanged here).
#
# Everything else (deterministic tree/fixed-order reduction of thread-local tables, no atomics,
# preallocated per-thread scratch, static contiguous draw-chunk partitioning) is UNCHANGED from the
# source branch -- see its own header comment, reproduced in spirit above.

"Per-thread scratch mirroring CMBinHessCtx's Ttab/Stab shapes, one set per Julia thread. Ported unchanged from diag/fullA-inner-blas-threading."
struct ThreadLocalBinScratch
    Ttab::Vector{Array{Float64,4}}   # [tid] -> D x D x (L+1) x (L+1)
    Stab::Vector{Array{Float64,3}}   # [tid] -> D x NCORE x (L+1)
end

function build_thread_local_scratch(cctx)
    nt = Threads.nthreads()
    D = cctx.D; NCORE = cctx.NCORE; L1 = cctx.L + 1
    Ttab = [zeros(D, D, L1, L1) for _ in 1:nt]
    Stab = [zeros(D, NCORE, L1) for _ in 1:nt]
    return ThreadLocalBinScratch(Ttab, Stab)
end

"""
Threaded drop-in replacement for build_bin_tables! (cm_hessian_architectures.jl). Writes the SAME
cctx.Ttab/cctx.Stab as the serial version, via a deterministic static-chunk-then-fixed-order
reduction (no atomics; ported unchanged from diag/fullA-inner-blas-threading).

`fill_S=false` (winner-aware H_ER phase, 2026-07-27) skips the `Sloc[x,j,bx] += ws*E[s,j]` inner
loop -- the only per-thread read of `E` -- mirroring the serial `build_bin_tables!`'s own `fill_S`
kwarg exactly, so the production default (`use_threaded_bins=true`) gets the SAME
no-dense-economic-column-read property when the `:winner_bin` cross-Hessian backend is active.
"""
function build_bin_tables_threaded!(cctx, tls::ThreadLocalBinScratch, H::Union{Nothing,AbstractMatrix{Float64}}, w::AbstractVector{Float64}; fill_S::Bool = true)
    D = cctx.D; NCORE = cctx.NCORE; Bidx = cctx.Bidx
    W = length(w)
    nt = Threads.nthreads()

    for t in 1:nt
        fill!(tls.Ttab[t], 0.0)
        fill!(tls.Stab[t], 0.0)
    end

    if fill_S
        # No-moments/no-composite-G task (2026-07-28): `E` constructed lazily, only here.
        # True no-H operator bundle (2026-07-28 continuation): was `H::AbstractMatrix{Float64}` --
        # rejected `nothing` at the METHOD SIGNATURE level (a MethodError, before this function body
        # ever ran) even though `E`/`H` is never read when `fill_S=false` -- this is the actual gap
        # `build_bin_tables!` (serial, cm_hessian_architectures.jl) already closed but this threaded
        # sibling did not. Mirrors that function's explicit fail-fast guard.
        H === nothing && error("build_bin_tables_threaded!: fill_S=true requested for an operator-mode bundle with no H field -- should be unreachable in production.")
        E = @view H[:, 2:1+NCORE]
        Threads.@threads :static for tid in 1:nt
            lo = 1 + div((tid - 1) * W, nt)
            hi = div(tid * W, nt)
            Tloc = tls.Ttab[tid]; Sloc = tls.Stab[tid]
            @inbounds for s in lo:hi
                ws = w[s]
                for x in 1:D
                    bx = Bidx[s, x]
                    for j in 1:NCORE
                        Sloc[x, j, bx] += ws * E[s, j]
                    end
                end
                for x in 1:D
                    bx = Bidx[s, x]
                    for y in 1:D
                        by = Bidx[s, y]
                        Tloc[x, y, bx, by] += ws
                    end
                end
            end
        end
    else
        Threads.@threads :static for tid in 1:nt
            lo = 1 + div((tid - 1) * W, nt)
            hi = div(tid * W, nt)
            Tloc = tls.Ttab[tid]
            @inbounds for s in lo:hi
                ws = w[s]
                for x in 1:D
                    bx = Bidx[s, x]
                    for y in 1:D
                        by = Bidx[s, y]
                        Tloc[x, y, bx, by] += ws
                    end
                end
            end
        end
    end

    T = cctx.Ttab; S = cctx.Stab
    fill!(T, 0.0); fill!(S, 0.0)
    for tid in 1:nt   # fixed order 1:nt (not completion order) -> deterministic
        T .+= tls.Ttab[tid]
        fill_S && (S .+= tls.Stab[tid])
    end
    return nothing
end

"Threaded drop-in replacement for prefix_sum_tables!. Embarrassingly parallel over the D*D (x,y)
pairs -- no reduction needed. Ported unchanged from diag/fullA-inner-blas-threading. `fill_S=false`
(winner-aware H_ER phase) skips the CScum prefix-sum, mirroring the serial version's own kwarg."
function prefix_sum_tables_threaded!(cctx; fill_S::Bool = true)
    D = cctx.D; L = cctx.L; NCORE = cctx.NCORE
    T = cctx.Ttab; CT = cctx.CT
    pairs = [(x, y) for x in 1:D for y in 1:D]
    Threads.@threads :static for k in 1:length(pairs)
        (x, y) = pairs[k]
        @inbounds for l in 1:L
            for lp in 1:L
                v = T[x, y, l, lp]
                v += (l > 1 ? CT[x, y, l-1, lp] : 0.0)
                v += (lp > 1 ? CT[x, y, l, lp-1] : 0.0)
                v -= (l > 1 && lp > 1) ? CT[x, y, l-1, lp-1] : 0.0
                CT[x, y, l, lp] = v
            end
        end
    end
    if fill_S
        S = cctx.Stab; CS_ = cctx.CScum
        @inbounds for x in 1:D, j in 1:NCORE
            acc = 0.0
            for l in 1:L
                acc += S[x, j, l]
                CS_[x, j, l] = acc
            end
        end
    end
    return nothing
end

"""
    hessian_cm_structured_v2!(h, obj, cctx; threaded_bins=false, tls=nothing, use_syrk=true)

Unified Architecture-C Hessian construction (see file header, port-change 3): builds the SAME
packed upper-triangular Hessian as `cm_hessian_architectures.jl::hessian_cm_structured!`
(production, untouched), choosing serial vs. Julia-thread-parallel bin-table construction via
`threaded_bins`/`tls`, and gemm vs. syrk for the H_EE block via `use_syrk`. The H_EC/H_CC/packing
tail is copied verbatim from the original (not re-derived) to minimize the chance of a second
divergent bug site.
"""
function hessian_cm_structured_v2!(h, obj, cctx, extension::Any = nothing; threaded_bins::Bool = false,
                                    tls::Union{Nothing,ThreadLocalBinScratch} = nothing,
                                    use_syrk::Bool = true)
    # Harmonization task (2026-07-28): `extension` is `nothing` for flexible CM/CM+ZC (unchanged)
    # or a `CMFrechetExtension` for common Fréchet -- same as the serial hessian_cm_structured!
    # (cm_hessian_architectures.jl); see that function's own comment for the `Any`-typing rationale.
    @unpack M, arg0, arg2, ddPsi! = obj
    H = _dense_H_or_nothing(obj)
    @cmhess_prof "ddpsi" ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins

    # No-moments/no-composite-G task (2026-07-28): `E` no longer constructed eagerly -- see the
    # identical change/rationale in cm_hessian_architectures.jl::hessian_cm_structured!.
    Hfull = cctx.Hfull
    @cmhess_prof "misc_bookkeeping" fill!(Hfull, 0.0)
    cf = cctx.core_cf_ref[]

    # ---- H_EE: shared exact winner-pair backend (port/shared-winner-pair-core-hessian-
    # production-2026-07-25), same `_fill_cm_HEE!` helper the serial Architecture C callback uses
    # (cm_hessian_architectures.jl) -- task §4.2's "Do not implement a second winner-pair variant
    # for this family". 2026-07-25 continuation fix: this used to have its OWN duplicate
    # winner-pair-vs-dense branch here (checking `cctx.core_cf_ref[] !== nothing` -- stale even on
    # its own terms post-Symbol-fallback-reasons -- with its own inline syrk/gemm dense fallback
    # that never called `record_core_hessian_call!`), so this path's dense-vs-winner-pair choice
    # was invisible to the runtime backend-use counters (task §2) and diverged from
    # `_fill_cm_HEE!`'s own logic. Now calls `_fill_cm_HEE!` unconditionally -- ONE decision point,
    # ONE counter-recording site, for both the serial and threaded Architecture-C callers.
    # `use_syrk` no longer has an effect (the shared dense fallback inside `_fill_cm_HEE!` always
    # uses `gemm!`) -- kept as a no-op parameter rather than a breaking signature change for
    # existing callers.
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    @cmhess_prof "H_EE" _fill_cm_HEE!(HEE, w, obj, cctx, H, M)   # may rebuild cctx.core_ws/core_ws_for for this cf

    # Performance closeout task (2026-08-02), Section 8: this threaded twin previously had NO
    # `cctx.profiled_layout` branch at all (confirmed by grep -- zero references before this edit),
    # so any profiled/reduced cctx would silently fall into the non-profiled `use_winner_bin` logic
    # below and either error (wctx = serial_ctx(cctx.core_ws) with core_ws never built for a
    # profiled cctx) or use the wrong (full, non-reduced) NCORE width. This is why every profiled/
    # reduced cctx in this codebase has had to pass `threaded_bins=false` -- there was no threaded
    # code path that understood the reduced layout at all, not merely an unthreaded one.
    #
    # Fix: port the serial `hessian_cm_structured!`'s own profiled branch
    # (cm_hessian_architectures.jl) into this function VERBATIM (same H_EC/H_CZ gather logic, same
    # `use_profiled_correction=true` calls, same `hcz_prep_dispatch!` call this task's Section 7
    # already wired into the serial branch) -- the ONLY change from that serial branch is the
    # bin-table prep call, which now reuses the SAME threaded_bins-conditional this function
    # already applies to the non-profiled branch a few lines below (`build_bin_tables_threaded!`/
    # `prefix_sum_tables_threaded!`, both pre-existing, already-validated kernels -- ported unchanged
    # from diag/fullA-inner-blas-threading per this file's own header). No new kernel, no new
    # orchestration primitive: this is the same dispatch-by-`threaded_bins` pattern used everywhere
    # else in this function, applied to the one branch that was missing it.
    if cctx.profiled_layout !== nothing
        local bin_zc_ws_ec
        if cctx.ncore_core < NCORE
            nz_ec = n_restriction(cctx.hzz_zc_op)
            bin_zc_ws_ec = ensure_bin_zc_cross_scratch!(cctx.bin_zc_cross, D, L, nz_ec)
            cctx.bin_zc_cross = bin_zc_ws_ec
            @cmhess_prof "H_CZ_prep" hcz_prep_dispatch!(bin_zc_ws_ec, cctx.hcz_prep_backend, cctx.Bidx, cctx.hzz_centered.ZcS, cctx;
                workers = cctx.cross_hessian_workers, threaded = cctx.cross_hessian_threaded)
        end
        @cmhess_prof "bintables_prep" if threaded_bins
            tls === nothing && error("hessian_cm_structured_v2!(threaded_bins=true) requires tls (build_thread_local_scratch(cctx))")
            build_bin_tables_threaded!(cctx, tls, H, w; fill_S = false)
            prefix_sum_tables_threaded!(cctx; fill_S = false)
        else
            build_bin_tables!(cctx, H, w; fill_S = false)
            prefix_sum_tables!(cctx; fill_S = false)
        end
        layout = cctx.profiled_layout
        has_france = layout.france_ratio_reduced_j > 0
        bi_slot = has_france ? dest_slot(cctx.econ_ctx, cctx.econ_ctx.bi) : 0
        gpσ = has_france ? cctx.profiled_theta_ref[][3 + D]^cctx.profiled_theta_ref[][2] : 0.0
        denom_cf = has_france ? gpσ * cctx.econ_ctx.γ.LPrime[cctx.econ_ctx.bi] : 0.0
        if cctx.profiled_full_wctx === nothing || cctx.profiled_full_wctx_for !== cf
            cctx.profiled_full_wctx = build_winner_pair_ctx(cf; bi_slot = bi_slot, gpσ = gpσ, denom_cf = denom_cf)
            cctx.profiled_full_wctx_for = cf
        end
        wctx_full = cctx.profiled_full_wctx
        ws_full = cctx.profiled_full_ws
        if ws_full === nothing || ws_full.ncolI != wctx_full.ncolI || ws_full.D != D || ws_full.L != L || ws_full.Ddest != wctx_full.Ddest
            ws_full = WinnerBinCrossScratch(wctx_full.ncolI, D, L, wctx_full.Ddest)
            cctx.profiled_full_ws = ws_full
        end
        @cmhess_prof "H_EC_prep" record_winner_cross_hessian_call!()
        @cmhess_prof "H_EC_prep" winner_pair_cross_hessian_fill!(wctx_full, ws_full, obj, cctx.Bidx)
        if size(cctx.profiled_hraw_ec_full) != (wctx_full.ncolI + 1, nO)
            cctx.profiled_hraw_ec_full = Matrix{Float64}(undef, wctx_full.ncolI + 1, nO)
        end
        Hraw_EC_full = cctx.profiled_hraw_ec_full
        Hraw_EC = cctx.Hraw_EC   # reduced-sized (NCORE x nO) -- reused as the gather TARGET here
        n_bilateral = length(layout.retained_full_factual_j)
        @cmhess_prof "H_EC_asm" @inbounds for l in 1:L
            winner_pair_cross_hessian_cm_block!(Hraw_EC_full, wctx_full, ws_full, l, origins, refIndex1, M; use_profiled_correction = true)
            @views Hraw_EC[1, :] .= Hraw_EC_full[1, :]
            @inbounds for k in 1:n_bilateral
                j_full = layout.retained_full_factual_j[k]
                @views Hraw_EC[1 + k, :] .= Hraw_EC_full[1 + j_full, :]
            end
            if has_france
                @views Hraw_EC[1 + layout.total_reduced_economic_moments, :] .= Hraw_EC_full[1 + wctx_full.ncolI, :]
            end
            if cctx.ncore_core < NCORE
                Hraw_EC_z = @view Hraw_EC[cctx.ncore_core+1:NCORE, :]
                bin_zc_cross_hessian_block!(Hraw_EC_z, bin_zc_ws_ec, l, origins, refIndex1, M)
            end
            cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            block_ec = if cctx.R === nothing
                Hraw_EC
            else
                mul!(cctx.block_ec, Hraw_EC, cctx.R)
            end
            @views Hfull[1:NCORE, cols] .= block_ec
            @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
        end
        @cmhess_prof "H_CC" fill_cm_HCC!(Hfull, cctx, M)
        if extension !== nothing
            _fill_frechet_level_blocks_profiled!(Hfull, cctx, w, wctx_full, ws_full, layout, extension, M)
        end
    else
    # winner-aware H_ER phase (2026-07-27): SAME decision function as the serial
    # hessian_cm_structured! (cm_hessian_architectures.jl), reused not re-derived -- see that
    # function's own docstring for the exact gating rationale.
    use_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cf)
    @cmhess_prof "bintables_prep" if threaded_bins
        tls === nothing && error("hessian_cm_structured_v2!(threaded_bins=true) requires tls (build_thread_local_scratch(cctx))")
        build_bin_tables_threaded!(cctx, tls, H, w; fill_S = !use_winner_bin)
        prefix_sum_tables_threaded!(cctx; fill_S = !use_winner_bin)
    else
        build_bin_tables!(cctx, H, w; fill_S = !use_winner_bin)
        prefix_sum_tables!(cctx; fill_S = !use_winner_bin)
    end

    use_direct_hcz = _cm_cross_hessian_wants_direct_hcz(cctx, cf)
    # harmonization task (2026-07-28): initialized to `nothing` -- see the serial
    # hessian_cm_structured!'s identical fix/comment (wctx/cross_ws are now also passed as plain
    # function arguments to _fill_frechet_level_blocks! below).
    local wctx, cross_ws, bin_zc_ws
    wctx = nothing; cross_ws = nothing; bin_zc_ws = nothing
    if use_winner_bin
        record_winner_cross_hessian_call!()
        wctx = serial_ctx(cctx.core_ws)
        cross_ws = _ensure_cm_cross_scratch!(cctx, wctx.ncolI, D, L)
        # optimize/structured-cross-hessian-ZC-CM-2026-07-28: opt-in threaded H_EC raw-table fill
        # (threaded_cross_hessian.jl), same output as the serial version (bit-identical, see that
        # file's own header) -- gated behind cctx.cross_hessian_threaded, default false.
        @cmhess_prof "H_EC_prep" if cctx.cross_hessian_threaded
            winner_pair_cross_hessian_fill_threaded!(wctx, cross_ws, obj, cctx.Bidx; workers = cctx.cross_hessian_workers)
        else
            winner_pair_cross_hessian_fill!(wctx, cross_ws, obj, cctx.Bidx)
        end
        if use_direct_hcz
            # CM+ZC E/C/Z block-partition + H_CZ release (2026-07-27): SAME pairing as the serial
            # hessian_cm_structured! (cm_hessian_architectures.jl) -- see that file's own comment.
            # `cctx.hzz_centered` was already refreshed this callback by `_fill_cm_HEE!` (H_ZZ, above).
            record_winner_cross_hessian_call!()
            nz = n_restriction(cctx.hzz_zc_op)
            bin_zc_ws = ensure_bin_zc_cross_scratch!(cctx.bin_zc_cross, D, L, nz)
            cctx.bin_zc_cross = bin_zc_ws
            @cmhess_prof "H_CZ_prep" hcz_prep_dispatch!(bin_zc_ws, cctx.hcz_prep_backend, cctx.Bidx, cctx.hzz_centered.ZcS, cctx;
                workers = cctx.cross_hessian_workers, threaded = cctx.cross_hessian_threaded)
        end
    else
        record_dense_cross_hessian_call!()
    end

    # ---- H_EC raw, then optional R congruence (right-multiply by R per threshold block) ----
    # (verbatim from cm_hessian_architectures.jl::hessian_cm_structured! -- see that file's own
    # comment for why the transposed mirror below is necessary)
    # Allocation/Hessian port task §4.2/§6.1: this v2 file was ported BEFORE the production §4.2
    # fix existed, so it still had the original fresh-Hraw_EC/Hraw_CC-per-call allocation pattern
    # (including the L^2-iteration H_CC congruence product, ~1.28 GB/callback at real L=50) --
    # applying the SAME fix here (persistent cctx.Hraw_EC/block_ec/Hraw_CC/RtHraw_CC/block_cc,
    # mul! instead of *) so the threaded-vs-serial benchmark below compares threading itself, not
    # a confound from one side having stale unfixed allocation the other doesn't.
    CS_ = cctx.CScum
    Hraw_EC = cctx.Hraw_EC
    ncore_core = cctx.ncore_core
    @cmhess_prof "H_EC_asm" @inbounds for l in 1:L
        if use_winner_bin
            Hraw_EC_core = use_direct_hcz ? (@view Hraw_EC[1:ncore_core, :]) : Hraw_EC
            winner_pair_cross_hessian_cm_block!(Hraw_EC_core, wctx, cross_ws, l, origins, refIndex1, M)
            if use_direct_hcz
                Hraw_EC_z = @view Hraw_EC[ncore_core+1:NCORE, :]
                bin_zc_cross_hessian_block!(Hraw_EC_z, bin_zc_ws, l, origins, refIndex1, M)
            end
        else
            for (oi, o) in enumerate(origins)
                for j in 1:NCORE
                    Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
                end
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = if cctx.R === nothing
            Hraw_EC
        else
            mul!(cctx.block_ec, Hraw_EC, cctx.R)
        end
        @views Hfull[1:NCORE, cols] .= block_ec
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end

    # ---- H_CC raw, then optional R congruence (per threshold-block pair) ----
    # harmonization task (2026-07-28): extracted to the shared fill_cm_HCC! (cm_hessian_architectures.jl),
    # also used by common Fréchet -- previously a third verbatim copy of this loop.
    @cmhess_prof "H_CC" fill_cm_HCC!(Hfull, cctx, M)

    # harmonization task (2026-07-28): common Fréchet's level blocks -- see the serial
    # hessian_cm_structured!'s identical note (cm_hessian_architectures.jl). `extension !== nothing`
    # (not `isa CMFrechetExtension`) for the same load-order reason documented there.
    if extension !== nothing
        _fill_frechet_level_blocks!(Hfull, cctx, w, H, M, use_winner_bin, wctx, cross_ws, extension)
    end
    end

    # Hessian upper-only cleanup (2026-07-28): now calls the ONE shared packing function
    # (`pack_upper_cm_hessian!`, cm_hessian_architectures.jl) instead of an independently
    # maintained copy of this loop -- this file's own header already flagged that manual-sync
    # duplication as a risk ("Any change to the production function must be mirrored here by
    # hand"); factoring it out removes the risk for this specific loop going forward.
    n = NCORE + ncm
    @cmhess_prof "packing" pack_upper_cm_hessian!(h, Hfull, NCORE, n)
    return h
end

"""
    archC_hess_cb_builder_v2(cctx; threaded_bins=false, tls=nothing, use_syrk=true)

KNITRO Hessian-callback builder wrapping `hessian_cm_structured_v2!`, mirroring
`cm_hessian_architectures.jl::archC_hess_cb_builder`'s wiring exactly (same `@prof` label suffixed
`_v2` so it doesn't collide with the serial-production profile bucket, same
`_INNER_CALL_COUNTERS[].n_hess_calls += 1` bookkeeping -- FIXED here, see file header port-change 1).
"""
function archC_hess_cb_builder_v2(cctx; threaded_bins::Bool = false,
                                   tls::Union{Nothing,ThreadLocalBinScratch} = nothing,
                                   use_syrk::Bool = true)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_v2" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_structured_v2!(evalResult.hess, o, cctx; threaded_bins = threaded_bins, tls = tls, use_syrk = use_syrk)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end

"""
    cm_production_value_v2_hess(x_free0, pcx, cctx; threaded_bins=false, tls=nothing, use_syrk=true) -> (K, base, n_fg, n_hess)

Benchmark entry point: runs a full inner CC dual solve with the CM production context `pcx`
(built via `build_cm_production_context_v2`/`CMConfig`, `cm_hessian_backend=:structured`), but
overrides the Hessian callback with `archC_hess_cb_builder_v2` instead of `pcx.hess_cb_builder`
-- lets ONE already-built `pcx`/`ctx` serve every cell of the serial/threaded x BLAS-thread-count
benchmark matrix without rebuilding the CM context (ctx build ~65s, CM context build a few more
seconds) once per cell.

`cctx` (a `CMBinHessCtx`) is NOT read off `pcx` -- `build_cm_production_context_v2` doesn't expose
it directly for the cumulative path (it's closed over inside `pcx.hess_cb_builder`), a fact already
noted and worked around the same way in `profile_archC_hessian_d20.jl`: rebuild it explicitly via
`build_cm_bin_ctx(ctx, pcx.aug)` (deterministic given `ctx.U`/`pcx.aug.z`/`pcx.aug.origins`, cheap --
reused, not re-derived).
"""
function cm_production_value_v2_hess(x_free0::AbstractVector, pcx, cctx; threaded_bins::Bool = false,
                                      tls::Union{Nothing,ThreadLocalBinScratch} = nothing,
                                      use_syrk::Bool = true)
    obj = pcx.ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, pcx.ctx_cm.m)
    hess_cb_builder = _obj -> archC_hess_cb_builder_v2(cctx; threaded_bins = threaded_bins, tls = tls, use_syrk = use_syrk)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0; hess_cb_builder = hess_cb_builder)
    nStatus in (0, -100, -101, -103) || error("cm_production_value_v2_hess: inner solve failed, nStatus=$nStatus")
    ζstar = x[1]; λstar = collect(x[2:end])
    base = BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
    return K, base, n_fg, n_hess
end
