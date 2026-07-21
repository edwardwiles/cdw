# Production integration continuation, Section 10: experimental Julia-thread-parallel
# structured Hessian (draw-chunk accumulation).
#
# The Section 9 profile (docs/fullA_archC_hessian_profile_d20_L50_W80000.csv, D20/L=50/W=80000)
# found `build_bin_tables!` alone is 80.2% of the callback's median wall time (2.51s of 3.14s);
# H_EE (BLAS gemm) is a distant second at 19.1%; every other stage is <1%. This file threads
# ONLY `build_bin_tables!` -- the one stage worth the engineering, per that measurement, not an
# a-priori guess. H_EE already delegates to BLAS (OPENBLAS_NUM_THREADS, orthogonal to Julia
# threads -- pinned to 1 throughout this experiment specifically so JULIA_NUM_THREADS scaling can
# be measured in isolation, avoiding oversubscription from nested threading, per the brief).
#
# Design: DRAW-CHUNK accumulation, no atomics. `1:W` is split into `nthreads()` contiguous,
# balanced ranges; each `@threads` iteration `tid` (the LOOP VARIABLE, not `threadid()` --
# Julia's scheduler does not guarantee iteration `tid` runs on physical thread `tid`, only that
# iterations don't collide when indexed by the loop variable itself) accumulates into its own
# PRIVATE `Tlocal[tid]`/`Slocal[tid]` buffer. A deterministic serial reduction (sum) then combines
# all thread-local buffers into the same `Ttab`/`Stab` fields `build_bin_tables!` would have
# produced directly -- `prefix_sum_tables!` (already validated, <1% of cost, not threaded) runs
# UNCHANGED afterward on the combined table, since prefix-summing commutes with this
# draw-partitioned linear accumulation (prefix-sum(sum_t T_t) == sum_t prefix-sum(T_t), but
# summing raw tables once and prefix-summing once is simpler and avoids redundant work).
using Base.Threads: nthreads, @threads

mutable struct CMBinHessCtxThreaded
    base::CMBinHessCtx
    Tlocal::Vector{Array{Float64,4}}   # nthreads() private D x D x (L+1) x (L+1) buffers
    Slocal::Vector{Array{Float64,3}}   # nthreads() private D x NCORE x (L+1) buffers
    ranges::Vector{UnitRange{Int}}     # nthreads() contiguous, balanced draw-index ranges (fixed once W is known)
end

"""
    build_cm_bin_ctx_threaded(ctx, aug) -> CMBinHessCtxThreaded

Thread-parallel analogue of `build_cm_bin_ctx`. Allocates `nthreads()` private scratch buffers
once (reused across every subsequent Hessian call, exactly like the serial `CMBinHessCtx`'s own
scratch fields) -- sized at construction time, not per-call.
"""
function build_cm_bin_ctx_threaded(ctx, aug)
    base = build_cm_bin_ctx(ctx, aug)
    D = base.D; NCORE = base.NCORE; L1 = base.L + 1
    nt = nthreads()
    Tlocal = [zeros(D, D, L1, L1) for _ in 1:nt]
    Slocal = [zeros(D, NCORE, L1) for _ in 1:nt]
    W = size(base.Bidx, 1)
    ranges = balanced_ranges(W, nt)
    return CMBinHessCtxThreaded(base, Tlocal, Slocal, ranges)
end

"Split `1:W` into `nt` contiguous, size-balanced (differ by at most 1) ranges, deterministic, covering `1:W` exactly once."
function balanced_ranges(W::Int, nt::Int)
    base_len, rem = divrem(W, nt)
    ranges = Vector{UnitRange{Int}}(undef, nt)
    start = 1
    for t in 1:nt
        len = base_len + (t <= rem ? 1 : 0)
        ranges[t] = start:(start + len - 1)
        start += len
    end
    @assert start - 1 == W
    return ranges
end

"Thread-parallel analogue of `build_bin_tables!`: draw-chunk accumulation into private per-thread buffers, then a deterministic serial reduction into `cctxt.base.Ttab`/`Stab` (same fields `build_bin_tables!` writes, so `prefix_sum_tables!`/`hessian_cm_structured!`'s downstream reads are unchanged)."
function build_bin_tables_threaded!(cctxt::CMBinHessCtxThreaded, E::AbstractMatrix{Float64}, w::AbstractVector{Float64})
    base = cctxt.base
    D = base.D; NCORE = base.NCORE; Bidx = base.Bidx
    nt = nthreads()
    @threads for tid in 1:nt
        T = cctxt.Tlocal[tid]; S = cctxt.Slocal[tid]
        fill!(T, 0.0); fill!(S, 0.0)
        @inbounds for s in cctxt.ranges[tid]
            ws = w[s]
            for x in 1:D
                bx = Bidx[s, x]
                for j in 1:NCORE
                    S[x, j, bx] += ws * E[s, j]
                end
            end
            for x in 1:D
                bx = Bidx[s, x]
                for y in 1:D
                    by = Bidx[s, y]
                    T[x, y, bx, by] += ws
                end
            end
        end
    end
    # Deterministic serial reduction (fixed iteration order 1:nt every call -- floating-point sum
    # order is therefore reproducible run-to-run, unlike e.g. an atomic-add reduction would be).
    Tsum = base.Ttab; Ssum = base.Stab
    fill!(Tsum, 0.0); fill!(Ssum, 0.0)
    @inbounds for tid in 1:nt
        Tsum .+= cctxt.Tlocal[tid]
        Ssum .+= cctxt.Slocal[tid]
    end
    return nothing
end

"""
    hessian_cm_structured_threaded!(h, obj, cctxt::CMBinHessCtxThreaded)

Thread-parallel analogue of `hessian_cm_structured!` (cm_hessian_architectures.jl:301-369):
identical in every respect except `build_bin_tables!` -> `build_bin_tables_threaded!`. Any change
to the production function must be mirrored here by hand, same discipline as the Section 9
profiled clone.
"""
function hessian_cm_structured_threaded!(h, obj, cctxt::CMBinHessCtxThreaded)
    cctx = cctxt.base
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins

    E = @view H[:, 2:1+NCORE]
    build_bin_tables_threaded!(cctxt, E, w)
    prefix_sum_tables!(cctx)

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)

    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)

    CS_ = cctx.CScum
    Hraw_EC = Matrix{Float64}(undef, NCORE, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
        @views Hfull[1:NCORE, cols] .= block_ec
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end

    CT = cctx.CT
    Hraw_CC = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
                Hraw_CC[oi, pi] = (CT[o, p, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, p, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = cctx.R === nothing ? Hraw_CC : (cctx.R' * Hraw_CC * cctx.R)
            @views Hfull[rows, cols] .= block
        end
    end

    n = NCORE + ncm
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            h[k] = 0.5 * (Hfull[i, j] + Hfull[j, i])
            k += 1
        end
    end
    return h
end

"Threaded hess_cb_builder, mirrors archC_hess_cb_builder's KNITRO calling convention exactly (cm_hessian_architectures.jl:481-492)."
function archC_threaded_hess_cb_builder(cctxt::CMBinHessCtxThreaded)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        _archC_prep_for_hessian!(o, xloc)
        hessian_cm_structured_threaded!(evalResult.hess, o, cctxt)
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
