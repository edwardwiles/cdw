# optimize/structured-cross-hessian-ZC-CM-2026-07-28, ADDENDUM (user-supplied 2026-07-28): H_ZZ
# backend candidates. Scope, per the addendum: dense BLAS is EXPLICITLY DISALLOWED for any block
# touching the economic block E (H_EE/H_EC/H_EZ stay on the winner-aware segmented-reduction
# kernels in winner_pair_cross_hessian.jl/threaded_cross_hessian.jl -- unchanged by this file), and
# H_CZ stays on the bin-structured segmented reduction (bin_zc_cross_hessian_fill!/_threaded!,
# unchanged). This file is ONLY about H_ZZ = (1/M) Z'SZ, the mean/pair-ZC restriction SELF block,
# where the addendum explicitly permits and asks for a BLAS-vs-threaded-Julia comparison.
#
# KEY STRUCTURAL CHANGE from the existing production `zc_restriction_gram!`
# (zc_restriction_operator.jl): that function rematerializes a CENTERED `Zc = Phi - 1t'` (and its
# `S`-weighted copy `ZcS`) from `op.Zraw_all`/`op.Zpairraw_all` EVERY Hessian callback, even though
# `Phi` itself never changes once the campaign's draws are fixed (theta-independent -- see
# `ZCRestrictionOperator`'s own docstring). The addendum's algebraic identity avoids ever
# rematerializing the centered matrix at all:
#
#   Z'SZ = (Phi - 1t')'S(Phi - 1t') = Phi'SPhi - u t' - t u' + s0 t t',
#   u = Phi'S1  (nx-vector),  s0 = 1'S1  (scalar)
#
# so the ONLY per-callback work is (a) a length-W elementwise row-weighting of the IMMUTABLE `Phi`
# (unavoidable -- S changes every callback) to build whichever weighted matrix the chosen backend
# needs, (b) the raw (uncentered) Gram itself, and (c) an O(nx^2) rank-2 correction applied to the
# UPPER TRIANGLE ONLY (matching the addendum's explicit "do not mirror or symmetrize" instruction
# for the raw Gram step -- the correction step still needs the mirror for the final packed-Hessian
# consumer, applied once at the end, not per candidate).

using LinearAlgebra: BLAS, mul!

"""
    ZCRawWeightedWorkspace

Persistent, campaign-lifetime `Phi` (`W x nx`, built ONCE from `op.Zraw_all`/`op.Zpairraw_all`,
NEVER rewritten after construction -- the addendum's own "construct and retain Phi once... do not
rematerialize" requirement) plus a single reusable per-callback row-weighted scratch `RW` (`W x
nx`) shared by every candidate below (holds `sqrt(S).*Phi` for SYRK, `S.*Phi` for GEMM/
threaded_packed -- never both at once, per the addendum's explicit "do not retain both R and Y in
production" instruction). `u`/`HZZraw` are small (`nx`-vector / `nx x nx`) scratch for the
target-correction term and the raw Gram respectively. `tasks` is a persistent `Threads.@spawn`
buffer for the threaded_packed candidate (no per-callback `Vector{Task}` allocation).
"""
mutable struct ZCRawWeightedWorkspace
    W::Int
    nx::Int
    Phi::Matrix{Float64}
    RW::Matrix{Float64}
    u::Vector{Float64}
    HZZraw::Matrix{Float64}
    tvec::Vector{Float64}
    tasks::Vector{Task}
end

"""
    build_zc_raw_weighted_workspace(op::ZCRestrictionOperator, W::Int) -> ZCRawWeightedWorkspace

Builds `Phi` once by concatenating `op.Zraw_all`/`op.Zpairraw_all` into one `W x nx` matrix (SAME
column layout `refresh_zc_centered!` already uses -- mean blocks first, then pair blocks, each in
`op`'s own `Zraw_all[k]`/`Zpairraw_all[k]` order) -- this is the ONE-TIME cost the addendum's
"construct and retain once" language refers to; every subsequent Hessian callback only re-weights
this same matrix, never rebuilds it.
"""
function build_zc_raw_weighted_workspace(op::ZCRestrictionOperator, W::Int)
    nx = n_restriction(op)
    Phi = Matrix{Float64}(undef, W, max(nx, 1))
    D = op.D; npair = op.npair
    @inbounds for k in 1:op.K_mean
        cols = (k-1)*D+1 : k*D
        @views Phi[:, cols] .= op.Zraw_all[k]
    end
    off = op.K_mean * D
    @inbounds for k in 1:op.K_pair
        cols = off+(k-1)*npair+1 : off+k*npair
        @views Phi[:, cols] .= op.Zpairraw_all[k]
    end
    return ZCRawWeightedWorkspace(W, nx, Phi, similar(Phi), zeros(nx), zeros(nx, nx), zeros(nx),
        Vector{Task}(undef, Threads.nthreads()))
end

"Rebuild (or reuse, if already the right size) `ws` for the current `(W, n_restriction(op))` -- campaign-lifetime constant in practice, same idiom as this codebase's other `ensure_*_scratch!` functions."
function ensure_zc_raw_weighted_workspace!(ws::Union{Nothing,ZCRawWeightedWorkspace}, op::ZCRestrictionOperator, W::Int)
    nx = n_restriction(op)
    if ws === nothing || ws.W != W || ws.nx != nx
        return build_zc_raw_weighted_workspace(op, W)
    end
    return ws
end

"""
    refresh_zc_raw_target_vector!(ws::ZCRawWeightedWorkspace, zws::ZCRestrictionWorkspace, op::ZCRestrictionOperator) -> ws

Flatten `zws.targets_mean`/`targets_pair` (already refreshed for the current outer point by
`refresh_zc_targets!`, unchanged) into `ws.tvec`, SAME column layout as `ws.Phi`. Call once per
inner solve (targets are theta-fixed for the whole solve, not per-Hessian-callback), mirroring
`refresh_zc_targets!`'s own "once per inner solve" discipline.
"""
function refresh_zc_raw_target_vector!(ws::ZCRawWeightedWorkspace, zws::ZCRestrictionWorkspace, op::ZCRestrictionOperator)
    D = op.D; npair = op.npair
    @inbounds for k in 1:op.K_mean
        cols = (k-1)*D+1 : k*D
        @views ws.tvec[cols] .= zws.targets_mean[:, k]
    end
    off = op.K_mean * D
    @inbounds for k in 1:op.K_pair
        cols = off+(k-1)*npair+1 : off+k*npair
        @views ws.tvec[cols] .= zws.targets_pair[:, k]
    end
    return ws
end

"Apply the addendum's rank-2 target-correction to the UPPER TRIANGLE of `ws.HZZraw`, writing `(1/M)*(raw - u*t' - t*u' + s0*t*t')` into `HZZ`'s upper triangle (lower triangle mirrored once, after, not per-candidate)."
@inline function _zc_gram_apply_correction!(HZZ::AbstractMatrix{Float64}, HZZraw::AbstractMatrix{Float64},
        u::AbstractVector{Float64}, tvec::AbstractVector{Float64}, s0::Float64, M::Real, nx::Int)
    invM = 1.0 / M
    @inbounds for j2 in 1:nx
        tj2 = tvec[j2]
        for j1 in 1:j2
            HZZ[j1, j2] = invM * (HZZraw[j1, j2] - u[j1] * tj2 - tvec[j1] * u[j2] + s0 * tvec[j1] * tj2)
        end
    end
    @inbounds for j2 in 1:nx, j1 in 1:j2-1
        HZZ[j2, j1] = HZZ[j1, j2]
    end
    return HZZ
end

"""
    zc_gram_blas_syrk!(HZZ, ws, tvec, S, M) -> HZZ

Candidate A. `RW[w,j] = sqrt(S[w])*Phi[w,j]`, then `BLAS.syrk!('U','T',1.0,RW,0.0,HZZraw)` (upper
triangle only, half the FLOPs of a full GEMM), then the rank-2 target correction. Requires `S .>=
0` (true by construction, `S = Psi''(r)`, a convex-conjugate second derivative — never negative for
this codebase's Psi; `sqrt` would throw/NaN otherwise, which is the correct fail-fast behavior, not
worth a defensive check here).
"""
function zc_gram_blas_syrk!(HZZ::AbstractMatrix{Float64}, ws::ZCRawWeightedWorkspace, S::AbstractVector{Float64}, M::Real)
    nx = ws.nx; W = ws.W
    size(HZZ) == (nx, nx) || error("zc_gram_blas_syrk!: size(HZZ)=$(size(HZZ)) != ($nx,$nx)")
    Phi = ws.Phi; RW = ws.RW
    @inbounds for j in 1:nx
        for w in 1:W
            RW[w, j] = sqrt(S[w]) * Phi[w, j]
        end
    end
    HZZraw = ws.HZZraw
    BLAS.syrk!('U', 'T', 1.0, (@view RW[:, 1:nx]), 0.0, HZZraw)
    u = ws.u
    mul!(u, (@view Phi[:, 1:nx])', S)
    s0 = sum(S)
    return _zc_gram_apply_correction!(HZZ, HZZraw, u, ws.tvec, s0, M, nx)
end

"""
    zc_gram_blas_gemm!(HZZ, ws, S, M) -> HZZ

Candidate B. `RW[w,j] = S[w]*Phi[w,j]`, then `mul!(HZZraw, Phi', RW)` (full GEMM, both triangles --
deliberately not restricted to upper, per the addendum's own "measure whether the unnecessary
lower-triangle work makes this slower than SYRK"), then the rank-2 correction (upper only, same as
every other candidate -- the lower triangle GEMM computed is simply discarded, its cost already
paid by the time correction runs).
"""
function zc_gram_blas_gemm!(HZZ::AbstractMatrix{Float64}, ws::ZCRawWeightedWorkspace, S::AbstractVector{Float64}, M::Real)
    nx = ws.nx; W = ws.W
    size(HZZ) == (nx, nx) || error("zc_gram_blas_gemm!: size(HZZ)=$(size(HZZ)) != ($nx,$nx)")
    Phi = ws.Phi; RW = ws.RW
    @inbounds for j in 1:nx
        for w in 1:W
            RW[w, j] = S[w] * Phi[w, j]
        end
    end
    HZZraw = ws.HZZraw
    mul!(HZZraw, (@view Phi[:, 1:nx])', (@view RW[:, 1:nx]))
    u = ws.u
    mul!(u, (@view Phi[:, 1:nx])', S)
    s0 = sum(S)
    return _zc_gram_apply_correction!(HZZ, HZZraw, u, ws.tvec, s0, M, nx)
end

"""
    zc_gram_threaded_packed!(HZZ, ws, S, M; workers) -> HZZ

Candidate C (non-BLAS). Column-block ownership (per the addendum's own "alternatively, benchmark
disjoint upper-column ownership if it avoids the thread-local reduction" suggestion -- adopted
over the thread-local-accumulator design since column-ownership needs no reduce step at all, same
reasoning as this task's other threaded kernels). `RW[w,j] = S[w]*Phi[w,j]` computed once (serial,
`O(W*nx)`, negligible next to the `O(W*nx^2)` main loop for `nx` beyond a handful of columns), then
each worker computes the full upper-triangle column `j2` for its owned columns directly from
`Phi[:,j1]'*RW[:,j2]`, disjoint output, no atomics.
"""
function zc_gram_threaded_packed!(HZZ::AbstractMatrix{Float64}, ws::ZCRawWeightedWorkspace,
        S::AbstractVector{Float64}, M::Real; workers::Int)
    workers <= Threads.nthreads() || error("zc_gram_threaded_packed!: workers=$workers exceeds Threads.nthreads()=$(Threads.nthreads())")
    nx = ws.nx; W = ws.W
    size(HZZ) == (nx, nx) || error("zc_gram_threaded_packed!: size(HZZ)=$(size(HZZ)) != ($nx,$nx)")
    Phi = ws.Phi; RW = ws.RW
    @inbounds for j in 1:nx
        for w in 1:W
            RW[w, j] = S[w] * Phi[w, j]
        end
    end
    HZZraw = ws.HZZraw
    tasks = ws.tasks
    col_chunks = cross_hessian_chunk_ranges(nx, workers)
    for wk in 1:workers
        cols = col_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for j2 in cols
                Yj2 = @view RW[:, j2]
                for j1 in 1:j2
                    Phij1 = @view Phi[:, j1]
                    acc = 0.0
                    for w in 1:W
                        acc += Phij1[w] * Yj2[w]
                    end
                    HZZraw[j1, j2] = acc
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end
    u = ws.u
    mul!(u, (@view Phi[:, 1:nx])', S)
    s0 = sum(S)
    return _zc_gram_apply_correction!(HZZ, HZZraw, u, ws.tvec, s0, M, nx)
end

"""
    ZC_GRAM_BACKEND_DEFAULT

Explicit backend choices for H_ZZ, shared by CM+ZC and origin-ZC (never a separate implementation
per family, per the addendum's own Section 12): `:reference` (existing `zc_restriction_gram!`,
materializes centered Zc/ZcS, single-thread BLAS gemm -- unmodified, the pre-addendum production
path and this task's dense-reference correctness anchor) | `:blas_syrk` | `:blas_gemm` |
`:threaded_packed`. Left at `:reference` (no behavior change) until this task's own correctness/
performance gates justify flipping it -- see `ZC_GRAM_BLAS_VS_THREADED_BENCHMARK_2026-07-28.csv`.
"""
const ZC_GRAM_BACKEND_DEFAULT = Ref{Symbol}(:reference)
const ZC_GRAM_THREADED_WORKERS_DEFAULT = Ref{Int}(resolve_cross_hessian_workers_default())

"""
    zc_gram_dispatch!(HZZ, backend, cs, raw_ws, S, M; workers) -> HZZ

ONE dispatcher, shared by CM+ZC's HMM and origin-ZC's HRR call sites (never two independent
per-family copies). `cs::ZCCenteredScratch` is only touched (via `zc_restriction_gram!`) for
`backend=:reference`; the three new candidates use `raw_ws::ZCRawWeightedWorkspace` instead and
never call `refresh_zc_centered!` at all (the whole point -- see file header).
"""
function zc_gram_dispatch!(HZZ::AbstractMatrix{Float64}, backend::Symbol,
        cs::Union{Nothing,ZCCenteredScratch}, op::ZCRestrictionOperator, raw_ws::Union{Nothing,ZCRawWeightedWorkspace},
        S::AbstractVector{Float64}, M::Real; workers::Int)
    if backend === :reference
        cs === nothing && error("zc_gram_dispatch!: backend=:reference requires a built ZCCenteredScratch")
        return zc_restriction_gram!(HZZ, cs, op, M)
    elseif backend === :blas_syrk
        raw_ws === nothing && error("zc_gram_dispatch!: backend=:blas_syrk requires a built ZCRawWeightedWorkspace")
        return zc_gram_blas_syrk!(HZZ, raw_ws, S, M)
    elseif backend === :blas_gemm
        raw_ws === nothing && error("zc_gram_dispatch!: backend=:blas_gemm requires a built ZCRawWeightedWorkspace")
        return zc_gram_blas_gemm!(HZZ, raw_ws, S, M)
    elseif backend === :threaded_packed
        raw_ws === nothing && error("zc_gram_dispatch!: backend=:threaded_packed requires a built ZCRawWeightedWorkspace")
        return zc_gram_threaded_packed!(HZZ, raw_ws, S, M; workers = workers)
    else
        error("zc_gram_dispatch!: unknown backend :$backend (must be :reference|:blas_syrk|:blas_gemm|:threaded_packed)")
    end
end
