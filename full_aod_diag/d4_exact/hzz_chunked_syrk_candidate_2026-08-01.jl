# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Section 10: row-chunked SYRK candidate
# for H_ZZ. The existing `:blas_syrk` (zc_gram_blas_candidates.jl) materializes the FULL `(W, nx)`
# row-weighted workspace `RW = sqrt(S).*Phi` before calling `BLAS.syrk!` once -- at W=100,000,
# nx=630 that is one ~504MB matrix (Phi, persistent) plus a second ~504MB scratch (RW, rebuilt every
# callback); at W=500,000 both become ~2.52GB (see HZZ_FULL_WORKSPACE_COST_AUDIT_2026-08-01.csv).
#
# This candidate instead streams `W` in chunks of a persistent `(chunk_size, nx)` buffer (far
# smaller, e.g. 2048*630*8 bytes =~ 10MB): for each chunk, (1) fill `sqrt(S_w)*Phi_{wj}` into the
# small buffer, (2) accumulate `X'X` into `HZZraw` via `BLAS.syrk!` with `beta=1.0` (upper triangle
# only, ACCUMULATING across chunks rather than overwriting), (3) accumulate `u += Phi_chunk'*S_chunk`
# and `s0 += sum(S_chunk)` via the same chunk's rows. The rank-2 correction is applied ONCE after
# all chunks, identical to the full-workspace candidate. Same target cells, same total FLOP count --
# this trades one large contiguous BLAS call for several smaller ones, in exchange for peak
# workspace memory that no longer scales with `W`.

mutable struct ZCChunkedSyrkWorkspace
    W::Int
    nx::Int
    chunk_size::Int
    RWchunk::Matrix{Float64}   # (chunk_size, nx) persistent, reused across chunks and callbacks
end
function ZCChunkedSyrkWorkspace(W::Int, nx::Int, chunk_size::Int)
    ZCChunkedSyrkWorkspace(W, nx, chunk_size, Matrix{Float64}(undef, chunk_size, max(nx, 1)))
end
function ensure_zc_chunked_syrk_workspace!(ws::Union{Nothing,ZCChunkedSyrkWorkspace}, W::Int, nx::Int, chunk_size::Int)
    if ws === nothing || ws.W != W || ws.nx != nx || ws.chunk_size != chunk_size
        return ZCChunkedSyrkWorkspace(W, nx, chunk_size)
    end
    return ws
end

"""
    zc_gram_blas_syrk_chunked!(HZZ, ws, cws, S, M; chunk_size) -> HZZ

Row-chunked SYRK candidate. `ws::ZCRawWeightedWorkspace` supplies the persistent `Phi`/`u`/`HZZraw`/
`tvec` (SAME fields the full-workspace `:blas_syrk` candidate uses -- this candidate does NOT use
`ws.RW` at all, only `cws.RWchunk`, so the two candidates can coexist against the same `ws` without
clobbering each other). `cws::ZCChunkedSyrkWorkspace` supplies the small persistent chunk buffer.
"""
function zc_gram_blas_syrk_chunked!(HZZ::AbstractMatrix{Float64}, ws::ZCRawWeightedWorkspace,
        cws::ZCChunkedSyrkWorkspace, S::AbstractVector{Float64}, M::Real; chunk_size::Int = cws.chunk_size)
    nx = ws.nx; W = ws.W
    size(HZZ) == (nx, nx) || error("zc_gram_blas_syrk_chunked!: size(HZZ)=$(size(HZZ)) != ($nx,$nx)")
    chunk_size == cws.chunk_size || error("zc_gram_blas_syrk_chunked!: chunk_size=$chunk_size != cws.chunk_size=$(cws.chunk_size) -- rebuild cws")
    Phi = ws.Phi
    HZZraw = ws.HZZraw
    fill!(HZZraw, 0.0)
    u = ws.u
    fill!(u, 0.0)
    s0 = 0.0
    RWchunk = cws.RWchunk

    w0 = 1
    while w0 <= W
        w1 = min(w0 + chunk_size - 1, W)
        clen = w1 - w0 + 1
        Phic = @view Phi[w0:w1, 1:nx]
        Sc = @view S[w0:w1]
        RWc = @view RWchunk[1:clen, 1:nx]
        @cmhess_prof "H_ZZ_weight_inner" begin
            @inbounds for j in 1:nx
                for (iw, w) in enumerate(w0:w1)
                    RWc[iw, j] = sqrt(S[w]) * Phi[w, j]
                end
            end
        end
        @cmhess_prof "H_ZZ_gemm_inner" BLAS.syrk!('U', 'T', 1.0, RWc, 1.0, HZZraw)   # beta=1.0: ACCUMULATE across chunks
        # u += Phi_chunk' * S_chunk (accumulate); s0 += sum(S_chunk)
        mul!(u, Phic', Sc, 1.0, 1.0)   # 5-arg mul!: u = 1.0*Phic'*Sc + 1.0*u
        s0 += sum(Sc)
        w0 = w1 + 1
    end

    return @cmhess_prof "H_ZZ_correction_inner" _zc_gram_apply_correction!(HZZ, HZZraw, u, ws.tvec, s0, M, nx)
end
