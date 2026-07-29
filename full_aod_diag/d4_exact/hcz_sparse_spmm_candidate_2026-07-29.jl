# Part C, Candidate 4 — precomputed sparse one-hot SpMM H_CZ prep (2026-07-29).
#
# `Bidx` (W x D bin-index matrix) is theta-independent, precomputed once per campaign. This
# candidate turns it into an explicit W x (D*(L+1)) sparse one-hot incidence matrix (D nonzeros per
# row -- one per CM origin's current bin membership) ONCE, then every Hessian callback's raw
# bin-feature reduction (docs/PART_C_HCZ_FORMULA_2026-07-29.md step 1) becomes a single sparse-CSC'
# times-dense multiply: `Tflat = Bsp' * ZcS`. Never materializes a dense CM matrix C.
using SparseArrays: SparseMatrixCSC, sparse

"""
    BinZCrossSparseScratch

`Bsp` (`W x (D*(L+1))`, CSC, theta-independent, built once per campaign from `Bidx`) plus the
per-callback dense output scratch `Tflat` (`(D*(L+1)) x n_z`). Column layout: column `(b-1)*D + x`
holds origin `x`'s bin-`b` indicator.
"""
mutable struct BinZCrossSparseScratch
    D::Int
    L::Int
    nz::Int
    Bsp::SparseMatrixCSC{Float64,Int}
    Tflat::Matrix{Float64}
end

"Build the theta-independent sparse incidence matrix from `Bidx` (W x D) -- call ONCE per campaign, never per Hessian callback."
function build_bin_zc_sparse_scratch(Bidx::AbstractMatrix{<:Integer}, D::Int, L::Int, nz::Int)
    W = size(Bidx, 1)
    nbins = L + 1
    ncols = D * nbins
    Ivec = Vector{Int}(undef, W * D)
    Jvec = Vector{Int}(undef, W * D)
    Vvec = ones(Float64, W * D)
    k = 0
    @inbounds for w in 1:W, x in 1:D
        k += 1
        Ivec[k] = w
        b = Bidx[w, x]
        Jvec[k] = (b - 1) * D + x
    end
    Bsp = sparse(Ivec, Jvec, Vvec, W, ncols)
    return BinZCrossSparseScratch(D, L, nz, Bsp, zeros(ncols, max(nz, 1)))
end

"Rebuild (or reuse, if already the right size) -- mirrors this file's own `ensure_*_scratch!` idiom. Rebuild is required whenever `Bidx` itself changes (never within one campaign in production, but always in a fresh test/gate)."
function ensure_bin_zc_sparse_scratch!(sc::Union{Nothing,BinZCrossSparseScratch}, Bidx::AbstractMatrix{<:Integer}, D::Int, L::Int, nz::Int)
    if sc === nothing || sc.D != D || sc.L != L || sc.nz != nz
        return build_bin_zc_sparse_scratch(Bidx, D, L, nz)
    end
    return sc
end

"""
    bin_zc_cross_hessian_fill_sparse!(ws::BinZCrossScratch, sc::BinZCrossSparseScratch, ZcS) -> ws

Sparse-SpMM candidate for the same `ws.ZBinTab`/`ws.ZBinCScum` outputs every other H_CZ-prep
candidate fills. `Tflat = Bsp' * ZcS` is the ENTIRE per-callback draw-level cost (a single call into
SparseArrays' CSC-adjoint-times-dense method) -- then a small `O(D*n_z*L)` reshape/cumulative pass,
same as every other candidate.
"""
function bin_zc_cross_hessian_fill_sparse!(ws::BinZCrossScratch, sc::BinZCrossSparseScratch, ZcS::AbstractMatrix{Float64})
    D = ws.D; L = ws.L; nz = ws.nz
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_sparse!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")
    Tflat = sc.Tflat
    mul!(Tflat, sc.Bsp', (@view ZcS[:, 1:nz]))
    ZBinTab = ws.ZBinTab
    @inbounds for b in 1:(L+1)
        for x in 1:D
            row = (b - 1) * D + x
            for j in 1:nz
                ZBinTab[x, j, b] = Tflat[row, j]
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
