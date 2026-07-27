# ================================================================================================
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 1 §4 (ZC blocks): exact,
# allocation-free forward/transpose operator for the origin-specific mean/pairwise-zero-covariance
# (ZC) restriction block, shared by origin-ZC (`G=[E|Z]`) and CM+ZC (`G=[E|C|Z]`).
#
# MATH (addendum §4, "For centered restrictions R = Φ - 1t'"): each mean/pair target block's moment
# column value is `G[s,·] = Φ[s,·] - t[·]` (`cm_originzc_moments.jl::mean_columns_direct`/
# `pair_columns`, UNCHANGED, re-derived here only as the R=Φ-1t' contraction, not re-derived
# mathematically) -- `Φ` = the RAW theta-INDEPENDENT feature matrix (`Zraw_all[k]`/
# `Zpairraw_all[k]`, `build_raw_mean_pair_matrix_levels`, immutable for the whole campaign), `t` =
# the per-outer-point target vector (`νtargets`/`νprod`, from `mean_targets`/`pair_targets`). Then:
#   R*λ  = Φ*λ - 1*(t'λ)          (forward,  O(W·n_o) BLAS gemv + O(n_o) dot)
#   R'*v = Φ'*v - t*(1'v)         (transpose, O(W·n_o) BLAS gemv + O(n_o) axpy)
# NEITHER requires ever constructing the centered `Φ-1t'` matrix, and NEITHER requires the
# per-outer-point `dest = Z .- νtargets'` dense materialization `wrap_moments_with_originzc`/
# `wrap_moments_with_cm_meanzc`'s `moments!` closures currently perform -- this operator reads
# straight from `Zraw_all[k]`/`Zpairraw_all[k]` (built ONCE, reused forever) plus the tiny target
# vector, which is strictly cheaper than the status quo (avoids a fresh (W,D)-or-(W,npair) centered
# copy every outer point) in addition to satisfying the addendum's "no dense G" ask.
#
# ALLOCATION: `lambda_mean`/`lambda_pair`/`g_mean`/`g_pair` are FLAT vectors (length
# `n_mean=K_mean*D` / `n_pair=K_pair*npair`), block `k` occupying `(k-1)*D+1:k*D` (mean) /
# `(k-1)*npair+1:k*npair` (pair) -- the SAME layout `wrap_moments_with_originzc`'s column ranges
# already use. `targets_mean`/`targets_pair` are precomputed ONCE per inner solve (theta/nu fixed
# for the whole KNITRO solve) into persistent `(D,K_mean)`/`(npair,K_pair)` matrices via
# `refresh_zc_targets!`, not recomputed per FG callback. Per-callback cost is `@view` slicing only
# (the same idiom `CMLookupState`'s own `@view x[2:1+ncore1]` already uses throughout this
# codebase) plus BLAS gemv! -- no per-callback heap allocation of new arrays.
#
# `npair = D*(D-1)/2` column ordering matches `Zpairraw_all[k]` exactly (whatever ordering
# `build_raw_mean_pair_matrix_levels` used) -- this operator never re-derives or re-orders it.
# ================================================================================================

using LinearAlgebra: BLAS, dot

"""
    ZCRestrictionOperator

Immutable (campaign-lifetime) description of the mean/pairwise-ZC restriction block: the raw
feature matrices only (`Φ`), never the targets (those are per-outer-point, refreshed once per
inner solve via `refresh_zc_targets!` into a caller-owned `ZCRestrictionWorkspace`). Shared,
unmodified, by origin-ZC and CM+ZC.
"""
struct ZCRestrictionOperator
    Zraw_all::Vector{Matrix{Float64}}       # K_mean matrices, each (W, D)
    Zpairraw_all::Vector{Matrix{Float64}}   # K_pair matrices, each (W, npair)
    D::Int
    npair::Int
    K_mean::Int
    K_pair::Int
end

function ZCRestrictionOperator(Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}}, D::Int)
    npair = D * (D - 1) ÷ 2
    ZCRestrictionOperator(Zraw_all, Zpairraw_all, D, npair, length(Zraw_all), length(Zpairraw_all))
end

"n_mean(op)/n_pair(op)/n_restriction(op): total column counts, matching `n_originzc_moments`/`n_meanzc_moments`."
n_mean(op::ZCRestrictionOperator) = op.K_mean * op.D
n_pair(op::ZCRestrictionOperator) = op.K_pair * op.npair
n_restriction(op::ZCRestrictionOperator) = n_mean(op) + n_pair(op)

"""
    ZCRestrictionWorkspace

Persistent per-inner-solve scratch: refreshed targets only (the gemv! calls themselves write
directly into caller-supplied `arg0`/gradient buffers, no intermediate buffers needed on this
struct). `targets_mean[:,k]`/`targets_pair[:,k]` hold block `k`'s target vector.
"""
mutable struct ZCRestrictionWorkspace
    targets_mean::Matrix{Float64}   # (D, K_mean)
    targets_pair::Matrix{Float64}   # (npair, K_pair)
end

ZCRestrictionWorkspace(op::ZCRestrictionOperator) =
    ZCRestrictionWorkspace(zeros(op.D, max(op.K_mean, 1)), zeros(op.npair, max(op.K_pair, 1)))

"""
    refresh_zc_targets!(ws, op, layout, νfull) -> ws

Recompute `targets_mean`/`targets_pair` for the CURRENT outer point's `νfull` -- call ONCE per
inner solve (before the KNITRO solve starts), not per FG callback, since ν is fixed for the whole
inner solve. Uses `mean_targets`/`pair_targets` (`cm_originzc_target_layout.jl`, UNCHANGED) as the
source of truth for the target values themselves; this function only owns where they're stored.
"""
function refresh_zc_targets!(ws::ZCRestrictionWorkspace, op::ZCRestrictionOperator, layout, νfull::AbstractVector{Float64})
    @inbounds for k in 1:op.K_mean
        ws.targets_mean[:, k] .= mean_targets(layout, νfull, k, op.D)
    end
    @inbounds for k in 1:op.K_pair
        ws.targets_pair[:, k] .= pair_targets(layout, νfull, k, op.D)
    end
    return ws
end

"""
    restriction_forward!(arg0, lambda_mean, lambda_pair, op::ZCRestrictionOperator, ws::ZCRestrictionWorkspace) -> arg0

ACCUMULATES `-(Rλ)` into `arg0` (i.e. `arg0 .-= Σ_k R_k*λ_mean_k .+ Σ_k R_k*λ_pair_k`), matching
every other family's FG convention of building `arg0 = -ζ - Σ(economic) - Σ(restriction)`
incrementally. `lambda_mean`/`lambda_pair` are FLAT vectors (length `n_mean(op)`/`n_pair(op)`,
block layout documented above).
"""
function restriction_forward!(arg0::AbstractVector{Float64},
                               lambda_mean::AbstractVector{Float64}, lambda_pair::AbstractVector{Float64},
                               op::ZCRestrictionOperator, ws::ZCRestrictionWorkspace)
    D = op.D; npair = op.npair
    @inbounds for k in 1:op.K_mean
        λk = @view lambda_mean[(k-1)*D+1:k*D]
        tk = @view ws.targets_mean[:, k]
        BLAS.gemv!('N', -1.0, op.Zraw_all[k], λk, 1.0, arg0)
        arg0 .+= dot(tk, λk)
    end
    @inbounds for k in 1:op.K_pair
        λk = @view lambda_pair[(k-1)*npair+1:k*npair]
        tk = @view ws.targets_pair[:, k]
        BLAS.gemv!('N', -1.0, op.Zpairraw_all[k], λk, 1.0, arg0)
        arg0 .+= dot(tk, λk)
    end
    return arg0
end

"""
    restriction_transpose!(g_mean, g_pair, draw_weights, op::ZCRestrictionOperator, ws::ZCRestrictionWorkspace) -> (g_mean, g_pair)

Writes `g_mean[block k] = -(1/M)*R_k'*draw_weights = -(1/M)*(Φ_k'*draw_weights - t_k*(1'draw_weights))`
into caller-supplied FLAT buffers (`g_mean`/`g_pair`, same block layout as `lambda_mean`/
`lambda_pair`), `M = length(draw_weights)`. No dense `R`/`Φ-1t'`.
"""
function restriction_transpose!(g_mean::AbstractVector{Float64}, g_pair::AbstractVector{Float64},
                                 draw_weights::AbstractVector{Float64},
                                 op::ZCRestrictionOperator, ws::ZCRestrictionWorkspace)
    D = op.D; npair = op.npair
    M = length(draw_weights)
    sw = sum(draw_weights)
    @inbounds for k in 1:op.K_mean
        gk = @view g_mean[(k-1)*D+1:k*D]
        tk = @view ws.targets_mean[:, k]
        BLAS.gemv!('T', -1.0 / M, op.Zraw_all[k], draw_weights, 0.0, gk)
        gk .+= (sw / M) .* tk
    end
    @inbounds for k in 1:op.K_pair
        gk = @view g_pair[(k-1)*npair+1:k*npair]
        tk = @view ws.targets_pair[:, k]
        BLAS.gemv!('T', -1.0 / M, op.Zpairraw_all[k], draw_weights, 0.0, gk)
        gk .+= (sw / M) .* tk
    end
    return g_mean, g_pair
end
