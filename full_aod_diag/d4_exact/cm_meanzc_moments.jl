# ============================================================================
# Common-marginals extension: exact equality of higher-order country moments,
# optionally plus pairwise zero covariance of the SAME power. Production
# integration of the independently audited prototype preserved at
# archive/fullA-cm-mean-zc-prototype-2026-07-22
# (docs/experiment_cm_pairwise_zero_cov/{FINAL_REPORT,MATH_IMPLEMENTATION_NOTE}.md);
# see docs/fullA_cm_meanzc_integration_note.md for the current derivation.
#
# GENERALIZATION beyond the archived prototype (which only covered K=1, i.e.
# "equal MEANS" (order-1) plus, optionally, pairwise zero covariance of the
# levels themselves): for a chosen maximum power K_mean, this restricts
#     E_F[z_o(ω)^k] = ν_k     for every origin o, k = 1..K_mean
# (K_mean outer scalars ν_1..ν_K_mean; under CM every origin already shares one
# marginal, hence shares EVERY raw moment, not just the mean -- exactly why a
# single ν_k per level suffices instead of one per (origin,level)). For a
# chosen K_pair <= K_mean, ALSO restricts
#     E_F[z_o(ω)^k * z_p(ω)^k] = ν_k^2     for every unordered pair o<p, k = 1..K_pair
# (pairwise zero covariance of the k-th powers, for every level k<=K_pair; NOT
# independence, and not "zero correlation" without a finite positive variance
# check -- see recovered_covariance_matrix). K_mean=1,K_pair=0 is exactly the
# old "cm_plus_equal_means" arm; K_mean=1,K_pair=1 is exactly the old
# "cm_plus_equal_means_zero_covariance" arm -- both reproduced bit-for-bit as
# special cases (see test_cm_meanzc_pure_moments.jl's K=1 regression testset
# and test_cm_meanzc_d4_gates.jl's K=1 gate suite).
#
# Why this generalizes cleanly: the restriction's FUNCTIONAL FORM in ν_k is
# identical at every level k -- only the underlying data column changes, from
# z_o (level 1) to z_o^k (level k). So `mean_columns_direct`/`pair_columns`/
# `d_mean_dnu_direct`/`d_pair_dnu` etc. (defined once, generic on "some data
# matrix Z and a scalar ν") are reused UNCHANGED at every level; the only new
# code is looping over k and stacking. The outer Jacobian is BLOCK-DIAGONAL in
# k (level k's moment columns depend on ν_k only, never on ν_{k'} for k'!=k),
# so d_delta_dual_d_nu_vec below is just K_mean independent applications of
# the same single-level envelope formula, one component per level.
#
# Column layout:
#     [ economic (ncore_econ-1) | mean_1(D) mean_2(D) ... mean_{K_mean}(D)
#       | pair_1(npair) ... pair_{K_pair}(npair) | CM-grid (ncm) | gravity ]
# -- all mean blocks grouped, then all pair blocks, then CM-grid, then
# gravity. Mean/pair columns sit BEFORE the CM-grid block so the Hessian's
# "economic" (BLAS) block can simply be widened to include them
# (cm_meanzc_hessian.jl / build_cm_meanzc_bin_ctx), while the CM-grid block
# keeps its own bin-indexed structure untouched.
#
# ν threading (audit finding 3.2, unchanged from the K=1 design): NOT a
# captured `Ref{Float64}`. `theta_ext = vcat(theta_econ, ν_1, ..., ν_{K_mean})`
# rides through the SAME `θ` argument every other outer parameter already
# flows through -- no shared mutable state, safe under this codebase's
# concurrent-callback machinery by construction. Every call site here is a
# plain Float64 evaluation, never a Dual-typed ForwardDiff call.
# ============================================================================

using LinearAlgebra: dot

"""
    packed_pair_index(D::Int) -> Vector{Tuple{Int,Int}}

Deterministic unordered-pair ordering: `(1,2),(1,3),...,(1,D),(2,3),...,(D-1,D)`
(row-major upper triangle, `o` outer loop, `p` inner loop, `o<p`). Exactly
`D*(D-1)/2` entries -- the single canonical ordering every pair-indexed
structure in this file (raw matrix columns, λ_pair slices, covariance-matrix
recovery) must agree on. Never duplicates `(o,p)` and `(p,o)`. Level-
independent (the SAME pair ordering is reused at every power k).
"""
function packed_pair_index(D::Int)
    pairs = Vector{Tuple{Int,Int}}(undef, div(D * (D - 1), 2))
    k = 0
    for o in 1:D-1, p in o+1:D
        k += 1
        pairs[k] = (o, p)
    end
    return pairs
end

pair_lin_to_oi(k::Int, D::Int) = packed_pair_index(D)[k]
function pair_oi_to_lin(o::Int, p::Int, D::Int)
    o, p = o < p ? (o, p) : (p, o)
    return div((o - 1) * (2D - o), 2) + (p - o)
end

"""
    n_meanzc_moments(D::Int, K_mean::Int, K_pair::Int) -> Int

Total new inner moments for `K_mean` mean-type levels and `K_pair<=K_mean`
pair-type levels: `K_mean*D + K_pair*D(D-1)/2`.
"""
function n_meanzc_moments(D::Int, K_mean::Int, K_pair::Int)
    K_mean >= 1 || error("n_meanzc_moments: K_mean must be >= 1, got $K_mean")
    0 <= K_pair <= K_mean || error("n_meanzc_moments: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
    return K_mean * D + K_pair * div(D * (D - 1), 2)
end

"""
    meanzc_extension_to_K(cm_extension::Symbol) -> (K_mean, K_pair)

Sugar mapping the task's original 3-way production enum onto the general
`(K_mean, K_pair)` parameterization -- `:cm_plus_equal_means` => (1,0),
`:cm_plus_equal_means_zero_covariance` => (1,1). `:cm_only` is rejected (that
arm never reaches this file at all -- it is the pre-existing, byte-unmodified
`build_cm_augmented_obj` path). Use `K_mean`/`K_pair` directly for any other
(generalized) configuration; e.g. K_mean=2,K_pair=2 for the order-1-and-2
nested extension.
"""
function meanzc_extension_to_K(cm_extension::Symbol)
    cm_extension === :cm_plus_equal_means && return (1, 0)
    cm_extension === :cm_plus_equal_means_zero_covariance && return (1, 1)
    error("meanzc_extension_to_K: cm_extension must be :cm_plus_equal_means or :cm_plus_equal_means_zero_covariance, got $cm_extension (use build_cm_augmented_obj directly for :cm_only, or pass K_mean/K_pair directly for a generalized configuration)")
end

"""
    build_raw_mean_pair_matrices(U::AbstractMatrix{Float64}, k::Int; want_pair::Bool) -> (Zraw_k, Zpairraw_k)

Precompute the theta-independent raw draw-side objects for POWER LEVEL `k`
ONCE per context: `Zraw_k = U.^k` (`W x D`), `Zpairraw_k[:,j] = (U[:,o].^k) .*
(U[:,p].^k)` for `(o,p) = packed_pair_index(D)[j]` (`W x D(D-1)/2`), built
ONLY when `want_pair=true`.
"""
function build_raw_mean_pair_matrices(U::AbstractMatrix{Float64}, k::Int = 1; want_pair::Bool)
    W, D = size(U)
    Uk = k == 1 ? Matrix{Float64}(U) : U .^ k
    want_pair || return Uk, nothing
    pairs = packed_pair_index(D)
    Zpairraw = Matrix{Float64}(undef, W, length(pairs))
    @inbounds for (j, (o, p)) in enumerate(pairs)
        @views Zpairraw[:, j] .= Uk[:, o] .* Uk[:, p]
    end
    return Uk, Zpairraw
end

"""
    build_raw_mean_pair_matrix_levels(U, K_mean, K_pair) -> (Zraw_all, Zpairraw_all)

`Zraw_all[k] = U.^k` for `k=1:K_mean`; `Zpairraw_all[k]` for `k=1:K_pair`
(empty vector if `K_pair==0`). Computed once per context, reused across every
subsequent inner solve.
"""
function build_raw_mean_pair_matrix_levels(U::AbstractMatrix{Float64}, K_mean::Int, K_pair::Int)
    Zraw_all = Vector{Matrix{Float64}}(undef, K_mean)
    Zpairraw_all = Vector{Matrix{Float64}}(undef, K_pair)
    for k in 1:K_mean
        Zk, Zpk = build_raw_mean_pair_matrices(U, k; want_pair = k <= K_pair)
        Zraw_all[k] = Zk
        k <= K_pair && (Zpairraw_all[k] = Zpk)
    end
    return Zraw_all, Zpairraw_all
end

"""
    nu_feasible_interval(U::AbstractMatrix{Float64}, k::Int=1) -> (lo, hi)

Hard finite-support interval for a common `k`-th moment ν_k:
`ν_k ∈ [max_o min_s U_so^k, min_o max_s U_so^k]`. Errors if the interval is
empty (no scalar can lie within every origin's observed `k`-th-power range).
This is a MINIMUM constraint on the production ν_k box, not a recommended box
width on its own (see cm_meanzc_config.jl's `meanzc_nu_bounds`).
"""
function nu_feasible_interval(U::AbstractMatrix{Float64}, k::Int = 1)
    Uk = k == 1 ? U : U .^ k
    col_min = vec(minimum(Uk, dims = 1))
    col_max = vec(maximum(Uk, dims = 1))
    lo = maximum(col_min)
    hi = minimum(col_max)
    lo < hi || error("nu_feasible_interval: empty interval [lo=$lo, hi=$hi] at k=$k -- no common ν_k value lies within every origin's observed k-th-power range")
    return lo, hi
end

"""
    mean_columns_direct(Z, ν) -> Matrix   (W x D)
    mean_columns_anchored(Z, ν, refIndex1) -> Matrix   (W x D)

Direct basis: `g_o(s;ν) = Z_so - ν` for every origin (generic on any data
matrix `Z`, e.g. `Z = U.^k` for level k). Anchored basis: column `refIndex1`
is `Z_{s,r} - ν` (the anchor), every other column `o` is `Z_so - Z_{s,r}` (a
contrast, independent of ν). `d_mean_dnu_direct`/`d_mean_dnu_anchored` give
each basis's per-column `∂g/∂ν` (level-independent: `-1` per column for
direct, `-1` at the anchor / `0` elsewhere for anchored -- the level enters
only through which `Z` was used to build the columns, not through the
derivative formula itself).
"""
function mean_columns_direct(Z::AbstractMatrix{Float64}, ν::Float64)
    return Z .- ν
end
function mean_columns_anchored(Z::AbstractMatrix{Float64}, ν::Float64, refIndex1::Int)
    D = size(Z, 2)
    out = similar(Z)
    @views out[:, refIndex1] .= Z[:, refIndex1] .- ν
    for o in 1:D
        o == refIndex1 && continue
        @views out[:, o] .= Z[:, o] .- Z[:, refIndex1]
    end
    return out
end
d_mean_dnu_direct(D::Int) = fill(-1.0, D)
function d_mean_dnu_anchored(D::Int, refIndex1::Int)
    v = zeros(D)
    v[refIndex1] = -1.0
    return v
end

"""
    pair_columns(Zpair, ν) -> Matrix   (W x npair)

`g_op(s;ν) = Zpair_s,op - ν²` for every unordered pair (`Zpair` built at a
fixed power level `k`, e.g. `Zpair[:,j] = (U[:,o].^k).*(U[:,p].^k)`).
`d_pair_dnu(ν, npair) = fill(-2ν, npair)` -- level-independent formula.
"""
pair_columns(Zpair::AbstractMatrix{Float64}, ν::Float64) = Zpair .- ν^2
d_pair_dnu(ν::Float64, npair::Int) = fill(-2ν, npair)

"""
    mean_columns_direct!(dest, Z, ν)
    mean_columns_anchored!(dest, Z, ν, refIndex1)
    pair_columns!(dest, Zpair, ν)

In-place analogs of `mean_columns_direct`/`mean_columns_anchored`/
`pair_columns`: write the centered/target-corrected block directly into
`dest` (expected to be a view into the destination moment matrix `G`) via a
single broadcast, with NO intermediate `W x D`/`W x npair` temporary
allocation. Mathematically identical output to the allocating versions
(same formula, `Z .- ν` / `Zpair .- ν^2`) -- these exist purely to avoid the
B5/B6 "materialize-then-copy" pattern (`G[:,cols] .= mean_columns_direct(...)`
allocates the RHS, then copies it into the view) for the immutable-feature
restriction operators (`Φ_R = Z`, target `t_R(η) = ν` or `ν^2`).
"""
function mean_columns_direct!(dest::AbstractMatrix{Float64}, Z::AbstractMatrix{Float64}, ν::Float64)
    @. dest = Z - ν
    return dest
end
function mean_columns_anchored!(dest::AbstractMatrix{Float64}, Z::AbstractMatrix{Float64}, ν::Float64, refIndex1::Int)
    D = size(Z, 2)
    @views dest[:, refIndex1] .= Z[:, refIndex1] .- ν
    for o in 1:D
        o == refIndex1 && continue
        @views dest[:, o] .= Z[:, o] .- Z[:, refIndex1]
    end
    return dest
end
function pair_columns!(dest::AbstractMatrix{Float64}, Zpair::AbstractMatrix{Float64}, ν::Float64)
    ν2 = ν^2
    @. dest = Zpair - ν2
    return dest
end

"""
    wrap_moments_with_cm_meanzc(core_moments!, ncore_econ, CM, Zraw_all, Zpairraw_all; meanzc_basis=:direct, refIndex1=1) -> Function

Returns a `moments!`-signature closure `(K, G, θ_ext, U, obj) -> nothing`
producing columns `[economic (ncore_econ-1) | mean_1..mean_{K_mean} | pair_1..
pair_{K_pair} | CM-grid (ncm) | gravity]` (see file header). `K_mean =
length(Zraw_all)`, `K_pair = length(Zpairraw_all)`.

`θ_ext` MUST be `vcat(θ_econ, ν_1, ..., ν_{K_mean})` -- `K_mean` elements
longer than what `core_moments!` itself expects (file header: this is how ν_k
reaches this closure without any shared mutable state). Built ONCE per
production context and reused, exactly like the CM-only path's own
`wrap_moments_with_cm`/`wrap_moments_with_cm_archB` closures.

B2/B5/B6 immutable-restriction-operator implementation (2026-07-24, Phase B
rescoped per the live B1 audit -- see
docs/CURRENT_CM_BASIS_AND_STORAGE_AUDIT_2026-07-24.md for why flexible CM's
own CM-grid block already has this treatment via Architecture B/C and is NOT
touched here): (1) `G_tmp` is now a closure-captured cache (`Gtmp_cache`,
keyed on `n`, mirroring `wrap_moments_with_cm_archB`'s pattern) instead of a
fresh `similar(G, n, ncore_econ)` allocation on every call -- at D=20/
W=80,000 this buffer is O(W x ncore_econ) ~ tens of MB, previously
reallocated every outer evaluation; (2) the mean/pair centered blocks
(`Φ_R - t_R(η)`, `Φ_R = Zraw_all[k]`/`Zpairraw_all[k]`, `t_R(η) = ν_k`/`ν_k^2`)
are now written DIRECTLY into the destination `G` view via
`mean_columns_direct!`/`mean_columns_anchored!`/`pair_columns!` (a single
in-place broadcast), instead of allocating a full temporary `W x D`/
`W x npair` matrix via the old allocating helpers and then copying it into
`G` (`G[:,cols] .= mean_columns_direct(...)`, which builds and immediately
discards a same-sized temporary on every call). `Zraw_all`/`Zpairraw_all`
themselves are unchanged: already-immutable, already-precomputed-once
per context (`build_raw_mean_pair_matrix_levels`), reused as-is -- this
function does not touch that part of the design, only the per-call fill.
The dense/allocating reference path (`wrap_moments_with_cm_meanzc_dense`,
below) is retained byte-for-byte for validation.
"""
function wrap_moments_with_cm_meanzc(core_moments!::Function, ncore_econ::Int, CM::Matrix{Float64},
                                      Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}};
                                      meanzc_basis::Symbol = :direct, refIndex1::Int = 1)
    meanzc_basis in (:direct, :anchored) || error("wrap_moments_with_cm_meanzc: meanzc_basis must be :direct or :anchored, got $meanzc_basis")
    pregrav = ncore_econ - 1
    D = size(Zraw_all[1], 2)
    K_mean = length(Zraw_all)
    K_pair = length(Zpairraw_all)
    n_mean_total = K_mean * D
    npair = D * (D - 1) ÷ 2
    n_pair_total = K_pair * npair
    ncm = size(CM, 2)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
    return function (K, G, θ_ext, U, obj)
        n = size(U, 1)
        θ_econ = @view θ_ext[1:end-K_mean]
        νs = @view θ_ext[end-K_mean+1:end]
        if size(Gtmp_cache[], 1) != n
            Gtmp_cache[] = Matrix{Float64}(undef, n, ncore_econ)
        end
        G_tmp = Gtmp_cache[]
        core_moments!(K, G_tmp, θ_econ, U, obj)
        @views G[:, 1:pregrav] .= G_tmp[:, 1:pregrav]
        for k in 1:K_mean
            cols = pregrav+(k-1)*D+1 : pregrav+k*D
            dest = @view G[:, cols]
            Zk = @view Zraw_all[k][1:n, :]
            meanzc_basis === :direct ? mean_columns_direct!(dest, Zk, νs[k]) :
                                        mean_columns_anchored!(dest, Zk, νs[k], refIndex1)
        end
        mean_end = pregrav + n_mean_total
        for k in 1:K_pair
            cols = mean_end+(k-1)*npair+1 : mean_end+k*npair
            dest = @view G[:, cols]
            Zpk = @view Zpairraw_all[k][1:n, :]
            pair_columns!(dest, Zpk, νs[k])
        end
        cm_cols = mean_end+n_pair_total+1 : mean_end+n_pair_total+ncm
        @views G[:, cm_cols] .= CM[1:n, :]
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    wrap_moments_with_cm_meanzc_dense(core_moments!, ncore_econ, CM, Zraw_all, Zpairraw_all; meanzc_basis=:direct, refIndex1=1) -> Function

Slow dense reference path, preserved byte-for-byte from the pre-2026-07-24
Phase B implementation (fresh `G_tmp` allocation every call, mean/pair blocks
built via the allocating `mean_columns_direct`/`mean_columns_anchored`/
`pair_columns` then copied into `G`). Kept ONLY for before/after correctness
and benchmark comparison against `wrap_moments_with_cm_meanzc` above -- not
used by any production entry point.
"""
function wrap_moments_with_cm_meanzc_dense(core_moments!::Function, ncore_econ::Int, CM::Matrix{Float64},
                                            Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}};
                                            meanzc_basis::Symbol = :direct, refIndex1::Int = 1)
    meanzc_basis in (:direct, :anchored) || error("wrap_moments_with_cm_meanzc_dense: meanzc_basis must be :direct or :anchored, got $meanzc_basis")
    pregrav = ncore_econ - 1
    D = size(Zraw_all[1], 2)
    K_mean = length(Zraw_all)
    K_pair = length(Zpairraw_all)
    n_mean_total = K_mean * D
    npair = D * (D - 1) ÷ 2
    n_pair_total = K_pair * npair
    ncm = size(CM, 2)
    return function (K, G, θ_ext, U, obj)
        n = size(U, 1)
        θ_econ = @view θ_ext[1:end-K_mean]
        νs = @view θ_ext[end-K_mean+1:end]
        G_tmp = similar(G, n, ncore_econ)
        core_moments!(K, G_tmp, θ_econ, U, obj)
        @views G[:, 1:pregrav] .= G_tmp[:, 1:pregrav]
        for k in 1:K_mean
            cols = pregrav+(k-1)*D+1 : pregrav+k*D
            @views G[:, cols] .= meanzc_basis === :direct ? mean_columns_direct(Zraw_all[k][1:n, :], νs[k]) :
                                                             mean_columns_anchored(Zraw_all[k][1:n, :], νs[k], refIndex1)
        end
        mean_end = pregrav + n_mean_total
        for k in 1:K_pair
            cols = mean_end+(k-1)*npair+1 : mean_end+k*npair
            @views G[:, cols] .= pair_columns(Zpairraw_all[k][1:n, :], νs[k])
        end
        cm_cols = mean_end+n_pair_total+1 : mean_end+n_pair_total+ncm
        @views G[:, cm_cols] .= CM[1:n, :]
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    build_cm_meanzc_augmented_obj(ctx, CS; L, K_mean, K_pair=0, contrasts=:anchored, meanzc_basis=:direct, probs=nothing, refIndex1=ctx.γ.refIndex1)

Full-A_od analog of `build_cm_augmented_obj` (common_marginals_moments.jl),
generalized to `K_mean` mean-type levels and `K_pair<=K_mean` pair-type
levels (`K_mean=1,K_pair=0` == the old `:cm_plus_equal_means` arm;
`K_mean=1,K_pair=1` == the old `:cm_plus_equal_means_zero_covariance` arm --
use `meanzc_extension_to_K(cm_extension)` to translate the old 3-way enum).

Returns a NamedTuple with the same shape as `build_cm_augmented_obj`'s result
(`obj_cm, CM, z, origins, ncore, ncm, L, contrasts, refIndex1`) PLUS the
mean/ZC-specific fields: `Zraw_all, Zpairraw_all, K_mean, K_pair, n_mean,
n_pair, meanzc_basis, ncore_econ` (`ncore_econ` is the ORIGINAL pre-CM
economic moment count, `obj0.d`). `ctx.obj` is left untouched. NOTE:
`obj_cm.moments!` now expects a `θ_ext = vcat(θ_econ, ν_1,...,ν_{K_mean})`
argument (`K_mean` elements longer than the un-augmented `obj0.moments!`) --
callers must go through `cm_meanzc_production.jl`'s
`archC_meanzc_base_state`/`archC_meanzc_verified_state`, not
`inner_loop_internal_archgeneric` directly with a plain `θ_econ`.
"""
function build_cm_meanzc_augmented_obj(ctx, CS; L::Int, K_mean::Int, K_pair::Int = 0,
                                        contrasts::Symbol = :anchored, meanzc_basis::Symbol = :direct,
                                        probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                        refIndex1::Int = ctx.γ.refIndex1)
    K_mean >= 1 || error("build_cm_meanzc_augmented_obj: K_mean must be >= 1, got $K_mean")
    0 <= K_pair <= K_mean || error("build_cm_meanzc_augmented_obj: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
    meanzc_basis in (:direct, :anchored) || error("build_cm_meanzc_augmented_obj: meanzc_basis must be :direct or :anchored, got $meanzc_basis")

    obj0 = ctx.obj
    ncore_econ = obj0.d
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; contrasts = contrasts, probs = probs)
    ncm = size(CM, 2)
    @assert ncm == n_cm_moments(ctx.D, L)

    Zraw_all, Zpairraw_all = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, K_pair)
    D = ctx.D
    npair = div(D * (D - 1), 2)
    n_mean = K_mean * D
    n_pair = K_pair * npair
    @assert n_mean + n_pair == n_meanzc_moments(D, K_mean, K_pair)

    d_new = ncore_econ + n_mean + n_pair + ncm
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair + ncm
    moments_meanzc! = wrap_moments_with_cm_meanzc(obj0.moments!, ncore_econ, CM, Zraw_all, Zpairraw_all;
                                                   meanzc_basis = meanzc_basis, refIndex1 = refIndex1)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_meanzc!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,   # 2026-07-24 release fix: was defaulting to Inf (disabled) on every rebuild
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, CM = CM, z = z, origins = origins, ncore = ncore_econ, ncm = ncm,
            L = L, contrasts = contrasts, refIndex1 = refIndex1,
            Zraw_all = Zraw_all, Zpairraw_all = Zpairraw_all, K_mean = K_mean, K_pair = K_pair,
            n_mean = n_mean, n_pair = n_pair, meanzc_basis = meanzc_basis,
            ncore_econ = ncore_econ)
end

"""
    d_delta_dual_d_nu_vec(λstar, aug, νvec::Vector{Float64}; mean_m::Float64) -> Vector{Float64}
    d_delta_dual_d_eta_nu_vec(λstar, aug, νvec::Vector{Float64}; mean_m::Float64) -> Vector{Float64}

Analytic envelope derivative, ONE component per level `k=1:aug.K_mean`
(the outer Jacobian is block-diagonal in k -- level k's moment columns depend
on ν_k only, see file header):
`∂Delta_dual/∂ν_k = -mean_m * ( Σ_o λ_mean,o,k* · d(mean)/dν_k + [k<=K_pair] Σ_{o<p} λ_pair,op,k* · d(pair)/dν_k )`.
`λstar` is the FULL inner dual vector (`base.λstar`); each level's mean/pair
slice is located via `aug.ncore_econ`/`aug.n_mean`(`=K_mean*D`)/`aug.n_pair`
(`=K_pair*npair`) using the SAME grouped-by-block-then-level layout
`wrap_moments_with_cm_meanzc` writes columns in. `νvec` is the caller's
current outer `(ν_1,...,ν_{K_mean})` -- passed explicitly, never read from a
stored/cached field. `mean_m` is `verify.m_mean` (≈1 at a verified solution).
`d_delta_dual_d_eta_nu_vec` applies the `ν_k` factor from `ν_k=exp(η_{ν,k})`,
per level.
"""
function d_delta_dual_d_nu_vec(λstar::AbstractVector{Float64}, aug, νvec::AbstractVector{Float64}; mean_m::Float64)
    K_mean = aug.K_mean; K_pair = aug.K_pair
    length(νvec) == K_mean || error("d_delta_dual_d_nu_vec: length(νvec)=$(length(νvec)) != aug.K_mean=$K_mean")
    ncore_econ = aug.ncore_econ
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    d_mean = aug.meanzc_basis === :direct ? d_mean_dnu_direct(D) : d_mean_dnu_anchored(D, aug.refIndex1)
    out = Vector{Float64}(undef, K_mean)
    mean_start = ncore_econ
    pair_start0 = ncore_econ + K_mean * D
    for k in 1:K_mean
        λ_mean_k = @view λstar[mean_start+(k-1)*D : mean_start+k*D-1]
        total = dot(λ_mean_k, d_mean)
        if k <= K_pair
            λ_pair_k = @view λstar[pair_start0+(k-1)*npair : pair_start0+k*npair-1]
            total += dot(λ_pair_k, d_pair_dnu(νvec[k], npair))
        end
        out[k] = mean_m * total
    end
    return out
end
function d_delta_dual_d_eta_nu_vec(λstar::AbstractVector{Float64}, aug, νvec::AbstractVector{Float64}; mean_m::Float64)
    d_nu = d_delta_dual_d_nu_vec(λstar, aug, νvec; mean_m = mean_m)
    return νvec .* d_nu
end

"""
    recovered_mean_residuals(m_weights, Z, ν) -> Vector{Float64}   (length D)
    recovered_pair_residuals(m_weights, Zpair, ν) -> Vector{Float64}   (length npair)
    recovered_covariance_matrix(m_weights, Z, D) -> Matrix{Float64}   (D x D)

Primal moment residual recovery: weighted moment `Σ_s m_s * g(s) / W` for each
mean/pair column at the recovered primal weights `m_weights` (`Z` = `U.^k` for
whichever level is being checked) -- should be ≈0 at a verified solution.
`recovered_covariance_matrix` reconstructs the FULL `D x D` weighted
covariance matrix of `Z`'s own columns (diagonal = weighted variances --
needed for a positive-variance diagnostic before ever describing a zero-
covariance restriction as "zero correlation").
"""
function recovered_mean_residuals(m_weights::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, ν::Float64)
    W = length(m_weights)
    D = size(Z, 2)
    out = Vector{Float64}(undef, D)
    @inbounds for o in 1:D
        out[o] = dot(m_weights, @view(Z[:, o])) / W - ν
    end
    return out
end
function recovered_pair_residuals(m_weights::AbstractVector{Float64}, Zpair::AbstractMatrix{Float64}, ν::Float64)
    W = length(m_weights)
    npair = size(Zpair, 2)
    out = Vector{Float64}(undef, npair)
    @inbounds for k in 1:npair
        out[k] = dot(m_weights, @view(Zpair[:, k])) / W - ν^2
    end
    return out
end
function recovered_covariance_matrix(m_weights::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, D::Int)
    W = length(m_weights)
    means = [dot(m_weights, @view(Z[:, o])) / W for o in 1:D]
    Σ = Matrix{Float64}(undef, D, D)
    @inbounds for o in 1:D, p in 1:D
        Σ[o, p] = dot(m_weights, Z[:, o] .* Z[:, p]) / W - means[o] * means[p]
    end
    return Σ
end
