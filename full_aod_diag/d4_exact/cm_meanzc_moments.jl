# ============================================================================
# Common-marginals extension: exact equality of country means, optionally plus
# pairwise zero covariance. Production integration of the independently
# audited prototype preserved at archive/fullA-cm-mean-zc-prototype-2026-07-22
# (docs/experiment_cm_pairwise_zero_cov/{FINAL_REPORT,MATH_IMPLEMENTATION_NOTE}.md);
# see docs/fullA_cm_meanzc_integration_note.md for the current derivation and
# the one substantive design change from that prototype (below).
#
# Naming (task brief): "finite-grid CM", "finite-grid CM + exact equal means",
# "finite-grid CM + exact equal means + pairwise zero covariance". The last
# arm is NOT independence and is described as zero covariance, not zero
# correlation, unless finite positive variances are also checked (see
# recovered_covariance_matrix / the positive-variance diagnostic in the gate
# tests).
#
# Additive only: does not modify common_marginals_moments.jl or cm_config.jl's
# pre-existing fields/functions. `cm_extension = :cm_only` never reaches any
# function defined here.
#
# Column layout (unchanged from the prototype's math note Section 4):
#     [ economic (ncore_econ-1) | mean (D) | pair (0 or D(D-1)/2) | CM-grid (ncm) | gravity ]
# -- mean/pair columns sit BEFORE the CM-grid block, so the Hessian's
# "economic" (BLAS) block can simply be widened to include them
# (cm_meanzc_hessian.jl), while the CM-grid block keeps its own bin-indexed
# structure untouched.
#
# DESIGN CHANGE from the archived prototype (audit finding 3.2): ν is NOT
# threaded through a captured `Ref{Float64}`. That design is unsafe under this
# codebase's threaded/concurrent-callback machinery (par_concurrent_evals) --
# a shared mutable Ref read by a closure built ONCE and reused across many
# outer-KNITRO evaluations can be raced by two concurrent evaluations at
# different ν. Instead, ν rides through the SAME `θ` argument every other
# outer parameter already flows through: `wrap_moments_with_cm_meanzc`'s
# closure expects `θ = vcat(θ_econ, ν)` (one element longer than the wrapped
# `core_moments!` expects) and slices it apart internally. `θ` is passed FRESH
# on every call (by `inner_loop_internal_archgeneric`, from a plain local
# variable the caller constructs), so this carries no shared mutable state at
# all -- functionally identical in spirit to how gravity/A_od already ride
# through θ. See docs/fullA_cm_meanzc_integration_note.md Section 5 for the
# full argument, including why this does not reintroduce an AD path onto ν
# (every call site here is a plain Float64 evaluation, never a Dual-typed
# ForwardDiff call).
# ============================================================================

using LinearAlgebra: dot

"""
    packed_pair_index(D::Int) -> Vector{Tuple{Int,Int}}

Deterministic unordered-pair ordering: `(1,2),(1,3),...,(1,D),(2,3),...,(D-1,D)`
(row-major upper triangle, `o` outer loop, `p` inner loop, `o<p`). Exactly
`D*(D-1)/2` entries -- the single canonical ordering every pair-indexed
structure in this file (raw matrix columns, λ_pair slices, covariance-matrix
recovery) must agree on. Never duplicates `(o,p)` and `(p,o)`.
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

"""
    pair_lin_to_oi(k::Int, D::Int) -> (o,p)
    pair_oi_to_lin(o::Int, p::Int, D::Int) -> k

Round-trip index mappings for `packed_pair_index`'s ordering. `pair_oi_to_lin`
accepts either order of `(o,p)`; always returns the index under `o<p`.
"""
pair_lin_to_oi(k::Int, D::Int) = packed_pair_index(D)[k]
function pair_oi_to_lin(o::Int, p::Int, D::Int)
    o, p = o < p ? (o, p) : (p, o)
    return div((o - 1) * (2D - o), 2) + (p - o)
end

n_meanzc_moments(D::Int, cm_extension::Symbol) =
    cm_extension === :cm_plus_equal_means ? D :
    cm_extension === :cm_plus_equal_means_zero_covariance ? D + div(D * (D - 1), 2) :
    error("n_meanzc_moments: cm_extension must be :cm_plus_equal_means or :cm_plus_equal_means_zero_covariance, got $cm_extension")

"""
    build_raw_mean_pair_matrices(U::AbstractMatrix{Float64}; want_pair::Bool) -> (Zraw, Zpairraw)

Precompute the theta-independent raw draw-side objects ONCE per context.
`Zraw` is `U` (`W x D`, copied not aliased). `Zpairraw` is the packed
`W x D(D-1)/2` pair-product matrix (`Zpairraw[:,k] = U[:,o].*U[:,p]` for
`(o,p) = packed_pair_index(D)[k]`), built ONLY when `want_pair=true` -- the
mean-only arm must call this with `want_pair=false` and never allocates it.
"""
function build_raw_mean_pair_matrices(U::AbstractMatrix{Float64}; want_pair::Bool)
    W, D = size(U)
    Zraw = Matrix{Float64}(U)
    want_pair || return Zraw, nothing
    pairs = packed_pair_index(D)
    Zpairraw = Matrix{Float64}(undef, W, length(pairs))
    @inbounds for (k, (o, p)) in enumerate(pairs)
        @views Zpairraw[:, k] .= U[:, o] .* U[:, p]
    end
    return Zraw, Zpairraw
end

"""
    nu_feasible_interval(U::AbstractMatrix{Float64}) -> (lo, hi)

Hard finite-support interval for a common mean ν:
`ν ∈ [max_o min_s U_so, min_o max_s U_so]`. Errors if the interval is empty
(no scalar can lie within every origin's observed draw range) -- callers must
check this before trusting the resulting bounds, never silently widen/clip.
This is a MINIMUM constraint on the production ν box, not a recommended box
width on its own (see cm_meanzc_config.jl's `meanzc_nu_bounds`).
"""
function nu_feasible_interval(U::AbstractMatrix{Float64})
    col_min = vec(minimum(U, dims = 1))
    col_max = vec(maximum(U, dims = 1))
    lo = maximum(col_min)
    hi = minimum(col_max)
    lo < hi || error("nu_feasible_interval: empty interval [lo=$lo, hi=$hi] -- no common-mean value lies within every origin's observed draw range")
    return lo, hi
end

"""
    mean_columns_direct(Zraw, ν) -> Matrix   (W x D)
    mean_columns_anchored(Zraw, ν, refIndex1) -> Matrix   (W x D)

Direct basis: `g_mean_o(s;ν) = z_so - ν` for every origin. Anchored basis:
column `refIndex1` is `z_{s,r} - ν` (the anchor), every other column `o` is
`z_so - z_{s,r}` (a mean CONTRAST, independent of ν) -- same feasible set,
same column count `D`, but only the anchor column carries any ν-dependence.
`d_mean_dnu_direct`/`d_mean_dnu_anchored` give each basis's per-column
`∂g_mean/∂ν` (differs between bases: direct is `-1` in every column, anchored
is `-1` in the anchor column only, `0` elsewhere).
"""
function mean_columns_direct(Zraw::AbstractMatrix{Float64}, ν::Float64)
    return Zraw .- ν
end
function mean_columns_anchored(Zraw::AbstractMatrix{Float64}, ν::Float64, refIndex1::Int)
    D = size(Zraw, 2)
    out = similar(Zraw)
    @views out[:, refIndex1] .= Zraw[:, refIndex1] .- ν
    for o in 1:D
        o == refIndex1 && continue
        @views out[:, o] .= Zraw[:, o] .- Zraw[:, refIndex1]
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
    pair_columns(Zpairraw, ν) -> Matrix   (W x npair)

`g_pair_op(s;ν) = z_so*z_sp - ν²` for every unordered pair. Identical under
either mean basis (the pair block's ν-dependence never changes) --
`d_pair_dnu(ν, npair) = fill(-2ν, npair)`.
"""
pair_columns(Zpairraw::AbstractMatrix{Float64}, ν::Float64) = Zpairraw .- ν^2
d_pair_dnu(ν::Float64, npair::Int) = fill(-2ν, npair)

"""
    wrap_moments_with_cm_meanzc(core_moments!, ncore_econ, CM, Zraw, Zpairraw; meanzc_basis=:direct, refIndex1=1) -> Function

Returns a `moments!`-signature closure `(K, G, θ_ext, U, obj) -> nothing`
producing columns `[economic (ncore_econ-1) | mean (D) | pair (0 or npair) |
CM-grid (ncm) | gravity]`. `Zpairraw === nothing` means the mean-only arm: no
pair columns are written, no `W x npair` temporary is ever touched inside this
closure.

`θ_ext` MUST be `vcat(θ_econ, ν)` -- one element longer than what
`core_moments!` itself expects (see file header: this is how ν reaches this
closure without any shared mutable state). This closure is built ONCE per
production context and reused, exactly like the CM-only path's own
`wrap_moments_with_cm`/`wrap_moments_with_cm_archB` closures.
"""
function wrap_moments_with_cm_meanzc(core_moments!::Function, ncore_econ::Int, CM::Matrix{Float64},
                                      Zraw::Matrix{Float64}, Zpairraw::Union{Nothing,Matrix{Float64}};
                                      meanzc_basis::Symbol = :direct, refIndex1::Int = 1)
    meanzc_basis in (:direct, :anchored) || error("wrap_moments_with_cm_meanzc: meanzc_basis must be :direct or :anchored, got $meanzc_basis")
    pregrav = ncore_econ - 1
    D = size(Zraw, 2)
    n_mean = D
    n_pair = Zpairraw === nothing ? 0 : size(Zpairraw, 2)
    ncm = size(CM, 2)
    return function (K, G, θ_ext, U, obj)
        n = size(U, 1)
        θ_econ = @view θ_ext[1:end-1]
        ν = θ_ext[end]
        G_tmp = similar(G, n, ncore_econ)
        core_moments!(K, G_tmp, θ_econ, U, obj)
        @views G[:, 1:pregrav] .= G_tmp[:, 1:pregrav]
        mean_cols = pregrav+1:pregrav+n_mean
        @views G[:, mean_cols] .= meanzc_basis === :direct ? mean_columns_direct(Zraw[1:n, :], ν) :
                                                              mean_columns_anchored(Zraw[1:n, :], ν, refIndex1)
        if n_pair > 0
            pair_cols = pregrav+n_mean+1:pregrav+n_mean+n_pair
            @views G[:, pair_cols] .= pair_columns(Zpairraw[1:n, :], ν)
        end
        cm_cols = pregrav+n_mean+n_pair+1:pregrav+n_mean+n_pair+ncm
        @views G[:, cm_cols] .= CM[1:n, :]
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    build_cm_meanzc_augmented_obj(ctx, CS; L, cm_extension, contrasts=:anchored, meanzc_basis=:direct, probs=nothing, refIndex1=ctx.γ.refIndex1)

Full-A_od analog of `build_cm_augmented_obj` (common_marginals_moments.jl),
generalized to `cm_extension in (:cm_plus_equal_means, :cm_plus_equal_means_zero_covariance)`
(NOT `:cm_only` -- that sentinel's path is the pre-existing, byte-for-byte
unmodified `build_cm_augmented_obj`; this function asserts against being
called with it).

Returns a NamedTuple with the same shape as `build_cm_augmented_obj`'s result
(`obj_cm, CM, z, origins, ncore, ncm, L, contrasts, refIndex1`) PLUS the
mean/ZC-specific fields: `Zraw, Zpairraw, n_mean, n_pair, meanzc_basis,
cm_extension, ncore_econ` (`ncore_econ` is the ORIGINAL pre-CM economic moment
count, `obj0.d`, needed by `meanzc_fixed_contribution` to locate the mean/pair
λ slice). `ctx.obj` is left untouched. NOTE: `obj_cm.moments!` now expects a
`θ_ext = vcat(θ_econ, ν)` argument (one element longer than the un-augmented
`obj0.moments!`) -- callers must go through `cm_meanzc_production.jl`'s
`archC_meanzc_base_state`/`archC_meanzc_verified_state`, not
`inner_loop_internal_archgeneric` directly with a plain `θ_econ`.
"""
function build_cm_meanzc_augmented_obj(ctx, CS; L::Int, cm_extension::Symbol,
                                        contrasts::Symbol = :anchored, meanzc_basis::Symbol = :direct,
                                        probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                        refIndex1::Int = ctx.γ.refIndex1)
    cm_extension in (:cm_plus_equal_means, :cm_plus_equal_means_zero_covariance) ||
        error("build_cm_meanzc_augmented_obj: cm_extension must be :cm_plus_equal_means or :cm_plus_equal_means_zero_covariance, got $cm_extension (use build_cm_augmented_obj directly for :cm_only)")
    meanzc_basis in (:direct, :anchored) || error("build_cm_meanzc_augmented_obj: meanzc_basis must be :direct or :anchored, got $meanzc_basis")
    want_pair = cm_extension === :cm_plus_equal_means_zero_covariance

    obj0 = ctx.obj
    ncore_econ = obj0.d
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; contrasts = contrasts, probs = probs)
    ncm = size(CM, 2)
    @assert ncm == n_cm_moments(ctx.D, L)

    Zraw, Zpairraw = build_raw_mean_pair_matrices(ctx.U; want_pair = want_pair)
    D = ctx.D
    n_mean = D
    n_pair = want_pair ? size(Zpairraw, 2) : 0
    @assert n_pair == (want_pair ? div(D * (D - 1), 2) : 0)
    @assert n_mean + n_pair == n_meanzc_moments(D, cm_extension)

    d_new = ncore_econ + n_mean + n_pair + ncm
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair + ncm
    moments_meanzc! = wrap_moments_with_cm_meanzc(obj0.moments!, ncore_econ, CM, Zraw, Zpairraw;
                                                   meanzc_basis = meanzc_basis, refIndex1 = refIndex1)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_meanzc!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, CM = CM, z = z, origins = origins, ncore = ncore_econ, ncm = ncm,
            L = L, contrasts = contrasts, refIndex1 = refIndex1,
            Zraw = Zraw, Zpairraw = Zpairraw, n_mean = n_mean, n_pair = n_pair,
            meanzc_basis = meanzc_basis, cm_extension = cm_extension,
            ncore_econ = ncore_econ)
end

"""
    d_delta_dual_d_nu(λstar, aug, ν; mean_m::Float64) -> Float64
    d_delta_dual_d_eta_nu(λstar, aug, ν; mean_m::Float64) -> Float64

Analytic envelope derivative:
`∂Delta_dual/∂ν = -mean_m * ( Σ_o λ_mean,o* · d(mean_o)/dν + Σ_{o<p} λ_pair,op* · d(pair_op)/dν )`,
which for the direct basis is `-mean_m*(Σλ_mean* + 2ν Σλ_pair*)` and for the
anchored basis drops every non-anchor λ_mean term. `λstar` is the FULL inner
dual vector (`base.λstar`); the mean/pair slice is located via
`aug.ncore_econ`/`aug.n_mean`/`aug.n_pair` (same slicing convention
`wrap_moments_with_cm_meanzc` writes columns in). `ν` is passed explicitly
(the caller's current outer ν) -- NOT read from any stored/cached field on
`aug`, since `aug` is built once and reused across many outer evaluations at
different ν. `mean_m` is `verify.m_mean` (mean recovered primal weight, ≈1 at
a verified solution) -- passed explicitly rather than recomputed.
`d_delta_dual_d_eta_nu` applies the extra `ν` factor from `ν=exp(η_ν)`.
"""
function d_delta_dual_d_nu(λstar::AbstractVector{Float64}, aug, ν::Float64; mean_m::Float64)
    ncore_econ = aug.ncore_econ; n_mean = aug.n_mean; n_pair = aug.n_pair
    λ_mean = @view λstar[ncore_econ:ncore_econ+n_mean-1]
    d_mean = aug.meanzc_basis === :direct ? d_mean_dnu_direct(n_mean) : d_mean_dnu_anchored(n_mean, aug.refIndex1)
    total = dot(λ_mean, d_mean)
    if n_pair > 0
        λ_pair = @view λstar[ncore_econ+n_mean:ncore_econ+n_mean+n_pair-1]
        d_pair = d_pair_dnu(ν, n_pair)
        total += dot(λ_pair, d_pair)
    end
    return mean_m * total
end
function d_delta_dual_d_eta_nu(λstar::AbstractVector{Float64}, aug, ν::Float64; mean_m::Float64)
    return ν * d_delta_dual_d_nu(λstar, aug, ν; mean_m = mean_m)
end

"""
    recovered_mean_residuals(m_weights, Zraw, ν) -> Vector{Float64}   (length D)
    recovered_pair_residuals(m_weights, Zpairraw, ν) -> Vector{Float64}   (length npair)
    recovered_covariance_matrix(m_weights, Zraw, D) -> Matrix{Float64}   (D x D)

Primal moment residual recovery: weighted moment `Σ_s m_s * g(s) / W` for each
mean/pair column at the recovered primal weights `m_weights` -- should be ≈0
at a verified CM+mean(+ZC) solution. `recovered_covariance_matrix`
reconstructs the FULL `D x D` weighted covariance matrix (diagonal = weighted
variances, needed for a positive-variance diagnostic before ever describing
the ZC restriction as "zero correlation") from the SAME recovered weights,
independent of whether pair columns were constrained.
"""
function recovered_mean_residuals(m_weights::AbstractVector{Float64}, Zraw::AbstractMatrix{Float64}, ν::Float64)
    W = length(m_weights)
    D = size(Zraw, 2)
    out = Vector{Float64}(undef, D)
    @inbounds for o in 1:D
        out[o] = dot(m_weights, @view(Zraw[:, o])) / W - ν
    end
    return out
end
function recovered_pair_residuals(m_weights::AbstractVector{Float64}, Zpairraw::AbstractMatrix{Float64}, ν::Float64)
    W = length(m_weights)
    npair = size(Zpairraw, 2)
    out = Vector{Float64}(undef, npair)
    @inbounds for k in 1:npair
        out[k] = dot(m_weights, @view(Zpairraw[:, k])) / W - ν^2
    end
    return out
end
function recovered_covariance_matrix(m_weights::AbstractVector{Float64}, Zraw::AbstractMatrix{Float64}, D::Int)
    W = length(m_weights)
    means = [dot(m_weights, @view(Zraw[:, o])) / W for o in 1:D]
    Σ = Matrix{Float64}(undef, D, D)
    @inbounds for o in 1:D, p in 1:D
        Σ[o, p] = dot(m_weights, Zraw[:, o] .* Zraw[:, p]) / W - means[o] * means[p]
    end
    return Σ
end
