# ============================================================================
# Nested CM extension: finite-grid CM + exact common first moments (+ pairwise
# zero covariance). Trial branch experiment/fullA-cm-pairwise-zero-cov, forked
# from remediation/fullA-exact-2026-07-22 @ 82dd485. See
# docs/experiment_cm_pairwise_zero_cov/MATH_IMPLEMENTATION_NOTE.md for the
# full derivation this file implements.
#
# Additive only: does not modify common_marginals_moments.jl, cm_config.jl, or
# any existing CM path. cm_extension = :cm_only (the default everywhere else in
# this repo) never reaches any function defined here.
#
# Column layout (see math note Section 4): the combined `moments!` wrapper
# built by `build_cm_meanzc_augmented_obj` produces
#     [ economic (ncore_econ-1) | mean (D) | pair (0 or D(D-1)/2) | CM-grid (ncm) | gravity ]
# -- mean/pair columns sit BEFORE the CM-grid block (not after), so that the
# Hessian "economic" (BLAS) block can simply be widened to include them
# (Section 4), while the CM-grid block keeps its own bin-indexed structure
# untouched.
#
# ν never enters θ_full (see math note Section 5) -- it is threaded through a
# captured `Ref{Float64}` (`nu_ref`) that the outer driver sets once per
# outer-loop evaluation, strictly before the inner solve. This keeps ν
# completely outside FreeParamMap/pivot_expand/gravity-elimination machinery
# and guarantees no automatic-differentiation path ever touches it.
# ============================================================================

using LinearAlgebra: dot

"""
    packed_pair_index(D::Int) -> Vector{Tuple{Int,Int}}

Deterministic unordered-pair ordering: `(1,2),(1,3),...,(1,D),(2,3),...,(D-1,D)`
(row-major upper triangle, `o` outer loop, `p` inner loop, `o<p`). Exactly
`D*(D-1)/2` entries -- the single canonical ordering every pair-indexed
structure in this file (raw matrix columns, λ_pair slices, covariance-matrix
recovery) must agree on.
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

Round-trip index mappings for `packed_pair_index`'s ordering (Section 2's
"test round-trip index mappings" requirement). `pair_oi_to_lin` accepts either
order of `(o,p)`; always returns the index under `o<p`.
"""
function pair_lin_to_oi(k::Int, D::Int)
    return packed_pair_index(D)[k]
end
function pair_oi_to_lin(o::Int, p::Int, D::Int)
    o, p = o < p ? (o, p) : (p, o)
    # closed form for row-major upper-triangle offset, avoids rebuilding the full list
    return div((o - 1) * (2D - o), 2) + (p - o)
end

"""
    build_raw_mean_pair_matrices(U::AbstractMatrix{Float64}; want_pair::Bool) -> (Zraw, Zpairraw)

Precompute the theta-independent raw draw-side objects (Section 4 of the task
brief: "Precompute draw-side objects once per context ... do not rebuild pair
products in every inner callback or outer probe"). `Zraw` is literally `U`
(the `W x D` productivity draw matrix -- copied, not aliased, so a caller can
never accidentally mutate `ctx.U` through this). `Zpairraw` is the packed
`W x D(D-1)/2` pair-product matrix (`Zpairraw[:,k] = U[:,o].*U[:,p]` for
`(o,p) = packed_pair_index(D)[k]`), built ONLY when `want_pair=true` -- the
mean-only arm must call this with `want_pair=false` and must never allocate
it (verified by `c40_test_meanzc_pure_moments.jl`'s allocation check and
Section 8's D=20 allocation counters).
"""
function build_raw_mean_pair_matrices(U::AbstractMatrix{Float64}; want_pair::Bool)
    W, D = size(U)
    Zraw = Matrix{Float64}(U)
    if !want_pair
        return Zraw, nothing
    end
    pairs = packed_pair_index(D)
    npair = length(pairs)
    Zpairraw = Matrix{Float64}(undef, W, npair)
    @inbounds for (k, (o, p)) in enumerate(pairs)
        @views Zpairraw[:, k] .= U[:, o] .* U[:, p]
    end
    return Zraw, Zpairraw
end

"""
    nu_feasible_interval(U::AbstractMatrix{Float64}) -> (lo, hi)

Hard finite-support interval for a common mean ν (task brief Section 3):
`ν ∈ [max_o min_s U_so, min_o max_s U_so]`. Errors if the interval is empty
(would mean no scalar can simultaneously lie within every origin's observed
draw range) -- callers must check this before trusting the resulting bounds,
never silently widen/clip.
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

Direct basis: `g_mean_o(s;ν) = z_so - ν` for every origin (math note Section
3). Anchored basis: column `refIndex1` is `z_{s,r} - ν` (the anchor), every
other column `o` is `z_so - z_{s,r}` (a mean CONTRAST, independent of ν) --
same feasible set, same column count `D`, but only the anchor column carries
any ν-dependence. `d_mean_dnu_direct`/`d_mean_dnu_anchored` give each basis's
per-column `∂g_mean/∂ν` (Section 6's envelope derivative needs this, and it
DIFFERS between bases: direct is `-1` in every column, anchored is `-1` in the
anchor column only, `0` elsewhere).
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

`g_pair_op(s;ν) = z_so*z_sp - ν²` for every unordered pair (math note Section
3). Identical under either mean basis (the pair block's ν-dependence never
changes) -- `d_pair_dnu(ν, npair) = fill(-2ν, npair)`.
"""
function pair_columns(Zpairraw::AbstractMatrix{Float64}, ν::Float64)
    return Zpairraw .- ν^2
end
d_pair_dnu(ν::Float64, npair::Int) = fill(-2ν, npair)

"""
    wrap_moments_with_cm_meanzc(core_moments!, ncore_econ, CM, Zraw, Zpairraw, nu_ref; meanzc_basis=:direct, refIndex1=1) -> Function

Returns a `moments!`-signature closure `(K, G, θ, U, obj) -> nothing` producing
columns `[economic (ncore_econ-1) | mean (D) | pair (0 or npair) | CM-grid
(ncm) | gravity]` (see file header). `Zpairraw === nothing` means the
mean-only arm: no pair columns are written, no `W x npair` temporary is ever
touched inside this closure. `ν` is read fresh from `nu_ref[]` on every call
(never cached across calls) -- the outer driver is responsible for setting
`nu_ref[]` to the CURRENT outer ν before invoking the inner solve; a
mismatched/stale `nu_ref[]` would silently solve the wrong inner problem, so
callers should assert `nu_ref[] == ν_expected` immediately before the inner
solve call (see `c40_meanzc_outer_driver.jl`).
"""
function wrap_moments_with_cm_meanzc(core_moments!::Function, ncore_econ::Int, CM::Matrix{Float64},
                                      Zraw::Matrix{Float64}, Zpairraw::Union{Nothing,Matrix{Float64}},
                                      nu_ref::Ref{Float64}; meanzc_basis::Symbol = :direct, refIndex1::Int = 1)
    meanzc_basis in (:direct, :anchored) || error("wrap_moments_with_cm_meanzc: meanzc_basis must be :direct or :anchored, got $meanzc_basis")
    pregrav = ncore_econ - 1
    D = size(Zraw, 2)
    n_mean = D
    n_pair = Zpairraw === nothing ? 0 : size(Zpairraw, 2)
    ncm = size(CM, 2)
    return function (K, G, θ, U, obj)
        n = size(U, 1)
        G_tmp = similar(G, n, ncore_econ)
        core_moments!(K, G_tmp, θ, U, obj)
        ν = nu_ref[]
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
    build_cm_meanzc_augmented_obj(ctx, CS; L, cm_extension, contrasts=:anchored, meanzc_basis=:direct, probs=nothing, refIndex1=ctx.γ.refIndex1, nu_ref=Ref(1.0))

Full-A_od analog of `build_cm_augmented_obj` (common_marginals_moments.jl),
generalized to `cm_extension in (:cm_plus_mean, :cm_plus_mean_zero_covariance)`
(NOT `:cm_only` -- that sentinel's path is the pre-existing, byte-for-byte
unmodified `build_cm_augmented_obj`; this function asserts against being
called with it, so a caller cannot silently construct a redundant/inconsistent
second CM-only object through this file).

Returns a NamedTuple with the same shape as `build_cm_augmented_obj`'s result
(`obj_cm, CM, z, origins, ncore, ncm, L, contrasts, refIndex1`) PLUS the
mean/ZC-specific fields: `Zraw, Zpairraw, n_mean, n_pair, nu_ref, meanzc_basis,
cm_extension, ncore_econ` (`ncore_econ` is the ORIGINAL pre-CM economic moment
count, `obj0.d`, needed by `meanzc_fixed_contribution` to locate the mean/pair
λ slice). `ctx.obj` is left untouched, matching every other builder in this
file family.
"""
function build_cm_meanzc_augmented_obj(ctx, CS; L::Int, cm_extension::Symbol,
                                        contrasts::Symbol = :anchored, meanzc_basis::Symbol = :direct,
                                        probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                        refIndex1::Int = ctx.γ.refIndex1,
                                        nu_ref::Ref{Float64} = Ref(1.0))
    cm_extension in (:cm_plus_mean, :cm_plus_mean_zero_covariance) ||
        error("build_cm_meanzc_augmented_obj: cm_extension must be :cm_plus_mean or :cm_plus_mean_zero_covariance, got $cm_extension (use build_cm_augmented_obj directly for :cm_only)")
    meanzc_basis in (:direct, :anchored) || error("build_cm_meanzc_augmented_obj: meanzc_basis must be :direct or :anchored, got $meanzc_basis")
    want_pair = cm_extension === :cm_plus_mean_zero_covariance

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

    d_new = ncore_econ + n_mean + n_pair + ncm
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair + ncm
    moments_meanzc! = wrap_moments_with_cm_meanzc(obj0.moments!, ncore_econ, CM, Zraw, Zpairraw, nu_ref;
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
            nu_ref = nu_ref, meanzc_basis = meanzc_basis, cm_extension = cm_extension,
            ncore_econ = ncore_econ)
end

"""
    d_delta_dual_d_nu(λstar, aug; mean_m::Float64) -> Float64
    d_delta_dual_d_eta_nu(λstar, aug, ν; mean_m::Float64) -> Float64

Analytic envelope derivative (math note Section 6):
`∂Delta_dual/∂ν = -mean_m * ( Σ_o λ_mean,o* · d(mean_o)/dν + Σ_{o<p} λ_pair,op* · d(pair_op)/dν )`,
which for the direct basis is `-mean_m*(Σλ_mean* + 2ν Σλ_pair*)` and for the
anchored basis drops every non-anchor λ_mean term (`d_mean_dnu_anchored` is
zero there). `λstar` is the FULL inner dual vector (`base.λstar`); the
mean/pair slice is located via `aug.ncore_econ`/`aug.n_mean`/`aug.n_pair`
(same slicing convention `wrap_moments_with_cm_meanzc` writes columns in).
`mean_m` is `verify.m_mean` (mean recovered primal weight, ≈1 at a verified
solution) -- passed explicitly rather than recomputed, since the caller
already has it from `archC_verified_state`/`cm_production_value_verified`.
`d_delta_dual_d_eta_nu` applies the extra `ν` factor from `ν=exp(η_ν)`.
"""
function d_delta_dual_d_nu(λstar::AbstractVector{Float64}, aug; mean_m::Float64)
    ncore_econ = aug.ncore_econ; n_mean = aug.n_mean; n_pair = aug.n_pair
    ν = aug.nu_ref[]
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
    return ν * d_delta_dual_d_nu(λstar, aug; mean_m = mean_m)
end

"""
    recovered_mean_residuals(m_weights, Zraw, ν) -> Vector{Float64}   (length D)
    recovered_pair_residuals(m_weights, Zpairraw, ν) -> Vector{Float64}   (length npair)
    recovered_covariance_matrix(m_weights, Zraw, D) -> Matrix{Float64}   (D x D)

Primal moment residual recovery (task brief Section 4/6.1): weighted moment
`Σ_s m_s * g(s) / W` for each mean/pair column at the recovered primal weights
`m_weights` (`obj.arg1` / `base.m` depending on caller) -- should be ≈0 at a
verified CM+mean(+ZC) solution (mean residual) since those are the imposed
equality restrictions. `recovered_covariance_matrix` reconstructs the FULL
`D x D` weighted covariance matrix (diagonal = weighted variances, needed for
the Section 6.1 positive-variance diagnostic and Section 8's "largest absolute
off-diagonal covariance / associated correlations" report) from the SAME
recovered weights, independent of whether pair columns were constrained
(so it can be computed as a pure diagnostic even for the mean-only arm).
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
