# ================================================================================================
# Common-Frechet matrix-free inner FG operator -- Phase 5.2 remediation (2026-07-26).
#
# Extends the validated flexible-CM lookup approach (cm_lookup_kernels.jl, CMLookupState) with the
# common-level-anchor block that turns flexible CM into fixed Frechet, per
# docs/FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md and the existing dense-reference
# construction this file's forward/backward math is verified against (cm_frechet_level.jl's
# `fill_frechet_level_columns_from_bins!`, the moments!-time dense fill this kernel replaces).
#
# Inner-solve variable layout: `x = [zeta; lambda_core (ncore-1); lambda_cm ((D-1)*L); lambda_level (L)]`
# -- see wrap_moments_with_cm_frechet_archB's own column-layout docstring (cm_frechet_level.jl),
# `[core | CM | level | gravity]`.
#
# CM block: IDENTICAL math to plain flexible CM (reuses cm_lookup_kernels.jl's free functions
# unchanged: apply_contrast!, suffix_sums!, cumulative_forward_contribution!, build_weighted_
# histogram!, prefix_sums!). Common Frechet's own moment construction (wrap_moments_with_cm_
# frechet_archB) always stores the CM block in the CUMULATIVE basis (same fill_cm_columns_from_
# bins! call flexible CM's own production path uses) -- so, unlike CMLookupState, this file does
# not support an :interval method at all; the CM block always uses the suffix-sum/cumulative
# lookup.
#
# Level block (genuinely new -- no existing production inner-FG kernel to reuse):
#   forward:  sum_l lambda_level[l]*G_level[s,l]
#           = invsqrtD * sum_o P_level[bin(s,o)] - sum_l lambda_level[l]*targets[l]
#     where P_level[k] = sum_{l>=k} lambda_level[l] (suffix sum, same convention as the CM block's
#     P), invsqrtD = 1/sqrt(D), and `sum_o P_level[bin(s,o)]` is computed by the EXISTING
#     `frechet_level_forward_sum!` (cm_frechet_cplus.jl, already used unchanged by the Lfix outer-
#     gradient path at the converged lambda* -- reused here unchanged for the LIVE inner-solve
#     lambda, same underlying identity).
#   backward: g_level[l] = -(invsqrtD/M)*Hpre_total[l] + (targets[l]/M)*sum_dPsi
#     where Hpre_total[l] = sum_{o=1}^D Hpre[o,l], Hpre = the SAME (D,L) prefix-sum-of-weighted-
#     histogram buffer the CM block's own cumulative backward gradient already computes (shared,
#     computed ONCE per callback, not duplicated) via `prefix_sums!(Hpre, h, L)`, and
#     sum_dPsi = sum_s dPsi(arg0)[s] (a free byproduct of the core block's own g[1] computation).
# Full derivation: see docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md (this session).
# ================================================================================================

isdefined(Main, :CompressedFactual) || include(joinpath(@__DIR__, "compressed_moments.jl"))
isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :HessianWeightCache) || include(joinpath(@__DIR__, "operator_hessian_weights.jl"))

"""
    frechet_level_suffix_sums!(P, λ_level)

`P[k] = sum_{l=k}^{L} λ_level[l]` for `k=1:L`, `P[L+1]=0` -- the level block's own (un-origin-
indexed) analogue of `suffix_sums!`. `λ_level` is length `L`, `P` is length `L+1`.
"""
function frechet_level_suffix_sums!(P::AbstractVector{Float64}, λ_level::AbstractVector{Float64})
    L = length(λ_level)
    acc = 0.0
    @inbounds for k in L:-1:1
        acc += λ_level[k]
        P[k] = acc
    end
    P[L + 1] = 0.0
    return P
end


# Harmonization task (2026-07-28): cumulative_backward_gradient_from_prefix! moved to
# cm_lookup_kernels.jl (defined right next to cumulative_backward_gradient!, which now delegates
# to it) -- every current include-list ordering (both equivalence-gate scripts, smoke scripts)
# loads that file no later than this one, so this is a pure move, not a duplication.

"""
    frechet_level_backward_gradient!(g_level, Hpre, D, L, M, invsqrtD, targets, sum_dPsi)

`g_level[l] = -(invsqrtD/M)*sum_{o=1}^D Hpre[o,l] + (targets[l]/M)*sum_dPsi`. O(D*L), reusing the
SAME `Hpre` buffer the CM block's own backward gradient already computed this callback.
"""
function frechet_level_backward_gradient!(g_level::AbstractVector{Float64}, Hpre::AbstractMatrix{Float64},
                                           D::Int, L::Int, M::Int, invsqrtD::Float64,
                                           targets::AbstractVector{Float64}, sum_dPsi::Float64)
    @inbounds for l in 1:L
        acc = 0.0
        for o in 1:D
            acc += Hpre[o, l]
        end
        g_level[l] = -(invsqrtD / M) * acc + (targets[l] / M) * sum_dPsi
    end
    return g_level
end

"""
    frechet_levelpow_prefix_sums!(Q, λ_levelpow)

`Q[k] = sum_{l=1}^{k-1} λ_levelpow[l]` for `k=1:L+1`, `Q[1]=0` -- the levelpow block's own PREFIX
(not suffix) analogue of `frechet_level_suffix_sums!`. The levelpow indicator is `1{Bidx[s,o]>l}`
(reflected vs the plain level block's `1{Bidx[s,o]<=l}`, see `precalc_frechet_levelpow_dense`'s own
docstring, cm_frechet_level.jl), so `sum_l λ_levelpow[l]*1{l<Bidx[s,o]} = Q[Bidx[s,o]]` is exactly
the prefix (not suffix) sum evaluated at the bin index.
"""
function frechet_levelpow_prefix_sums!(Q::AbstractVector{Float64}, λ_levelpow::AbstractVector{Float64})
    L = length(λ_levelpow)
    Q[1] = 0.0
    acc = 0.0
    @inbounds for k in 1:L
        acc += λ_levelpow[k]
        Q[k + 1] = acc
    end
    return Q
end

"""
    frechet_levelpow_forward_sum!(out, bins, D, Q, Pow)

`out[s] = sum_{o=1}^D Q[bins[s,o]] * Pow[s,o]` -- the levelpow block's own Pow-weighted analogue of
`frechet_level_forward_sum!`. `Q` a length-`(L+1)` prefix-sum-extended vector (from
`frechet_levelpow_prefix_sums!`), `Pow` the SAME `W x D` `z^(sigma-1)` matrix the CM block's own
eq.36 two-family extension already uses (`st.Pow`, cm_lookup_kernels.jl).
"""
function frechet_levelpow_forward_sum!(out::AbstractVector{Float64}, bins::AbstractMatrix{<:Unsigned}, D::Int,
                                        Q::AbstractVector{Float64}, Pow::AbstractMatrix{Float64})
    W = length(out)
    @inbounds for s in 1:W
        acc = 0.0
        for o in 1:D
            acc += Q[Int(bins[s, o])] * Pow[s, o]
        end
        out[s] = acc
    end
    return out
end

"""
    CMFrechetLookupState

Per-context mutable bundle, common-Frechet analogue of `CMLookupState`. `obj` is the DENSE
CM+level-augmented `PsiObjectiveBundleImplicit` (needed for its core-column H buffer, `M`,
`Psi!`/`dPsi!`, and to serve the Hessian callback unchanged -- Architecture C's Hessian,
`archC_frechet_hess_cb_builder`, is completely independent of this FG kernel). `R` is the `nO x nO`
orthonormal contrast matrix for the CM block (or `nothing` for `:anchored`) -- the level block is
NEVER rotated by `R` (it is a single un-rotated column per threshold, not part of the CM block's
per-threshold `nO`-dimensional rotation, per the math doc's `[C u]`/`[CR u]` construction).
"""
# port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26 Phase A item 4: `obj::O`
# (was `obj::Any`) per docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md's own recommended
# follow-on ("(a) restructuring the callable so obj's concrete type is captured once at construction
# via a type parameter... rather than read fresh from an Any field on every property access") --
# the prior session measured a real, reproducible 14,066,064 bytes/callback regression here vs
# CMLookupState's near-identical `obj::Any`-fielded but textually SHORTER callable (2,384 bytes),
# suspected to be a devirtualization failure that scales with function-body size/branch count. `O`
# is inferred automatically from the `obj` argument at construction (Julia's default parametric-
# struct outer constructor) -- no forward-type-reference load-order dependency is introduced (this
# was the ONLY reason the field was `Any` in the first place; a type parameter has the same
# load-order-agnostic property, since `O` is resolved from the CONCRETE runtime object passed in,
# never a textual type name that needs `PsiObjectiveBundleImplicit` predeclared).
mutable struct CMFrechetLookupState{O}
    obj::O
    ncore::Int
    ncm_cm::Int         # (D-1)*L, or 2*(D-1)*L when Pow!==nothing (two-family CM sub-block)
    ncm_level::Int      # L, or 2*L when Pow!==nothing (level + levelpow)
    L::Int
    D::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    nbins::Int
    nthreads_use::Int
    level_targets::Vector{Float64}         # length L, CDF-only level targets
    invsqrtD::Float64
    n_fg_calls::Int
    # persistent scratch (Phase 5.5 allocation-hygiene pattern, applied here from the start rather
    # than fixed as a follow-on -- see cm_lookup_kernels.jl's own history for why)
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    cm_contrib::Vector{Float64}
    level_contrib::Vector{Float64}
    xsub::Vector{Float64}
    λmat_block::Matrix{Float64}    # (nO, L)
    λmat_ext::Matrix{Float64}      # (nO, L+1)
    P_level::Vector{Float64}       # (L+1,)
    hist_partials::Vector{Matrix{Float64}}
    hist_h::Matrix{Float64}        # (D, nbins)
    Hpre::Matrix{Float64}          # (D, L) -- SHARED between CM and level backward
    g_block::Matrix{Float64}       # (nO, L)
    g_stored::Matrix{Float64}      # (nO, L)
    g_level::Vector{Float64}       # (L,)
    # 2026-08-06 (levelpow kernel task): two-family CM sub-block scratch, exact analogue of
    # CMLookupState's own `_2`-suffixed fields (cm_lookup_kernels.jl) -- `cm_forward_contribution!`/
    # `cm_transpose_into_g!` (the SAME shared kernels this struct's dual_index!/FG functor already
    # call) read these fields directly whenever `st.Pow!==nothing`. Always allocated (cheap, O(D*L)/
    # O(D*nbins)) regardless of family count, matching CMLookupState's own unconditional-allocation
    # constructor discipline.
    cm_contrib2::Vector{Float64}
    λmat_ext2::Matrix{Float64}
    λmat_block2::Matrix{Float64}
    hist_partials2::Vector{Matrix{Float64}}
    hist_h2::Matrix{Float64}
    Hpre2::Matrix{Float64}
    g_block2::Matrix{Float64}
    g_stored2::Matrix{Float64}
    # 2026-08-06 (levelpow kernel task): the level block's OWN two-family extension --
    # `E_{F*}[z(w)^{sigma-1}*1{z(w)<Z_l}]`, the Pow-weighted, reflected-indicator analogue of the
    # plain `level`/`level_targets`/`level_contrib`/`g_level` block above (see
    # `frechet_levelpow_prefix_sums!`/`frechet_levelpow_forward_sum!`, cm_frechet_level.jl's own
    # `precalc_frechet_levelpow_dense`/`fill_frechet_levelpow_columns_from_bins!` for the dense-
    # reference construction this mirrors). `nothing`/empty when `Pow===nothing` (single-family).
    levelpow_targets::Union{Nothing,Vector{Float64}}   # length L, or nothing (single-family)
    Q_levelpow::Vector{Float64}      # (L+1,) -- prefix-sum-extended lambda_levelpow
    levelpow_contrib::Vector{Float64}   # (M,)
    Hpre_pow::Matrix{Float64}        # (D, L) -- Pow-weighted, REFLECTED prefix sum (see backward pass)
    g_levelpow::Vector{Float64}      # (L,)
    # Phase A item 4 (second half): shared economic operator retrofit, IDENTICAL pattern/rationale
    # to CMLookupState's own core_cf_ref/econ_ws/econ_ws_for/econ_buf/n_dense_econ_fallback fields
    # -- see that struct's docstring for the full contract. `core_cf_ref` defaults to
    # `Ref{Any}(nothing)` for the standalone constructor (dense fallback, byte-identical to
    # pre-port); only `cm_frechet_lookup_production.jl`'s production wiring passes the real
    # `cctx.core_cf_ref` (populated by `wrap_moments_with_cm_frechet_archB`, same box the shared
    # winner-pair Hessian already reads).
    core_cf_ref::Ref{Any}
    econ_ws::Any
    econ_ws_for::Any
    econ_buf::Vector{Float64}
    n_dense_econ_fallback::Int
    # No-moments/no-composite-G task (2026-07-28): same-point cache for the Hessian-weight prep
    # (operator_hessian_weights.jl) -- see that file's own docstring for the full contract.
    hw_cache::HessianWeightCache
    # 2026-08-06 outer-production-closeout task, BUG FIX: `cm_forward_contribution!`/
    # `cm_transpose_into_g!` (cm_lookup_kernels.jl) are SHARED, untyped-`st` kernel functions this
    # struct's own `dual_index!`/FG functor call directly (see those functions' own docstrings:
    # "calls the EXACT SAME shared functions CMLookupState's own dual_index! calls"). The 2026-08-05
    # truncated-power task added an unconditional `fam2 = st.Pow !== nothing` duck-typed read at the
    # top of BOTH shared kernels, and gave `CMLookupState`/`CMMeanZCOperatorState`/
    # `OriginZCOperatorState` a matching `Pow` field -- but never added the same field here, so EVERY
    # call into either shared kernel via a `CMFrechetLookupState` (i.e. every real evaluation under
    # `CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]=:cm_frechet_lookup`, the actual production default)
    # threw `FieldError(CMFrechetLookupState, :Pow)` -- masked by KNITRO.jl's own callback-error
    # swallowing exactly like the CM+ZC missing-Pow= bug (895b99b). 2026-08-06 (levelpow kernel
    # task): common-Fréchet NOW has a real two-family extension (CM sub-block via the shared
    # kernels above, level sub-block via `levelpow_targets`/`Q_levelpow`/etc. above) -- `Pow`
    # (`nothing` for single-family, the real `W x D` `z^(sigma-1)` matrix for two-family) is what
    # gates BOTH extensions, matching `CMLookupState`'s own `Pow`-gated `fam2` convention exactly.
    Pow::Union{Nothing,Matrix{Float64}}
end

function CMFrechetLookupState(obj, ncore::Int, ncm_cm::Int, ncm_level::Int, L::Int, D::Int,
                               origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
                               R::Union{Nothing,Matrix{Float64}}, level_targets::Vector{Float64};
                               nthreads_use::Int = 1, core_cf_ref::Ref{Any} = Ref{Any}(nothing),
                               Pow::Union{Nothing,Matrix{Float64}} = nothing,
                               levelpow_targets::Union{Nothing,Vector{Float64}} = nothing)
    nO = length(origins)
    fam2 = Pow !== nothing
    if fam2
        ncm_cm == 2 * nO * L || error("CMFrechetLookupState: Pow given but ncm_cm=$ncm_cm != 2*nO*L=$(2*nO*L) -- a two-family state needs the doubled CM width")
        ncm_level == 2L || error("CMFrechetLookupState: Pow given but ncm_level=$ncm_level != 2*L=$(2L) -- a two-family state needs the doubled level width (level+levelpow)")
        levelpow_targets !== nothing || error("CMFrechetLookupState: Pow given but levelpow_targets===nothing -- required (no default) for a two-family state")
        length(levelpow_targets) == L || error("CMFrechetLookupState: length(levelpow_targets)=$(length(levelpow_targets)) != L=$L")
    else
        ncm_cm == nO * L || error("CMFrechetLookupState: ncm_cm=$ncm_cm must equal nO*L=$(nO*L) for a single-family state")
        ncm_level == L || error("CMFrechetLookupState: ncm_level=$ncm_level must equal L=$L for a single-family state")
    end
    length(level_targets) == L || error("CMFrechetLookupState: length(level_targets)=$(length(level_targets)) != L=$L")
    M = size(obj.U, 1)
    ncore1 = ncore - 1
    W = size(bins, 1)
    nbins = L + 1
    Dcheck = size(bins, 2)
    Dcheck == D || error("CMFrechetLookupState: D=$D != size(bins,2)=$Dcheck")
    nt = max(1, min(nthreads_use, W))
    hist_partials = [zeros(D, nbins) for _ in 1:nt]
    hist_partials2 = [zeros(D, nbins) for _ in 1:nt]
    CMFrechetLookupState(obj, ncore, ncm_cm, ncm_level, L, D, nO, origins, refIndex1, bins, R,
        nbins, nthreads_use, level_targets, 1.0 / sqrt(D), 0,
        zeros(M), zeros(M), zeros(M), zeros(M),
        zeros(1 + ncore1), zeros(nO, L), zeros(nO, L + 1), zeros(L + 1),
        hist_partials, zeros(D, nbins), zeros(D, L), zeros(nO, L), zeros(nO, L), zeros(L),
        zeros(M), zeros(nO, L + 1), zeros(nO, L), hist_partials2, zeros(D, nbins), zeros(D, L), zeros(nO, L), zeros(nO, L),
        levelpow_targets, zeros(L + 1), zeros(M), zeros(D, L), zeros(L),
        core_cf_ref, nothing, nothing, zeros(M), 0,
        HessianWeightCache(1 + ncore1 + ncm_cm + ncm_level), Pow)
end

"""
    dual_index!(st::CMFrechetLookupState, x) -> st.arg0

Computes `st.arg0 = r = -ζ·1 - E·λ_core - cm_contribution - level_contribution` in place. The
`[E|C]` prefix calls the EXACT SAME shared functions `CMLookupState`'s own `dual_index!` calls
(`economic_forward_into_arg0!`/`cm_forward_contribution!`, cm_lookup_kernels.jl -- see that file's
harmonization-header comment); only the trailing `[F]` level-block extension below is
Fréchet-specific.
"""
function dual_index!(st::CMFrechetLookupState, x::AbstractVector{Float64})
    ncore1 = st.ncore - 1
    fam2 = st.Pow !== nothing
    λ_cm = @view x[2+ncore1:1+ncore1+st.ncm_cm]
    λ_level_full = @view x[2+ncore1+st.ncm_cm:1+ncore1+st.ncm_cm+st.ncm_level]
    λ_level = fam2 ? (@view λ_level_full[1:st.L]) : λ_level_full

    economic_forward_into_arg0!(st, x)
    cm_forward_contribution!(st, λ_cm, :suffix)

    frechet_level_suffix_sums!(st.P_level, λ_level)
    frechet_level_forward_sum!(st.level_contrib, st.bins, st.D, st.P_level)
    const_term = 0.0
    @inbounds for l in 1:st.L
        const_term += λ_level[l] * st.level_targets[l]
    end
    @inbounds for s in 1:length(st.arg0)
        st.arg0[s] -= st.invsqrtD * st.level_contrib[s] - const_term
    end

    if fam2
        λ_levelpow = @view λ_level_full[st.L+1:2*st.L]
        frechet_levelpow_prefix_sums!(st.Q_levelpow, λ_levelpow)
        frechet_levelpow_forward_sum!(st.levelpow_contrib, st.bins, st.D, st.Q_levelpow, st.Pow)
        const_term_pow = 0.0
        @inbounds for l in 1:st.L
            const_term_pow += λ_levelpow[l] * st.levelpow_targets[l]
        end
        @inbounds for s in 1:length(st.arg0)
            st.arg0[s] -= st.invsqrtD * st.levelpow_contrib[s] - const_term_pow
        end
    end
    return st.arg0
end

"""
    (st::CMFrechetLookupState)(x, g=Float64[]) -> f

FG evaluator, same signature/semantics as `obj(x, g)`. `x = [ζ; λ_core; λ_cm; λ_level]`. The
`[E|C]` prefix (forward AND backward) calls the EXACT SAME shared functions `CMLookupState`'s own
FG functor calls (`economic_transpose_into_g1_and_gE!`/`cm_transpose_into_g!`,
cm_lookup_kernels.jl); only the trailing `[F]` level-block extension is Fréchet-specific,
implementing the composition `frechet_cm_fg = economic_fg + cm_fg + frechet_extension_fg`.
"""
function (st::CMFrechetLookupState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = size(obj.U, 1)
    ncore1 = st.ncore - 1

    ζ = x[1]

    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        sum_dPsi = economic_transpose_into_g1_and_gE!(g, st, cf)
        cm_transpose_into_g!(g, st, :suffix, st.D, ncore1, st.ncm_cm, M)   # leaves st.Hpre populated, reused below

        frechet_level_backward_gradient!(st.g_level, st.Hpre, st.D, st.L, M, st.invsqrtD, st.level_targets, sum_dPsi)
        g_level_off = 2 + ncore1 + st.ncm_cm
        @views g[g_level_off:g_level_off+st.L-1] .= st.g_level

        if st.Pow !== nothing
            # levelpow backward: SAME Pow-weighted-histogram + reflected-prefix-sum reuse pattern
            # `cm_transpose_into_g!`'s own eq.36 branch already established (cm_lookup_kernels.jl) --
            # `frechet_level_backward_gradient!` is reused UNCHANGED, only the table it reads differs
            # (Pow-weighted via build_weighted_histogram_pow!, then reflected total-minus-prefix).
            build_weighted_histogram_pow!(st.hist_h2, st.hist_partials2, st.bins, st.arg1, st.Pow, st.D, st.nbins)
            prefix_sums!(st.Hpre_pow, st.hist_h2, st.L)
            @inbounds for o in 1:st.D
                total_o = sum(@view st.hist_h2[o, :])
                for l in 1:st.L
                    st.Hpre_pow[o, l] = total_o - st.Hpre_pow[o, l]
                end
            end
            frechet_level_backward_gradient!(st.g_levelpow, st.Hpre_pow, st.D, st.L, M, st.invsqrtD, st.levelpow_targets, sum_dPsi)
            g_levelpow_off = g_level_off + st.L
            @views g[g_levelpow_off:g_levelpow_off+st.L-1] .= st.g_levelpow
        end
    end

    obj.arg0 .= st.arg0
    _publish_dual_index_cache!(st, x)   # let a same-point Hessian call reuse this r instead of recomputing
    st.n_fg_calls += 1
    return f
end
