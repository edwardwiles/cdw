# ============================================================================
# Origin-specific pairwise-zero-covariance restriction, WITHOUT common
# marginals. See docs/ORIGIN_SPECIFIC_ZC_MATH_NOTE_2026-07-23.md for the full
# derivation and docs/CM_MEANZC math note conventions this file inherits.
#
# Reuses UNCHANGED from cm_meanzc_moments.jl: `packed_pair_index`,
# `pair_lin_to_oi`/`pair_oi_to_lin`, `build_raw_mean_pair_matrices`,
# `build_raw_mean_pair_matrix_levels` (all purely about the theta-independent
# raw draw data U.^k / pair products -- they have no dependency on WHICH
# target layout is used, so nothing here re-derives them).
#
# NOT reused: `wrap_moments_with_cm_meanzc`/`build_cm_meanzc_augmented_obj`
# (both always splice in a CM-grid block -- this arm has none at all) and
# `d_delta_dual_d_nu_vec` (hard-codes the shared-nu block-diagonal structure).
# New sibling functions below add the no-CM, origin-by-power arm via
# `MeanZCTargetLayout` dispatch (cm_originzc_target_layout.jl), reusing
# `mean_columns_direct`/`pair_columns`'s EXISTING Float64-scalar methods only
# for the SharedByPowerLayout special case (unused in production for this
# arm, kept for the mean-only-arm implementation-equivalence test, task
# brief Section 3) via new AbstractVector-argument methods added here.
# ============================================================================

isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))   # true no-H operator bundle (2026-07-28 continuation): OperatorPsiBundle/prime_operator!, load-bearing for build_originzc_augmented_obj below
#
# Column layout (no CM-grid block at all):
#     [ economic (ncore_econ-1) | mean_1(D) ... mean_{K_mean}(D)
#       | pair_1(npair) ... pair_{K_pair}(npair) | gravity ]
# ============================================================================

using LinearAlgebra: dot

"""
    mean_columns_direct(Z, νtargets) -> Matrix   (W x D)

Direct basis, origin-specific target: `g_o(s;ν) = Z_so - νtargets[o]` for
every origin `o` (new AbstractVector method, dispatched alongside the
existing `Float64`-scalar `mean_columns_direct`/`mean_columns_anchored` in
cm_meanzc_moments.jl -- does not modify or shadow them). `:anchored` has no
origin-specific analog (see math note Section 3) and is not provided here.
"""
mean_columns_direct(Z::AbstractMatrix{Float64}, νtargets::AbstractVector{Float64}) = Z .- νtargets'

"""
    pair_columns(Zpair, νprod) -> Matrix   (W x npair)

`g_op(s;ν) = Zpair_s,op - νprod[op]` where `νprod[op] = nu_{o,k}*nu_{p,k}`
for pair `(o,p)` (new AbstractVector method alongside the existing
`Float64`-scalar `pair_columns` in cm_meanzc_moments.jl).
"""
pair_columns(Zpair::AbstractMatrix{Float64}, νprod::AbstractVector{Float64}) = Zpair .- νprod'

"""
    mean_columns_direct!(dest, Z, νtargets)
    pair_columns!(dest, Zpair, νprod)

In-place, no-intermediate-allocation analogs of the `AbstractVector`-target
methods above (mirrors `cm_meanzc_moments.jl`'s own `mean_columns_direct!`/
`pair_columns!`). `dest` is expected to be a view directly into the
destination moment matrix `G`.
"""

function mean_columns_direct!(dest::AbstractMatrix{Float64}, Z::AbstractMatrix{Float64}, νtargets::AbstractVector{Float64})
    @. dest = Z - νtargets'
    return dest
end
function pair_columns!(dest::AbstractMatrix{Float64}, Zpair::AbstractMatrix{Float64}, νprod::AbstractVector{Float64})
    @. dest = Zpair - νprod'
    return dest
end

"""
    n_originzc_moments(D, K_mean, K_pair) -> Int

Total new inner moments for the no-CM origin-specific arm:
`K_mean*D + K_pair*D(D-1)/2` -- same formula as `n_meanzc_moments` (the
CM-side moment COUNT is unaffected by which target layout produced the
column values), reused directly rather than redefined.
"""
n_originzc_moments(D::Int, K_mean::Int, K_pair::Int) = n_meanzc_moments(D, K_mean, K_pair)

"""
    wrap_moments_with_originzc(core_moments!, ncore_econ, Zraw_all, Zpairraw_all, layout) -> Function

Returns a `moments!`-signature closure `(K, G, θ_ext, U, obj) -> nothing`
producing columns `[economic (ncore_econ-1) | mean_1..mean_{K_mean} |
pair_1..pair_{K_pair} | gravity]` -- NO CM-grid block (contrast with
`wrap_moments_with_cm_meanzc`, which always splices one in). `K_mean =
length(Zraw_all)`, `K_pair = length(Zpairraw_all)`, both read off `layout`.

`θ_ext` MUST be `vcat(θ_econ, ν_1,...,ν_{n_eta(layout)})` -- `νfull` is
ALREADY exponentiated (nu, not eta -- same convention `cm_meanzc_moments.jl`
uses: the KNITRO outer vector carries eta=log(nu), but `θ_ext` fed to
`moments!` carries nu itself, exponentiated once per outer evaluation by the
caller, matching `cb_F!`/`cb_G!`'s own `νvec = exp.(w[...])` convention in
cm_checkpoint.jl).
"""
function wrap_moments_with_originzc(core_moments!::Function, ncore_econ::Int,
                                     Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}},
                                     layout::MeanZCTargetLayout;
                                     ctx = nothing, use_compressed_core::Bool = true,
                                     core_cf_ref::Ref{Any} = Ref{Any}(nothing),
                                     skip_fill::Bool = false)   # Legacy-H cleanup (2026-07-28):
                                     # mirrors wrap_moments_with_cm_archB/wrap_moments_with_cm_meanzc's
                                     # now-fixed skip_fill kwarg -- economic-block-only skip.
    pregrav = ncore_econ - 1
    D = size(Zraw_all[1], 2)
    K_mean = layout.K_mean
    K_pair = layout.K_pair
    n_mean_total = K_mean * D
    npair = D * (D - 1) ÷ 2
    n_pair_total = K_pair * npair
    n_eta_total = n_eta(layout)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
    # port/shared-winner-pair-core-hessian-production-2026-07-25 (task §4.4): origin-ZC's H_EE
    # needs the SAME shared winner-pair backend, which needs a `CompressedFactual` for the current
    # theta -- built here, mirroring `wrap_moments_with_cm_archB`/`wrap_moments_with_cm_meanzc`,
    # and published via `core_cf_ref` for `archA_partitioned_hess_cb_builder` to pick up.
    can_compress = ctx !== nothing && use_compressed_core
    return function (K, G, θ_ext, U, obj)
        n = size(U, 1)
        θ_econ = @view θ_ext[1:end-n_eta_total]
        νfull = @view θ_ext[end-n_eta_total+1:end]
        if size(Gtmp_cache[], 1) != n
            Gtmp_cache[] = Matrix{Float64}(undef, n, ncore_econ)
        end
        G_tmp = Gtmp_cache[]
        if can_compress
            # No-moments/no-composite-G task (2026-07-28): check_ties=false -- see the identical
            # change/rationale in cm_hessian_architectures.jl::wrap_moments_with_cm_archB.
            cf = cf_build(collect(θ_econ), ctx; check_ties = false)   # Phase E remediation (2026-07-26): reuses ctx.cf_workspace when attached
            if !skip_fill
                materialize_dense_factual_structured!(@view(G_tmp[:, 1:pregrav]), cf)
            end
            grav_raw = compressed_gravity_raw(collect(θ_econ), ctx)
            fill_gravity_column_into!(@view(G_tmp[:, ncore_econ]), grav_raw, ctx, ncore_econ)
            fill_K_directgp!(K, collect(θ_econ), ctx)
            core_cf_ref[] = cf
        else
            core_moments!(K, G_tmp, θ_econ, U, obj)
            core_cf_ref[] = :compressed_state_unavailable
        end
        if !skip_fill
            @views G[:, 1:pregrav] .= G_tmp[:, 1:pregrav]
        end
        for k in 1:K_mean
            cols = pregrav+(k-1)*D+1 : pregrav+k*D
            dest = @view G[:, cols]
            νo_k = mean_targets(layout, νfull, k, D)
            mean_columns_direct!(dest, (@view Zraw_all[k][1:n, :]), νo_k)
        end
        mean_end = pregrav + n_mean_total
        for k in 1:K_pair
            cols = mean_end+(k-1)*npair+1 : mean_end+k*npair
            dest = @view G[:, cols]
            νprod_k = pair_targets(layout, νfull, k, D)
            pair_columns!(dest, (@view Zpairraw_all[k][1:n, :]), νprod_k)
        end
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    wrap_moments_with_originzc_dense(core_moments!, ncore_econ, Zraw_all, Zpairraw_all, layout) -> Function

Slow dense reference path, preserved byte-for-byte from the pre-2026-07-24
Phase B implementation (fresh `G_tmp` allocation every call, mean/pair
blocks built via the allocating `AbstractVector`-target
`mean_columns_direct`/`pair_columns` then copied into `G`). Kept ONLY for
before/after correctness and benchmark comparison against
`wrap_moments_with_originzc` above -- not used by any production entry point.
"""
function wrap_moments_with_originzc_dense(core_moments!::Function, ncore_econ::Int,
                                           Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}},
                                           layout::MeanZCTargetLayout)
    pregrav = ncore_econ - 1
    D = size(Zraw_all[1], 2)
    K_mean = layout.K_mean
    K_pair = layout.K_pair
    n_mean_total = K_mean * D
    npair = D * (D - 1) ÷ 2
    n_pair_total = K_pair * npair
    n_eta_total = n_eta(layout)
    return function (K, G, θ_ext, U, obj)
        n = size(U, 1)
        θ_econ = @view θ_ext[1:end-n_eta_total]
        νfull = @view θ_ext[end-n_eta_total+1:end]
        G_tmp = similar(G, n, ncore_econ)
        core_moments!(K, G_tmp, θ_econ, U, obj)
        @views G[:, 1:pregrav] .= G_tmp[:, 1:pregrav]
        for k in 1:K_mean
            cols = pregrav+(k-1)*D+1 : pregrav+k*D
            νo_k = mean_targets(layout, νfull, k, D)
            @views G[:, cols] .= mean_columns_direct(Zraw_all[k][1:n, :], νo_k)
        end
        mean_end = pregrav + n_mean_total
        for k in 1:K_pair
            cols = mean_end+(k-1)*npair+1 : mean_end+k*npair
            νprod_k = pair_targets(layout, νfull, k, D)
            @views G[:, cols] .= pair_columns(Zpairraw_all[k][1:n, :], νprod_k)
        end
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    build_originzc_augmented_obj(ctx, CS, layout::MeanZCTargetLayout) -> NamedTuple

No-CM analog of `build_cm_meanzc_augmented_obj`: constructs a NEW
`PsiObjectiveBundleImplicit` restricted ONLY by the normal production
economic/gravity moments plus origin-specific mean/pairwise-ZC moments --
NO finite-grid CM CDF contrasts, NO CM tail-moment contrasts, NO common
marginal equality are constructed anywhere in this function (contrast with
`build_cm_meanzc_augmented_obj`, which always calls
`precalc_common_marginals_cdf`). `ctx.obj` is left untouched.

Returns the same NamedTuple shape as `build_cm_meanzc_augmented_obj` MINUS
the CM-specific fields (`CM, z, origins, ncm, L, contrasts, refIndex1`),
PLUS `layout` itself (so callers can recover `K_mean`/`K_pair`/target-index
mapping without re-threading them separately).
"""
function build_originzc_augmented_obj(ctx, CS, layout::MeanZCTargetLayout;
        moment_representation::Symbol = :dense_reference)   # true no-H operator bundle (2026-07-28
        # continuation): :dense_reference (default, unchanged) | :operator (explicit opt-in).
    layout isa OriginByPowerLayout || layout isa SharedByPowerLayout ||
        error("build_originzc_augmented_obj: unsupported layout type $(typeof(layout))")

    obj0 = ctx.obj
    ncore_econ = obj0.d
    D = ctx.D
    K_mean = layout.K_mean; K_pair = layout.K_pair

    Zraw_all, Zpairraw_all = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, K_pair)
    npair = div(D * (D - 1), 2)
    n_mean = K_mean * D
    n_pair = K_pair * npair
    @assert n_mean + n_pair == n_originzc_moments(D, K_mean, K_pair)

    d_new = ncore_econ + n_mean + n_pair
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair
    core_cf_ref = Ref{Any}(nothing)
    moments_originzc_skip! = nothing
    if moment_representation === :dense_reference
        moments_originzc! = wrap_moments_with_originzc(obj0.moments!, ncore_econ, Zraw_all, Zpairraw_all, layout;
            ctx = ctx, core_cf_ref = core_cf_ref, skip_fill = false)
        # Legacy-H cleanup (2026-07-28): second closure, same shared core_cf_ref, skip_fill=true --
        # mirrors build_cm_meanzc_augmented_obj's identical dual-closure pattern.
        moments_originzc_skip! = wrap_moments_with_originzc(obj0.moments!, ncore_econ, Zraw_all, Zpairraw_all, layout;
            ctx = ctx, core_cf_ref = core_cf_ref, skip_fill = true)

        obj_oz = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
            γ = obj0.γ, (moments!) = moments_originzc!, moments_jacobian! = error,
            d = d_new, outer_constr_index = outer_constr_index_new,
            inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
            l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
            use_cached_x = obj0.use_cached_x,
            threshold_state = obj0.threshold_state,   # 2026-07-24 release fix: was defaulting to Inf (disabled) on every rebuild
            outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
            needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
        @assert obj_oz.outer_constr_index == obj_oz.d
    elseif moment_representation === :operator
        obj_oz = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
            γ = obj0.γ, l = obj0.l, outer_constr_index = outer_constr_index_new,
            inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
            U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
            use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
            inner_loop_opt = obj0.inner_loop_opt)
    else
        error("build_originzc_augmented_obj: moment_representation must be :operator or :dense_reference, got :$moment_representation")
    end

    return (obj_cm = obj_oz, ncore = ncore_econ,
            Zraw_all = Zraw_all, Zpairraw_all = Zpairraw_all, layout = layout,
            K_mean = K_mean, K_pair = K_pair, n_mean = n_mean, n_pair = n_pair,
            ncore_econ = ncore_econ, core_cf_ref = core_cf_ref, moments_skip! = moments_originzc_skip!)
end

"""
    d_delta_dual_d_eta_origin_vec(λstar, aug, νfull::Vector{Float64}; mean_m) -> Vector{Float64}

Analytic envelope derivative of `Delta_dual` w.r.t. every `eta_{o,k}`
(length `n_eta(aug.layout)`), math note Section 4:

    d(Delta_dual)/d(nu_{o,k}) = -mean_m * ( lambda_mean,o,k*
                                            + sum_{p != o, k<=K_pair} nu_{p,k} * lambda_pair,op,k* )
    d(Delta_dual)/d(eta_{o,k}) = nu_{o,k} * d(Delta_dual)/d(nu_{o,k})

`λstar` is the FULL inner dual vector (`base.λstar`). Each level's mean/pair
slice is located via `aug.ncore_econ`/`aug.n_mean`/`aug.n_pair`, matching
`wrap_moments_with_originzc`'s column layout exactly. Reduces, under
`SharedByPowerLayout`, to `d_delta_dual_d_eta_nu_vec`'s existing formula
(verified both algebraically, math note Section 4, and by finite differences,
test_cm_originzc_pure_moments.jl).
"""
function d_delta_dual_d_eta_origin_vec(λstar::AbstractVector{Float64}, aug, νfull::AbstractVector{Float64}; mean_m::Float64)
    layout = aug.layout
    K_mean = layout.K_mean; K_pair = layout.K_pair
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    length(νfull) == n_eta(layout) || error("d_delta_dual_d_eta_origin_vec: length(νfull)=$(length(νfull)) != n_eta(layout)=$(n_eta(layout))")
    ncore_econ = aug.ncore_econ
    mean_start = ncore_econ
    pair_start0 = ncore_econ + K_mean * D
    d_nu = zeros(n_eta(layout))
    pairs = packed_pair_index(D)
    for k in 1:K_mean
        λ_mean_k = @view λstar[mean_start+(k-1)*D : mean_start+k*D-1]
        for o in 1:D
            idx = target_index(layout, o, k)
            d_nu[idx] -= λ_mean_k[o]
        end
        if k <= K_pair
            λ_pair_k = @view λstar[pair_start0+(k-1)*npair : pair_start0+k*npair-1]
            for (j, (o, p)) in enumerate(pairs)
                idx_o = target_index(layout, o, k)
                idx_p = target_index(layout, p, k)
                nu_o = νfull[idx_o]; nu_p = νfull[idx_p]
                d_nu[idx_o] -= nu_p * λ_pair_k[j]
                d_nu[idx_p] -= nu_o * λ_pair_k[j]
            end
        end
    end
    d_nu .*= mean_m
    return νfull .* d_nu   # chain rule, nu = exp(eta)
end

"""
    originzc_fixed_contribution(base, aug, νfull::Vector{Float64}) -> Vector{Float64}

No-CM analog of `meanzc_fixed_contribution` (cm_meanzc_production.jl):
`out[s] = sum_k lambda_mean,k*'*(Z_k,s - nutargets_k) + sum_k lambda_pair,k*'*(Zpair_k,s - nuprod_k)`
for every draw `s`, at fixed `νfull` -- the SAME "fold the mean/pair fixed
contribution into q0 once per outer evaluation" pattern
`meanzc_fixed_contribution` establishes, generalized to per-origin targets
via `mean_targets`/`pair_targets`. There is no CM-grid block to fold in for
this arm (contrast with `build_lfix_base_cache_cm_meanzc`, which folds BOTH
`cm_fixed_contribution_meanzc_layout` and `meanzc_fixed_contribution`).
"""
function originzc_fixed_contribution(base, aug, νfull::AbstractVector{Float64})
    layout = aug.layout
    K_mean = layout.K_mean; K_pair = layout.K_pair
    length(νfull) == n_eta(layout) || error("originzc_fixed_contribution: length(νfull)=$(length(νfull)) != n_eta(layout)=$(n_eta(layout))")
    ncore_econ = aug.ncore_econ
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    @assert length(base.λstar) >= ncore_econ - 1 + aug.n_mean + aug.n_pair "base.λstar too short for aug's (ncore_econ,n_mean,n_pair) -- was base solved against aug.obj_cm?"
    W = size(aug.Zraw_all[1], 1)
    out = zeros(W)
    mean_start = ncore_econ
    pair_start0 = ncore_econ + K_mean * D
    for k in 1:K_mean
        λ_mean_k = @view base.λstar[mean_start+(k-1)*D : mean_start+k*D-1]
        out .+= aug.Zraw_all[k] * λ_mean_k
        out .-= dot(mean_targets(layout, νfull, k, D), λ_mean_k)
    end
    for k in 1:K_pair
        λ_pair_k = @view base.λstar[pair_start0+(k-1)*npair : pair_start0+k*npair-1]
        out .+= aug.Zpairraw_all[k] * λ_pair_k
        out .-= dot(pair_targets(layout, νfull, k, D), λ_pair_k)
    end
    return out
end

"""
    recovered_mean_residuals_origin(m_weights, Z, νtargets) -> Vector{Float64}   (length D)
    recovered_pair_residuals_origin(m_weights, Zpair, νprod) -> Vector{Float64}   (length npair)

Origin-specific analogs of `recovered_mean_residuals`/`recovered_pair_residuals`
(cm_meanzc_moments.jl), taking a per-origin target vector instead of one
shared scalar. Should be ≈0 at a verified solution.
"""
function recovered_mean_residuals_origin(m_weights::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, νtargets::AbstractVector{Float64})
    W = length(m_weights)
    D = size(Z, 2)
    out = Vector{Float64}(undef, D)
    @inbounds for o in 1:D
        out[o] = dot(m_weights, @view(Z[:, o])) / W - νtargets[o]
    end
    return out
end
function recovered_pair_residuals_origin(m_weights::AbstractVector{Float64}, Zpair::AbstractMatrix{Float64}, νprod::AbstractVector{Float64})
    W = length(m_weights)
    npair = size(Zpair, 2)
    out = Vector{Float64}(undef, npair)
    @inbounds for k in 1:npair
        out[k] = dot(m_weights, @view(Zpair[:, k])) / W - νprod[k]
    end
    return out
end
