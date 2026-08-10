# Genuinely independent dense-G reference implementation for OZC-CROSS (2026-08-09, user request:
# "build the matrix with all the moments and calculate the hessian with matrix multiplication" --
# does the SAME calibration point give the SAME high Delta* through a completely different
# computational path, or is the 7x jump an artifact of the operator/ZCRestrictionOperator FG
# machinery specifically?).
#
# `wrap_moments_with_originzc_cross` below is the OZC-CROSS analog of `wrap_moments_with_originzc`
# (cm_originzc_moments.jl lines 97-161, the base family's DENSE moments! closure, `moment_
# representation=:dense_reference` -- actually production's DEFAULT there, not a rare fallback):
# same core-block/mean-block logic verbatim, pair-block loop swapped for the cross-grid convention
# (cross_pair_level_index/pair_targets(OriginByPowerCrossLayout,...), K_pair^2 blocks instead of
# K_pair). This produces a REAL, dense (W x d) moment Jacobian G every call -- the inner KNITRO solve
# then uses Architecture A's generic dense-BLAS Hessian (`inner_loop_internal_archgeneric`,
# H = G' * diag(...) * G via literal matrix multiplication, cm_hessian_architectures.jl), a
# COMPLETELY DIFFERENT computational path from the operator/ZCRestrictionOperator FG machinery the
# rest of this session's work goes through -- if both agree, the operator-path machinery is not the
# source of the 7x jump.
#
# Deliberately NOT wired into `build_originzc_cross_augmented_obj` (which stays operator-only, per
# its own docstring and the project's no-dense-fallback-in-production rule) -- this is a standalone,
# explicitly-invoked comparison tool only, exactly the role `wrap_moments_with_originzc_dense`
# already plays for the base family ("kept ONLY for before/after correctness and benchmark
# comparison... not used by any production entry point").
isdefined(Main, :cross_pair_level_index) || include(joinpath(@__DIR__, "cm_originzc_cross_moments.jl"))
isdefined(Main, :OriginByPowerCrossLayout) || include(joinpath(@__DIR__, "cm_originzc_cross_target_layout.jl"))

"""
    wrap_moments_with_originzc_cross(core_moments!, ncore_econ, Zraw_all, Zpairraw_all,
                                      layout::OriginByPowerCrossLayout; ctx=nothing,
                                      use_compressed_core=true, core_cf_ref=Ref{Any}(nothing),
                                      skip_fill=false) -> Function

OZC-CROSS analog of `wrap_moments_with_originzc` (cm_originzc_moments.jl). Mean-block loop
byte-identical (unaffected by the cross-pair extension). Pair-block loop iterates
`cross_pair_level_index(layout.K_pair)`'s `K_pair^2` blocks (instead of `1:K_pair` diagonal
levels), using `pair_targets(layout, νfull, klin, D)` -- the SAME target formula already verified
(D4 + D20, residuals at machine precision, D20 Monte-Carlo-average-under-F* check) for the operator
path, just assembled into a dense `G` column range instead of fed to `ZCRestrictionOperator`.
"""
function wrap_moments_with_originzc_cross(core_moments!::Function, ncore_econ::Int,
                                           Zraw_all::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}},
                                           layout::OriginByPowerCrossLayout;
                                           ctx = nothing, use_compressed_core::Bool = true,
                                           core_cf_ref::Ref{Any} = Ref{Any}(nothing),
                                           skip_fill::Bool = false)
    pregrav = ncore_econ - 1
    D = size(Zraw_all[1], 2)
    K_mean = layout.K_mean
    K_pair = layout.K_pair
    n_mean_total = K_mean * D
    npair = D * (D - 1) ÷ 2
    levels = cross_pair_level_index(K_pair)
    n_pair_total = length(levels) * npair
    n_eta_total = n_eta(layout)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
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
            cf = cf_build(collect(θ_econ), ctx; check_ties = false)
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
        for (klin, (k1, k2)) in enumerate(levels)
            cols = mean_end+(klin-1)*npair+1 : mean_end+klin*npair
            dest = @view G[:, cols]
            νprod_klin = pair_targets(layout, νfull, klin, D)
            pair_columns!(dest, (@view Zpairraw_all[klin][1:n, :]), νprod_klin)
        end
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    build_originzc_cross_augmented_obj_dense(ctx, CS, layout::OriginByPowerCrossLayout) -> NamedTuple

Dense-reference (`PsiObjectiveBundleImplicit`) analog of `build_originzc_cross_augmented_obj`, for
correctness comparison ONLY -- no `aml`/Variant D support (matches the base family's own restriction
of row-omission to `:operator`, `build_originzc_augmented_obj`'s dense_reference branch). Returns the
same NamedTuple shape as `build_originzc_cross_augmented_obj` so `build_originzc_core_hess_ctx`
(fg_backend=:dense_reference) works unchanged.
"""
function build_originzc_cross_augmented_obj_dense(ctx, CS, layout::OriginByPowerCrossLayout)
    obj0 = ctx.obj
    ncore_econ = obj0.d
    D = ctx.D
    K_mean = layout.K_mean; K_pair = layout.K_pair

    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = ctx.μHat)
    Zpairraw_all = build_raw_cross_pair_matrix_levels(Zraw_all, K_pair)

    npair = div(D * (D - 1), 2)
    n_mean = K_mean * D
    n_pair = K_pair^2 * npair
    @assert n_mean + n_pair == n_originzc_cross_moments(D, K_mean, K_pair)

    d_new = ncore_econ + n_mean + n_pair
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair
    core_cf_ref = Ref{Any}(nothing)

    moments_originzc_cross! = wrap_moments_with_originzc_cross(obj0.moments!, ncore_econ, Zraw_all, Zpairraw_all, layout;
        ctx = ctx, core_cf_ref = core_cf_ref, skip_fill = false)
    moments_originzc_cross_skip! = wrap_moments_with_originzc_cross(obj0.moments!, ncore_econ, Zraw_all, Zpairraw_all, layout;
        ctx = ctx, core_cf_ref = core_cf_ref, skip_fill = true)

    obj_oz = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_originzc_cross!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_oz.outer_constr_index == obj_oz.d

    return (obj_cm = obj_oz, ncore = ncore_econ,
            Zraw_all = Zraw_all, Zpairraw_all = Zpairraw_all, layout = layout,
            K_mean = K_mean, K_pair = K_pair, n_mean = n_mean, n_pair = n_pair,
            ncore_econ = ncore_econ, core_cf_ref = core_cf_ref, moments_skip! = moments_originzc_cross_skip!,
            aml = nothing)
end

"""
    build_originzc_cross_production_context_dense(ctx, CS, layout::OriginByPowerCrossLayout) -> (ctx_cm, aug, octx)

Dense-reference analog of `build_originzc_cross_production_context`, `fg_backend=:dense_reference`.
"""
function build_originzc_cross_production_context_dense(ctx, CS, layout::OriginByPowerCrossLayout)
    println(stdout, "cm_restriction_basis [OZC-CROSS-DENSE] = none (dense-reference comparison build, NOT production path)")
    flush(stdout)
    aug = build_originzc_cross_augmented_obj_dense(ctx, CS, layout)
    octx = build_originzc_core_hess_ctx(aug, ctx; fg_backend = :dense_reference)
    ctx_cm = merge(ctx, (obj = aug.obj_cm, octx = octx))
    return (ctx_cm = ctx_cm, aug = aug, octx = octx)
end
