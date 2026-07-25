# ============================================================================
# Q2 (whitened) fast path: task brief Part III.6's literal instruction --
# Q2 is NOT a fourth independent kernel. The forward (moment) function uses
# the literal dense CM_q2 = CM_q1 * T2 matrix (built ONCE at context-build
# time, O(W*ncm*ncm2) -- cheap as a one-time cost, unlike a per-callback
# operation) through the ALREADY-GENERIC `wrap_moments_with_cm`. Only the
# HESSIAN needs a special path, and it is exactly the congruence transform
# `H_q2 = Tfull' H_q1 Tfull` (`Tfull = blockdiag(I_NCORE, T2)`) applied to
# the Q1 STRUCTURED Hessian's output (cm_frechet_bases_structured.jl) --
# reuses that kernel's O(W*L) table-building pass unchanged; only adds a
# cheap O(ncm^2) dense congruence multiply per callback.
# ============================================================================

"""
    build_cm_frechet_q2_augmented_obj(ctx, CS, targets; feature_set=:cdf_only, contrasts=:anchored,
                                       refIndex1=ctx.γ.refIndex1, ridge=1e-10) -> NamedTuple

Builds the Q2 (whitened) augmented objective directly (generic dense-Hessian-compatible via
`wrap_moments_with_cm`, exactly as `build_cm_frechet_augmented_obj_basis` does for Q0/Q1) plus the
`T2`/`Tfull` transform needed by the fast Hessian path below.
"""
function build_cm_frechet_q2_augmented_obj(ctx, CS, targets::FrechetReferenceTargets;
                                            feature_set::Symbol = :cdf_only, contrasts::Symbol = :anchored,
                                            refIndex1::Int = ctx.γ.refIndex1, ridge::Float64 = 1e-10)
    aug_q1 = build_cm_frechet_augmented_obj_basis(ctx, CS, targets; basis = :interval, feature_set = feature_set,
                                                    contrasts = contrasts, refIndex1 = refIndex1)
    T2, meta = frechet_full_whitening_transform(ctx, targets; feature_set = feature_set, contrasts = contrasts,
                                                 refIndex1 = refIndex1, ridge = ridge)
    CM_q2 = aug_q1.CM * T2
    ncm = size(CM_q2, 2)
    @assert ncm == aug_q1.ncm

    obj0 = ctx.obj
    ncore = obj0.d
    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    moments_cm! = wrap_moments_with_cm(obj0.moments!, ncore, CM_q2)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, CM = CM_q2, aug_q1 = aug_q1, T2 = T2, ncore = ncore, ncm = ncm,
            L = aug_q1.L, feature_set = feature_set, contrasts = contrasts, refIndex1 = refIndex1,
            targets = targets, whiten_meta = meta)
end

"""
    build_cm_frechet_q2_bin_ctx(ctx, aug_q2) -> (fctx_q1=..., Tfull=Matrix, ncore=Int, ncm=Int)

Precomputation for the fast Q2 Hessian: the Q1 bin-ctx (unchanged) plus the FULL
`(NCORE+ncm) x (NCORE+ncm)` congruence matrix `Tfull = blockdiag(I_NCORE, T2)`.
"""
function build_cm_frechet_q2_bin_ctx(ctx, aug_q2)
    aug_q1 = aug_q2.aug_q1
    aug_q1_cdf_only = aug_q2.feature_set === :cdf_only
    fctx_q1 = build_cm_frechet_interval_bin_ctx(ctx, aug_q1)
    NCORE = aug_q2.ncore; ncm = aug_q2.ncm
    n = NCORE + ncm
    Tfull = zeros(n, n)
    for i in 1:NCORE
        Tfull[i, i] = 1.0
    end
    Tfull[NCORE+1:end, NCORE+1:end] .= aug_q2.T2
    return (fctx_q1 = fctx_q1, Tfull = Tfull, ncore = NCORE, ncm = ncm)
end

"""
    hessian_cm_frechet_q2_structured!(h, obj, q2ctx)

Fast Q2 Hessian: computes Q1's structured Hessian into a scratch packed vector, unpacks to dense,
applies the congruence transform `Tfull' * H_q1 * Tfull`, repacks upper-triangular.
"""
function hessian_cm_frechet_q2_structured!(h, obj, q2ctx)
    n = q2ctx.ncore + q2ctx.ncm
    nh = div(n * (n + 1), 2)
    h_q1 = Vector{Float64}(undef, nh)
    hessian_cm_frechet_interval_structured!(h_q1, obj, q2ctx.fctx_q1)

    Hd = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            Hd[i, j] = h_q1[k]; Hd[j, i] = h_q1[k]
            k += 1
        end
    end
    Hq2 = q2ctx.Tfull' * Hd * q2ctx.Tfull
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            h[k] = 0.5 * (Hq2[i, j] + Hq2[j, i])
            k += 1
        end
    end
    return h
end

"Architecture C's hess_cb_builder for the Q2 (whitened) fixed-Fréchet basis."
function archC_frechet_q2_hess_cb_builder(q2ctx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet_q2" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_q2_structured!(evalResult.hess, o, q2ctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
