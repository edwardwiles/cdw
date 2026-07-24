# ============================================================================
# Continuation (branch diag/fullA-d4-exact-cm-hessian-arch): Hessian
# architecture comparison for the CM-augmented inner CC dual solve, D=4.
#
# BASELINE, unchanged (Architecture A): `build_cm_augmented_obj`'s resulting
# `obj_cm` fed straight into the EXISTING generic dense-BLAS
# `cc_algo/PsiObjectiveBundle.jl::hessian!` -- that function already operates
# on whatever `H`/`d`/`outer_constr_index` the bundle carries, so it needs NO
# code change to work for the CM-augmented moment layout; this file does not
# redefine it, only re-exercises it via the standard KNITRO wiring
# (`inner_loop_KNITRO_profiled` from oracle_fast.jl, included by callers).
#
# This file adds THREE additive architectures (B, C, D) that all produce the
# SAME packed upper-triangular Hessian `cc_algo/PsiObjectiveBundle.jl::hessian!`
# does (validated in c13_validate_hessian_archs.jl), for the SAME CM-augmented
# `obj_cm`, so any of them can be swapped in as a KNITRO Hessian callback with
# zero change to FG callback, bounds, or complementarity wiring.
#
# ---- Notation (see docs/fullA_cm_hessian_architecture_report.md sec 2) ----
# The inner Newton Hessian is w.r.t. x=(zeta,lambda), dimension
# n = outer_constr_index = NCORE + ncm, where:
#   NCORE = aug.ncore  (E-block width: 1 "ones"/zeta column + (NCORE-1)
#           pregrav economic moment columns -- gravity itself is excluded,
#           it is the sole OUTER-only column, see common_marginals_moments.jl)
#   ncm   = aug.ncm    (common-marginals block width, (D-1)*L for
#           include_truncated_moment=false)
# E = H[:, 2:1+NCORE]         (W x NCORE)   -- "existing economic moments" (+intercept)
# C = H[:, 2+NCORE:1+NCORE+ncm]  (W x ncm)  -- common-marginals block
# H_partition = [[H_EE H_EC];[H_EC' H_CC]], H_EE=(1/M)E'DE, H_EC=(1/M)E'DC,
# H_CC=(1/M)C'DC, D=diag(w), w=ddPsi!(arg0) (obj.arg2 after `ddPsi!(arg2,arg0)`).
# ============================================================================

using LinearAlgebra: BLAS, mul!

# ----------------------------------------------------------------------------
# Shared: per-draw bin indices w.r.t. the SAME thresholds `z` that
# `precalc_common_marginals_cdf` uses (reused, not re-derived -- see
# `build_cm_bin_ctx` below, which calls `precalc_common_marginals_cdf` itself
# so `z` is byte-identical to what produced the reference dense CM matrix).
# bin(u) = searchsortedfirst(z, u) in {1,...,L+1}; satisfies, for l in 1:L,
# 1{u<=z_l} == (bin(u) <= l)  (z sorted ascending, verified in the header
# comment derivation, docs/fullA_cm_hessian_architecture_report.md sec 2).
# ----------------------------------------------------------------------------
function compute_bin_indices(U::AbstractMatrix{Float64}, z::AbstractVector{Float64})
    W, D = size(U)
    Bidx = Matrix{Int}(undef, W, D)
    @inbounds for x in 1:D, s in 1:W
        Bidx[s, x] = searchsortedfirst(z, U[s, x])
    end
    return Bidx
end

# ============================================================================
# ARCHITECTURE B: chunked/cached common-block materialization.
#
# Avoids (1) `wrap_moments_with_cm`'s fresh-`similar`-every-call `G_tmp`
# (cached and reused across calls instead) and (2) storing/copying from a
# persistent dense W x ncm CM matrix (built fresh from bin indices, in row
# chunks, directly into the destination G view instead). The Hessian
# CONTRACTION itself is untouched -- Architecture B's obj still dispatches to
# the same generic `hessian!` as Architecture A; only moment CONSTRUCTION
# differs. Mathematically identical G matrix content to Architecture A's (up
# to floating point summation order in the chunked recompute vs the
# precalc'd-once-then-copied path) -- validated, not assumed.
# ============================================================================

"Fill `Gdest` (a W x ncm view, threshold-major layout matching precalc_common_marginals_cdf) directly from bin indices, in row-chunks of `chunk_size`. Never materializes a persistent W x ncm matrix."
function fill_cm_columns_from_bins!(Gdest::AbstractMatrix{Float64}, Bidx::AbstractMatrix{Int},
                                     origins::Vector{Int}, refIndex1::Int, L::Int,
                                     R::Union{Nothing,Matrix{Float64}}; chunk_size::Int = 2000)
    W = size(Gdest, 1)
    nO = length(origins)
    @assert size(Gdest, 2) == L * nO
    cs = min(chunk_size, W)
    buf = Matrix{Float64}(undef, cs, nO)
    start = 1
    @inbounds while start <= W
        stop = min(start + cs - 1, W)
        rows = start:stop
        n = length(rows)
        bview = @view buf[1:n, :]
        for l in 1:L
            for (oi, o) in enumerate(origins)
                for (ridx, s) in enumerate(rows)
                    bview[ridx, oi] = Float64(Bidx[s, o] <= l) - Float64(Bidx[s, refIndex1] <= l)
                end
            end
            cols = (l - 1) * nO + 1 : l * nO
            if R === nothing
                @views Gdest[rows, cols] .= bview
            else
                @views Gdest[rows, cols] .= bview * R
            end
        end
        start = stop + 1
    end
    return nothing
end

"""
    wrap_moments_with_cm_archB(core_moments!, ncore_full, Bidx, origins, refIndex1, L, R; chunk_size)

Architecture B analogue of `wrap_moments_with_cm` (common_marginals_moments.jl).
Same external contract (a `moments!`-signature closure), same column layout,
but (a) caches the `G_tmp` scratch buffer across calls (keyed on `n=size(U,1)`,
reallocated only if `n` changes) instead of `similar`-ing a fresh one every
call, and (b) builds the CM columns fresh from bin indices in row-chunks
(`fill_cm_columns_from_bins!`) instead of copying from a persistent dense CM
matrix.
"""
function wrap_moments_with_cm_archB(core_moments!::Function, ncore_full::Int,
                                     Bidx::Matrix{Int}, origins::Vector{Int}, refIndex1::Int, L::Int,
                                     R::Union{Nothing,Matrix{Float64}}; chunk_size::Int = 2000)
    pregrav = ncore_full - 1
    nO = length(origins)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
    return function (K, G, θ, U, obj)
        n = size(U, 1)
        if size(Gtmp_cache[], 1) != n
            Gtmp_cache[] = Matrix{Float64}(undef, n, ncore_full)
        end
        Gtmp = Gtmp_cache[]
        core_moments!(K, Gtmp, θ, U, obj)
        @views G[:, 1:pregrav] .= Gtmp[:, 1:pregrav]
        @views G[:, end] .= Gtmp[:, end]
        cm_cols = pregrav + 1 : pregrav + L * nO
        fill_cm_columns_from_bins!(@view(G[:, cm_cols]), Bidx, origins, refIndex1, L, R; chunk_size = chunk_size)
        return nothing
    end
end

"""
    build_cm_augmented_obj_archB(ctx, CS; L, contrasts=:anchored) -> (obj_cm=..., ...)

Architecture-B analogue of `build_cm_augmented_obj`: same returned obj shape
and same `outer_constr_index`/`d` bookkeeping, but the `moments!` closure is
`wrap_moments_with_cm_archB` instead of `wrap_moments_with_cm`. `z`/`origins`
are taken from a throwaway `precalc_common_marginals_cdf` call (so `z` is
IDENTICAL to what Architecture A's reference CM matrix uses -- required for
the correctness comparison to be apples-to-apples) but the dense CM matrix it
returns is discarded immediately (never stored) -- Architecture B's whole
point is to not carry that persistent buffer.
"""
function build_cm_augmented_obj_archB(ctx, CS; L::Int, contrasts::Symbol = :anchored,
                                       refIndex1::Int = ctx.γ.refIndex1, chunk_size::Int = 2000)
    obj0 = ctx.obj
    ncore = obj0.d
    _CM_throwaway, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; contrasts = contrasts)
    ncm = L * length(origins)
    @assert ncm == n_cm_moments(ctx.D, L)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    Bidx = compute_bin_indices(ctx.U, z)

    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    moments_cm! = wrap_moments_with_cm_archB(obj0.moments!, ncore, Bidx, origins, refIndex1, L, R; chunk_size = chunk_size)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,   # 2026-07-24 release fix: was defaulting to Inf (disabled) on every rebuild
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, z = z, origins = origins, ncore = ncore, ncm = ncm, L = L,
            contrasts = contrasts, refIndex1 = refIndex1, Bidx = Bidx)
end

# ============================================================================
# ARCHITECTURE C: exact structured Hessian via weighted bin contingency
# tables. See docs/fullA_cm_hessian_architecture_report.md sec 2 for the
# full derivation; summary:
#   bin(u) = searchsortedfirst(z,u) in 1:(L+1);  1{u<=z_l} == (bin(u)<=l)
#   T_xy(k,h) = sum_s w_s 1{bin(U[s,x])=k} 1{bin(U[s,y])=h}     (D x D bin tables)
#   S_x(j,k)  = sum_s w_s E[s,j] 1{bin(U[s,x])=k}                (per-origin,
#                                                                  per-econ-col)
#   CT_xy(l,l') = sum_{k<=l,h<=l'} T_xy(k,h)   (2D prefix sum, O(L^2) per pair)
#   CS_x(j,l)   = sum_{k<=l} S_x(j,k)          (1D prefix sum)
# raw (anchored, R=I) blocks:
#   H_EE       = (1/M) E'DE                        (small dense BLAS, as usual)
#   H_EC[j,(o,l)]     = (1/M)[CS_o(j,l) - CS_ref(j,l)]
#   H_CC[(o,l),(p,l')] = (1/M)[CT_op(l,l') - CT_o,ref(l,l') - CT_ref,p(l,l') + CT_ref,ref(l,l')]
# orthonormal contrasts: CM = raw*R within each threshold block (nO x nO
# right-multiply), so H_EC_final[:,block_l] = H_EC_raw[:,block_l]*R and
# H_CC_final[block_l,block_l'] = R' * H_CC_raw[block_l,block_l'] * R
# (congruence, per threshold-block pair).
#
# Per-Hessian-call cost: O(W*(D*NCORE + D^2)) to build T/S tables (bin
# indices Bidx are PRECOMPUTED once, theta/weight-independent) + O(D^2*L^2 +
# D*NCORE*L) to prefix-sum + assemble the dense (NCORE+ncm)^2 output (this
# LAST step is unavoidably O((NCORE+ncm)^2) since KNITRO's dense callback API
# demands the full matrix -- Architecture C only cheapens the INGREDIENT
# computation, not the final materialization).
# ============================================================================

mutable struct CMBinHessCtx
    L::Int
    D::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    z::Vector{Float64}
    Bidx::Matrix{Int}          # W x D
    NCORE::Int
    ncm::Int
    contrasts::Symbol
    R::Union{Nothing,Matrix{Float64}}
    # scratch, rebuilt every call (sized once)
    Ttab::Array{Float64,4}     # D x D x (L+1) x (L+1)
    Stab::Array{Float64,3}     # D x NCORE x (L+1)
    CT::Array{Float64,4}       # D x D x L x L  (prefix-summed, 1:L only)
    CScum::Array{Float64,3}    # D x NCORE x L
    Ews::Matrix{Float64}       # W x NCORE scratch for sqrt(w)-scaled E
    Hfull::Matrix{Float64}     # (NCORE+ncm) x (NCORE+ncm) scratch
end

"""
    build_cm_bin_ctx(ctx, aug) -> CMBinHessCtx

Builds the Architecture-C precomputation (bin indices + scratch buffers) for
an existing `aug = build_cm_augmented_obj(...)` result. `z`/`origins` are
taken from `aug` itself so this is guaranteed to use IDENTICAL thresholds to
whatever CM matrix Architecture A is using.
"""
function build_cm_bin_ctx(ctx, aug)
    L = aug.L; D = ctx.D; origins = aug.origins; nO = length(origins)
    refIndex1 = aug.refIndex1; z = aug.z
    NCORE = aug.ncore; ncm = aug.ncm
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = compute_bin_indices(ctx.U, z)
    W = size(ctx.U, 1)
    L1 = L + 1
    return CMBinHessCtx(L, D, nO, origins, refIndex1, z, Bidx, NCORE, ncm, aug.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE, L1), zeros(D, D, L, L), zeros(D, NCORE, L),
        Matrix{Float64}(undef, W, NCORE), Matrix{Float64}(undef, NCORE + ncm, NCORE + ncm))
end

"Build the D x D and D x NCORE x (L+1) weighted bin tables from CURRENT weights `w` (obj.arg2) and economic block `E`. O(W*(D*NCORE + D^2))."
function build_bin_tables!(cctx::CMBinHessCtx, E::AbstractMatrix{Float64}, w::AbstractVector{Float64})
    D = cctx.D; NCORE = cctx.NCORE; Bidx = cctx.Bidx
    T = cctx.Ttab; S = cctx.Stab
    fill!(T, 0.0); fill!(S, 0.0)
    W = size(E, 1)
    @inbounds for s in 1:W
        ws = w[s]
        for x in 1:D
            bx = Bidx[s, x]
            for j in 1:NCORE
                S[x, j, bx] += ws * E[s, j]
            end
        end
        for x in 1:D
            bx = Bidx[s, x]
            for y in 1:D
                by = Bidx[s, y]
                T[x, y, bx, by] += ws
            end
        end
    end
    return nothing
end

"2D-prefix-sum `Ttab` into `CT` (restricted to l,l' in 1:L) and 1D-prefix-sum `Stab` into `CScum`. O(D^2*L^2 + D*NCORE*L)."
function prefix_sum_tables!(cctx::CMBinHessCtx)
    D = cctx.D; L = cctx.L; NCORE = cctx.NCORE
    T = cctx.Ttab; CT = cctx.CT
    @inbounds for x in 1:D, y in 1:D
        for l in 1:L
            for lp in 1:L
                v = T[x, y, l, lp]
                v += (l > 1 ? CT[x, y, l-1, lp] : 0.0)
                v += (lp > 1 ? CT[x, y, l, lp-1] : 0.0)
                v -= (l > 1 && lp > 1) ? CT[x, y, l-1, lp-1] : 0.0
                CT[x, y, l, lp] = v
            end
        end
    end
    S = cctx.Stab; CS_ = cctx.CScum
    @inbounds for x in 1:D, j in 1:NCORE
        acc = 0.0
        for l in 1:L
            acc += S[x, j, l]
            CS_[x, j, l] = acc
        end
    end
    return nothing
end

"""
    hessian_cm_structured!(h, obj, cctx::CMBinHessCtx)

Architecture C Hessian callback. Requires `obj.arg0` to already reflect the
CURRENT (zeta,lambda) (same precondition as `chunked_hessian.jl`'s
`hessian_chunked!` -- caller must run `_prep_for_hessian!(obj,x)` first, see
below). Writes the packed upper-triangular Hessian into `h`, matching
`cc_algo/PsiObjectiveBundle.jl::hessian!`'s own packing exactly.
"""
function hessian_cm_structured!(h, obj, cctx::CMBinHessCtx)
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins

    E = @view H[:, 2:1+NCORE]
    build_bin_tables!(cctx, E, w)
    prefix_sum_tables!(cctx)

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)

    # ---- H_EE: small dense BLAS on E only (NCORE x NCORE, cheap regardless of L) ----
    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)

    # ---- H_EC raw, then optional R congruence (right-multiply by R per threshold block) ----
    CS_ = cctx.CScum
    Hraw_EC = Matrix{Float64}(undef, NCORE, nO)   # reused per threshold block
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
        @views Hfull[1:NCORE, cols] .= block_ec
        # BUG FIX (found via c13_debug_archC.jl: uniform 2x discrepancy in H_EC vs
        # Architecture A): the symmetrize-by-averaging step below reads BOTH
        # Hfull[i,j] and Hfull[j,i] -- must mirror this block into the transposed
        # (CM-row, E-col) position too, or the average silently halves every H_EC
        # entry (H_CC was unaffected because that loop already visits both (l,l')
        # orderings explicitly; H_EE unaffected because BLAS gemm! fills both
        # triangles of a symmetric product).
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end

    # ---- H_CC raw, then optional R congruence (per threshold-block pair) ----
    CT = cctx.CT
    Hraw_CC = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
                Hraw_CC[oi, pi] = (CT[o, p, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, p, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = cctx.R === nothing ? Hraw_CC : (cctx.R' * Hraw_CC * cctx.R)
            @views Hfull[rows, cols] .= block
        end
    end

    # symmetrize defensively (analytically symmetric; absorbs FP-order noise, same
    # defensive pattern as compressed_inner_alt_solvers.jl's denseaccum callback)
    n = NCORE + ncm
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            h[k] = 0.5 * (Hfull[i, j] + Hfull[j, i])
            k += 1
        end
    end
    return h
end

"Same prep step chunked_hessian.jl uses (`_prep_for_hessian!`), duplicated here so this file has no load-order dependency on chunked_hessian.jl."
function _archC_prep_for_hessian!(obj, x)
    @unpack H, arg0, arg1, outer_constr_index, Psi! = obj
    BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
    Psi!(arg1, arg0)
    return nothing
end

# ============================================================================
# ARCHITECTURE D: matrix-free Hessian-vector-product diagnostic.
# Hv = (1/M) Z' (w .* (Z*v)), Z = H[:,2:1+outer_constr_index]. O(W*n) per
# call, O(n) extra memory (vs O(n^2) for the dense architectures). Diagnostic/
# validation candidate only, per task brief -- not expected to be a production
# winner unless it unexpectedly is (measured, not assumed).
# ============================================================================
function hvp_dense!(Hv::AbstractVector, obj, v::AbstractVector, zbuf::AbstractVector)
    @unpack H, M, arg0, arg2, ddPsi!, outer_constr_index = obj
    ddPsi!(arg2, arg0)
    Z = @view H[:, 2:1+outer_constr_index]
    mul!(zbuf, Z, v)                 # zbuf = Z*v            (W)
    zbuf .*= arg2                    # zbuf = w .* (Z*v)     (W)
    mul!(Hv, transpose(Z), zbuf)     # Hv   = Z' * zbuf      (n)
    Hv ./= M
    return Hv
end

function _callbackEvalHV_inner_dense!(kc, cb, evalRequest, evalResult, userParams)
    obj = userParams
    x = evalRequest.x
    v = evalRequest.vec
    @prof "inner_dual_hvp_callback_dense" begin
        _archC_prep_for_hessian!(obj, x)
        n = length(v)
        zbuf = Vector{Float64}(undef, size(obj.H, 1))
        Hv = Vector{Float64}(undef, n)
        hvp_dense!(Hv, obj, v, zbuf)
        @views evalResult.hessVec[1:n] .= Hv
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

# ============================================================================
# Generic KNITRO wiring, mirroring chunked_hessian.jl's
# inner_loop_KNITRO_chunked / inner_loop_internal_chunked pattern EXACTLY,
# parameterized by a `hess_cb_builder(obj) -> Function` (dense hessopt=1
# variants) or by `hvp=true` (hessopt=5 product variant). FG callback,
# variable/bound/init-value setup, complementarity wiring are all reused
# UNCHANGED from oracle_fast.jl (included by callers before this file).
# ============================================================================
function inner_loop_KNITRO_archgeneric(obj; hess_cb_builder = nothing, hvp::Bool = false)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)

    # Ported from diag/fullA-inner-blas-threading (parallelism_guards.jl); see the same note in
    # oracle_fast.jl::inner_loop_KNITRO_profiled -- this is the CM (Architecture B/C) production
    # inner solve, reached via cm_production_value_v2 -> cm_base_state_v2 ->
    # inner_loop_internal_archgeneric.
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_profiled!)
        KNITRO.KN_set_cb_user_params(kc, cb, obj)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        hessopt = KNITRO.KN_get_int_param(kc, "hessopt")
        if hvp
            hessopt == 5 || error("inner_loop_KNITRO_archgeneric(hvp=true): expected hessopt=product(5), got $hessopt -- obj.inner_loop_opt must point at ek_inner_hvp.opt")
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalHV_inner_dense!)
        elseif hessopt == 1
            hess_cb = hess_cb_builder(obj)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        @prof "inner_knitro_dual_solve_arch" begin
            KNITRO.KN_solve(kc)
        end
        nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        KNITRO.KN_free(kc)

        return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
    finally
        CS.guard_exit_inner_solve!()
    end
end

function inner_loop_internal_archgeneric(obj, θ; hess_cb_builder = nothing, hvp::Bool = false)
    @prof "inner_moment_build" begin
        obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    end
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_archgeneric(obj; hess_cb_builder = hess_cb_builder, hvp = hvp)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end
    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess
    end
end

"Architecture A's hess_cb_builder: the unchanged generic `hessian!`, wired via the profiled callback exactly as oracle_fast.jl does."
archA_hess_cb_builder(obj) = _callbackEvalH_inner_profiled!

"Architecture C's hess_cb_builder: closes over a CMBinHessCtx."
function archC_hess_cb_builder(cctx::CMBinHessCtx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_structured!(evalResult.hess, o, cctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
