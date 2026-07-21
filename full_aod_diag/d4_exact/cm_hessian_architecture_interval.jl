# ============================================================================
# Continuation 13 addendum (user-elevated to pre-production priority):
# interval-NATIVE Architecture C.
#
# cm_hessian_architectures.jl's Architecture C (`hessian_cm_structured!`) 2D-prefix-sums the raw
# weighted bin-contingency tables `Ttab`/`Stab` into CUMULATIVE second moments (`CT`/`CScum`) before
# assembling H_CC/H_EC -- built and validated specifically against the CUMULATIVE/CDF moment basis
# (`build_cm_augmented_obj`). The interval basis's moment columns are RAW bin-membership indicators
# (`1{bin(U_o)=l} - 1{bin(U_ref)=l}`, not `1{U_o<=z_l} - 1{U_ref<=z_l}`), so the matching exact
# Hessian for that basis needs the RAW (un-prefix-summed) `Ttab`/`Stab` entries at bin l directly --
# no CT/CScum, no prefix-sum step at all. This file builds that: a genuinely lean context (no
# CT/CScum fields, so there is nothing to accidentally fall back on) plus a matching Hessian
# callback, paired with cm_lookup_kernels.jl's ALREADY-VALIDATED interval lookup FG (`:interval`
# method) -- not re-derived, reused as instructed.
# ============================================================================

"""
    CMBinHessCtxInterval

Lean analog of `CMBinHessCtx` for the interval basis: same bin-index/scratch machinery, but NO
`CT`/`CScum` fields at all (nothing to skip -- there is no prefix-sum step to accidentally call).
"""
struct CMBinHessCtxInterval
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
    Ttab::Array{Float64,4}     # D x D x (L+1) x (L+1) -- RAW weighted bin-pair table, never prefix-summed
    Stab::Array{Float64,3}     # D x NCORE x (L+1)      -- RAW weighted bin table, never prefix-summed
    Ews::Matrix{Float64}
    Hfull::Matrix{Float64}
end

"""
    build_cm_bin_ctx_interval(ctx, aug_interval) -> CMBinHessCtxInterval

`aug_interval` is a `build_cm_augmented_obj_interval(...)` result: reuses its OWN `.bins` field
directly (already computed by `precalc_common_marginals_interval` from the SAME `z`/`U`) rather
than recomputing bin indices a second time -- guarantees bit-identical bins to whatever produced
the dense reference `aug_interval.CM`, not merely numerically-close ones.
"""
function build_cm_bin_ctx_interval(ctx, aug_interval)
    L = aug_interval.L; D = ctx.D; origins = aug_interval.origins; nO = length(origins)
    refIndex1 = aug_interval.refIndex1; z = aug_interval.z
    NCORE = aug_interval.ncore; ncm = aug_interval.ncm
    R = aug_interval.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = Int.(aug_interval.bins)
    W = size(ctx.U, 1)
    L1 = L + 1
    return CMBinHessCtxInterval(L, D, nO, origins, refIndex1, z, Bidx, NCORE, ncm, aug_interval.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE, L1),
        Matrix{Float64}(undef, W, NCORE), Matrix{Float64}(undef, NCORE + ncm, NCORE + ncm))
end

"Identical body to cm_hessian_architectures.jl::build_bin_tables! -- duplicated (not shared) because it is type-constrained to CMBinHessCtx there; this is the SAME raw weighted bin-pair/bin table construction, just typed for CMBinHessCtxInterval so there is no ambient dispatch ambiguity between the two contexts."
function build_bin_tables_interval!(cctx::CMBinHessCtxInterval, E::AbstractMatrix{Float64}, w::AbstractVector{Float64})
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

"""
    hessian_cm_structured_interval!(h, obj, cctx::CMBinHessCtxInterval)

Interval-native Architecture C. Same H_EE (dense BLAS on the core block, basis-independent) and
same overall packing as `hessian_cm_structured!`; H_EC/H_CC differ ONLY in reading `Ttab[o,p,l,lp]`/
`Stab[o,j,l]` directly (bin l itself) instead of `CT[o,p,l,lp]`/`CScum[o,j,l]` (bins `<=l`) -- the
exact analytic Hessian of `sum_oi (T_o,l - T_ref,l)`-type interval columns, not a re-derivation from
scratch: differentiating the interval moment's own defining indicator (`1{bin=l}`, not `1{bin<=l}`)
w.r.t. the dual weights gives precisely the RAW bin-l contingency entry, by the same chain-rule
argument `docs/fullA_cm_hessian_architecture_report.md` sec 2 already spells out for the cumulative
case (this file's docstring for `hessian_cm_structured!`-vs-interval difference is exactly that:
swap `CT_xy(l,l')=sum_{k<=l,h<=l'}T_xy(k,h)` back out for the raw `T_xy(l,l')` term it was built from).
"""
function hessian_cm_structured_interval!(h, obj, cctx::CMBinHessCtxInterval)
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins

    E = @view H[:, 2:1+NCORE]
    build_bin_tables_interval!(cctx, E, w)

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)

    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)

    T = cctx.Ttab; S = cctx.Stab
    Hraw_EC = Matrix{Float64}(undef, NCORE, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (S[o, j, l] - S[refIndex1, j, l]) / M
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
        @views Hfull[1:NCORE, cols] .= block_ec
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end

    Hraw_CC = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
                Hraw_CC[oi, pi] = (T[o, p, l, lp] - T[o, refIndex1, l, lp] - T[refIndex1, p, l, lp] + T[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = cctx.R === nothing ? Hraw_CC : (cctx.R' * Hraw_CC * cctx.R)
            @views Hfull[rows, cols] .= block
        end
    end

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

"Architecture-C-interval hess_cb_builder, mirrors archC_hess_cb_builder exactly (cm_hessian_architectures.jl)."
function archC_interval_hess_cb_builder(cctx::CMBinHessCtxInterval)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_interval" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_structured_interval!(evalResult.hess, o, cctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
