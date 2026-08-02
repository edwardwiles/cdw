# ZC lane task (2026-08-02): the REAL CM+ZC implementation of the five-accessor contract, mirroring
# profiled_originzc_family_adapter_2026-08-02.jl exactly (see that file's own header for the naming/
# dispatch-safety note -- identical reasoning applies here: `CMZCFamilyCtx` is a NEW adapter type,
# never confused with `CMBinHessCtx` itself despite sharing some accessor NAMES with
# profiled_restricted_accessors_2026-08-01.jl's own (deliberately incompatible, differently-indexed)
# methods on that type).
#
# CM+ZC's own economic_dual_range is the TRUE economic block ONLY (1:total_reduced_economic_moments)
# -- NOT the widened NCORE (which also includes the mean/pair-Z columns, per this session's own
# widened-core Hessian work / CMZC_WIDENED_CORE_FINDING_2026-08-01.md). The mean/pair-Z AND CM-grid
# blocks are both "restriction" for this accessor's purposes, matching the task's own conceptual
# split (reduced economic | Z mean | Z pair | gravity | CM-grid C block) even though the Hessian
# widens NCORE to include Z mean/pair alongside the true economic block.
# ADDITIVE ONLY.

isdefined(Main, :validate_family_layout_contract) || error("profiled_cmzc_family_adapter_2026-08-02.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")
isdefined(Main, :restriction_contrib0_cmzc!) || error("profiled_cmzc_family_adapter_2026-08-02.jl requires profiled_restriction_contrib0_operators_2026-08-01.jl to be included first.")

"""
    CMZCFamilyCtx

Real CM+ZC adapter. Wraps the already-validated `(ctx, spec, pe, layout)` unchanged (same economic
layout every family shares) plus the ZC restriction operator state AND the CM-grid state
(`bins`/`refIndex1`/`origins`/`R`) `restriction_contrib0_cmzc!` needs.
"""
struct CMZCFamilyCtx
    ctx::Any
    spec::AnchorSpec
    pe::PivotGravityElimOnRetained
    layout::ProfiledEconomicMomentLayout
    n_lambda_meanpair::Int   # DUAL-coordinate width of the mean/pair restriction block
    # (K_mean*D + K_pair*npair) -- NOT `n_eta(zc_layout)` (=K_mean, the count of ν_k OUTER
    # parameters; a different quantity entirely, despite the tempting name collision. Confirmed
    # live 2026-08-02: using n_eta(zc_layout) here silently truncated the :zc_targets dual range
    # to K_mean=1 columns instead of the true 4 (K_mean=1,K_pair=0,D=4), corrupting
    # restriction_contrib0's λ_pair slice and the CM-grid slice that followed it -- caught via the
    # q0-vs-fixed-dual-residual diagnostic (0.15 mismatch, same diagnostic that caught the origin-ZC
    # gravity bug).
    ncm::Int
    op::ZCRestrictionOperator
    zc_ws::ZCRestrictionWorkspace
    zc_layout::Any        # SharedByPowerLayout -- for refresh_zc_targets!
    cm_ws::FlexCMRestrictionWorkspace
    bins::Matrix{<:Unsigned}
    refIndex1::Int
    origins::Vector{Int}
    R::Union{Nothing,Matrix{Float64}}
end

"""
    build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced, cctx_reduced, bins) -> CMZCFamilyCtx

`aug_reduced` is `build_cm_meanzc_augmented_obj(...)`'s own return value; `cctx_reduced` is the
matching `build_cm_meanzc_bin_ctx(...)` result (source of `origins`/`refIndex1`/`R`/`L`); `bins` is
the SAME bin-index matrix (`cm_bin_indices_for(ctx, aug_reduced)`) the real solve's own CM-grid
forward kernel uses.
"""
function build_cmzc_family_ctx(ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained,
        layout::ProfiledEconomicMomentLayout, aug_reduced, cctx_reduced, bins::Matrix{<:Unsigned})
    zc_layout = SharedByPowerLayout(aug_reduced.K_mean, aug_reduced.K_pair)
    op = ZCRestrictionOperator(aug_reduced.Zraw_all, aug_reduced.Zpairraw_all, ctx.D)
    zc_ws = ZCRestrictionWorkspace(op)
    cm_ws = FlexCMRestrictionWorkspace(length(cctx_reduced.origins), cctx_reduced.L, size(ctx.U, 1))
    n_lambda_meanpair = aug_reduced.K_mean * ctx.D + aug_reduced.K_pair * op.npair
    return CMZCFamilyCtx(ctx, spec, pe, layout, n_lambda_meanpair, cctx_reduced.ncm, op, zc_ws, zc_layout,
        cm_ws, bins, cctx_reduced.refIndex1, cctx_reduced.origins, cctx_reduced.R)
end

profiled_economic_layout(fctx::CMZCFamilyCtx) = fctx.layout
economic_dual_range(fctx::CMZCFamilyCtx) = 1:fctx.layout.total_reduced_economic_moments
function restriction_dual_ranges(fctx::CMZCFamilyCtx)
    n_econ = fctx.layout.total_reduced_economic_moments
    return [RestrictionDualRange(:zc_targets, (n_econ + 1):(n_econ + fctx.n_lambda_meanpair)),
            RestrictionDualRange(:cm_marginals, (n_econ + fctx.n_lambda_meanpair + 1):(n_econ + fctx.n_lambda_meanpair + fctx.ncm))]
end
profiled_anchor_spec(fctx::CMZCFamilyCtx) = fctx.spec
profiled_outer_coordinate_layout(fctx::CMZCFamilyCtx) = fctx.pe
family_kind(::CMZCFamilyCtx) = :CM_plus_ZC
layout_checksum(fctx::CMZCFamilyCtx) =
    structural_checksum(fctx.layout, fctx.spec, fctx.pe, restriction_dual_ranges(fctx))

"""
    restriction_contrib0(fctx::CMZCFamilyCtx, ev) -> Vector{Float64}

`ev.result.beta`'s restriction slice is `[λ_mean; λ_pair; λ_cm]` (matching CM+ZC's own widened-core
+ CM-grid column order, `restriction_dual_ranges`'s own order above). `ev.nu_full` supplies the
current outer point's ZC targets for `refresh_zc_targets!` -- same discipline as the origin-ZC
adapter's identical field.
"""
function restriction_contrib0(fctx::CMZCFamilyCtx, ev)
    n_econ = fctx.layout.total_reduced_economic_moments
    K_mean = fctx.zc_layout.K_mean; K_pair = fctx.zc_layout.K_pair
    D = fctx.op.D; npair = fctx.op.npair
    n_mean = K_mean * D; n_pair = K_pair * npair
    β = ev.result.beta
    λ_mean = @view β[n_econ+1 : n_econ+n_mean]
    λ_pair = @view β[n_econ+n_mean+1 : n_econ+fctx.n_lambda_meanpair]
    λ_cm = @view β[n_econ+fctx.n_lambda_meanpair+1 : n_econ+fctx.n_lambda_meanpair+fctx.ncm]
    refresh_zc_targets!(fctx.zc_ws, fctx.op, fctx.zc_layout, ev.nu_full)
    dest = Vector{Float64}(undef, ev.st.cf.W)
    restriction_contrib0_cmzc!(dest, λ_cm, λ_mean, λ_pair, fctx.bins, fctx.refIndex1, fctx.origins,
        fctx.R, fctx.op, fctx.zc_ws, ev.st.cf.SW, fctx.cm_ws)
    return dest
end
