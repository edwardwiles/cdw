# ============================================================================
# Phase B (2026-07-24), B1: startup diagnostic recording the CM restriction
# basis actually in effect. See docs/CURRENT_CM_BASIS_AND_STORAGE_AUDIT_2026-07-24.md
# for the full source trace this asserts. This file changes no behavior --
# it is a self-contained, dependency-free constant/print pair, safe to
# include anywhere (no macro/global dependency on instrumentation.jl or any
# other file, so it cannot break an existing script's include order).
# ============================================================================

"The CM dual moment columns are cumulative CDF origin contrasts (U[:,o] <= z_l) - (U[:,ref] <= z_l)
evaluated at each threshold z_l -- see precalc_common_marginals_cdf (common_marginals_moments.jl)
and fill_cm_columns_from_bins! (cm_hessian_architectures.jl). NOT disjoint interval-mass moments."
const CM_RESTRICTION_BASIS = :cumulative_cdf_contrasts

"Bidx (compute_bin_indices) stores each draw's INTERVAL/bin membership (bin(u) in 1:(L+1)) as an
internal implementation device for cheaply recovering the cumulative contrast above
(1{u<=z_l} == (bin(u)<=l)) and for the structured Hessian's bin-contingency-table contraction.
It is never itself used as the dual restriction basis."
const CM_INTERNAL_FEATURE_STORAGE = :bin_indices

"""
    report_cm_basis_diagnostic(io::IO = stdout; context_label::AbstractString = "")

Print the two-line startup diagnostic B1 requires. Call once from every
CM-family production-context builder (`build_cm_production_context`,
`build_cm_meanzc_production_context`, `build_originzc_production_context`) so
it is visible whenever a live campaign starts up.
"""
function report_cm_basis_diagnostic(io::IO = stdout; context_label::AbstractString = "")
    label = isempty(context_label) ? "" : " [" * context_label * "]"
    println(io, "cm_restriction_basis", label, " = ", CM_RESTRICTION_BASIS)
    println(io, "cm_internal_feature_storage", label, " = ", CM_INTERNAL_FEATURE_STORAGE)
    flush(io)
    return nothing
end
