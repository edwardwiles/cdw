# Phase 0 gate closure (2026-07-26, production-audit task): threaded Architecture-C Hessian
# variant for the common-Fréchet level block. Closes the disclosed gap in
# `cm_frechet_hessian.jl::archC_frechet_hess_cb_builder`'s own docstring: "The threaded bin-table
# variant (hessian_cm_structured_v2!) is NOT extended for the level block yet".
#
# Design: per COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md, the level block's three new
# Hessian sub-blocks (H_E,level / H_CM,level / H_level,level) are pure linear combinations of the
# SAME Ttab/CT/Stab/CScum tables the CM block already reads -- no separate O(W) pass. This means
# threading the bin-table construction (`build_bin_tables_threaded!`/`prefix_sum_tables_threaded!`,
# cm_hessian_threaded.jl, UNCHANGED) automatically threads the dominant cost for the level block
# too; only the small O(D*NCORE*L + D^2*L^2) assembly tail stays serial, exactly as it does for
# plain CM's own H_EC/H_CC in `hessian_cm_structured_v2!`.
#
# Both the H_EC/H_CC tail and the three level-block correction terms below are copied VERBATIM from
# `hessian_cm_structured_v2!` (cm_hessian_threaded.jl) and `hessian_cm_frechet_structured!`
# (cm_frechet_hessian.jl) respectively -- not re-derived -- per this project's own stated
# methodology for this feature ("reuse, don't rebuild the bin-contingency tables"; "the H_EC/H_CC
# tail is copied verbatim from the original to minimize the chance of a second divergent bug
# site"). Only the dispatch on `threaded_bins`/`tls` is new.
#
# Depends on (must already be included): cm_hessian_architectures.jl (CMBinHessCtx, build_bin_tables!,
# prefix_sum_tables!, _fill_cm_HEE!), cm_hessian_threaded.jl (ThreadLocalBinScratch,
# build_bin_tables_threaded!, prefix_sum_tables_threaded!), cm_frechet_hessian.jl (this file's
# serial sibling, for the docstring cross-reference and the level_targets semantics).

"""
    hessian_cm_frechet_structured_v2!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64}; threaded_bins=false, tls=nothing)

Harmonization task (2026-07-28): thin backward-compatibility wrapper. The real implementation is
now the shared `hessian_cm_structured_v2!` (cm_hessian_threaded.jl) with a resolved
`CMFrechetExtension` -- kept here (not deleted) because several pre-existing, still-referenced
diagnostic/gate scripts (`test_cm_frechet_threaded_hessian_gates.jl`,
`test_frechet_winner_bin_her_wiring_d4.jl`/`_d20.jl`) call this exact name with a bare
`level_targets::Vector{Float64}`, not through `archC_frechet_hess_cb_builder`. `use_syrk` is
accepted and ignored for backward compatibility (it was already a no-op in the pre-harmonization
version -- `_fill_cm_HEE!`'s shared dense fallback always uses `gemm!`).
"""
function hessian_cm_frechet_structured_v2!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64};
                                            threaded_bins::Bool = false,
                                            tls::Union{Nothing,ThreadLocalBinScratch} = nothing,
                                            use_syrk::Bool = true)
    return hessian_cm_structured_v2!(h, obj, cctx, _resolve_frechet_ext!(cctx, level_targets);
                                      threaded_bins = threaded_bins, tls = tls, use_syrk = use_syrk)
end

    # harmonization task (2026-07-28): removed dead archC_frechet_hess_cb_builder_v2 -- confirmed
    # zero call sites in the repo (only archC_frechet_hess_cb_builder in cm_frechet_hessian.jl is
    # the real production dispatcher, which already handles both threaded_bins branches). This
    # dead function also called the legacy dense-only _archC_prep_for_hessian! rather than the
    # dense-G-free _prep_dual_index_for_archC! flexible CM/the real Fréchet builder use -- another
    # sign it predates the no-H work and was never updated because nothing calls it.
