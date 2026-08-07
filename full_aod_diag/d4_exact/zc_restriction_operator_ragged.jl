# ============================================================================
# fix/zc-profile-focal-sigmaminus1-mean-2026-08-07: the `ActiveMeanLayout`-consuming
# `ZCRestrictionOperator` constructor.
#
# WHY THIS IS ITS OWN FILE (not just added to zc_restriction_operator.jl): confirmed live
# (2026-08-07) that `zc_restriction_operator.jl` loads EARLY in the production include chain
# (transitively, via `compressed_live.jl` -> `operator_verification.jl`, from
# `c10_d20_production_driver.jl`'s core include list), while `cm_originzc_target_layout.jl`
# (defines `ActiveMeanLayout`) is only included LATER, from the origin-ZC/CM+ZC-family-specific
# checkpoint drivers and gate scripts. A constructor referencing `ActiveMeanLayout` by type cannot
# live inside `zc_restriction_operator.jl` itself without an `UndefVarError` at that file's own
# `include` time for every caller that pulls it in via the core chain (confirmed: this exact error
# was hit live while building this task, via `c10_d20_production_driver.jl` -> `compressed_live.jl`
# -> `operator_verification.jl` -> `zc_restriction_operator.jl`). Splitting the ActiveMeanLayout-
# dependent constructor into its own leaf file, self-guarding BOTH dependencies via the same
# `isdefined(Main, :X) || include(...)` idiom this codebase already uses pervasively, lets it be
# included from anywhere (the origin-ZC/CM+ZC checkpoint drivers) regardless of what the core
# chain already loaded, with no reordering of the existing, widely-depended-on core include list.
# ============================================================================
isdefined(Main, :ZCRestrictionOperator) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))
isdefined(Main, :ActiveMeanLayout) || include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))

"""
    ZCRestrictionOperator(Zraw_all_full, Zpairraw_all, D, aml::ActiveMeanLayout)

Row-omission constructor: `Zraw_all_full[k]` is the FULL, uncompacted `(W, D)` compact feature
table (task Section 7: "retain the complete compact table containing focal k_*" is acceptable at
the CALLER level -- this constructor is what performs the ONE-TIME compaction into the ACTIVE
operator, so the caller's own full table can still be kept around for other uses/diagnostics if
needed). If `!aml.active`, this is byte-identical to the 3-arg constructor in
`zc_restriction_operator.jl` (every column retained, `Zraw_all[k] === Zraw_all_full[k]`, no copy).
"""
function ZCRestrictionOperator(Zraw_all_full::Vector{Matrix{Float64}}, Zpairraw_all::Vector{Matrix{Float64}},
                                D::Int, aml::ActiveMeanLayout)
    npair = D * (D - 1) ÷ 2
    K_mean = length(Zraw_all_full)
    K_mean == aml.base.K_mean || error("ZCRestrictionOperator(...,aml): length(Zraw_all_full)=$K_mean != aml.base.K_mean=$(aml.base.K_mean)")
    Zraw_all = Vector{Matrix{Float64}}(undef, K_mean)
    for k in 1:K_mean
        active = aml.mean_active_origins[k]
        Zraw_all[k] = length(active) == D ? Zraw_all_full[k] : Zraw_all_full[k][:, active]
    end
    widths = [length(aml.mean_active_origins[k]) for k in 1:K_mean]
    mean_offset = vcat(0, cumsum(widths))
    ZCRestrictionOperator(Zraw_all, Zpairraw_all, D, npair, K_mean, length(Zpairraw_all),
                           aml.mean_active_origins, mean_offset)
end
