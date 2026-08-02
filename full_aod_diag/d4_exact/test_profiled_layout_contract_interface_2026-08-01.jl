# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §11
# "Interface tests": prove no hard-coded family offsets, an incorrect layout
# checksum throws, a wrong dual length throws, and an anchor coordinate
# cannot appear in the economic block. D4, fast (no KNITRO solve needed --
# these are pure structural/bookkeeping checks against a real ctx/spec/pe).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "profiled_lfix_incremental_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_layout_contract_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_shared_economic_gradient_engine_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_family_adapters_2026-08-01.jl"))
using Test, CSV, DataFrames

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
ev = evaluate_profiled_point(w_calib, ctx, spec, pe)

ufctx = build_unrestricted_family_ctx(ctx, spec, pe, ev)
mfctx = build_mock_restricted_family_ctx(ufctx, :flexible_CM; n_restriction = 5)
mfctx_bad = build_mock_restricted_family_ctx(ufctx, :flexible_CM; n_restriction = 5, bad_checksum = true)

rows = NamedTuple[]
function record!(name::String, pass::Bool, detail::String)
    push!(rows, (test = name, pass = pass, detail = detail))
    println(pass ? "PASS  " : "FAIL  ", name, "  -- ", detail)
end

# 1. No hard-coded family offsets: economic_dual_range for the mock restricted
#    family is IDENTICAL to the unrestricted family's (same real economic
#    layout reused, not re-derived from a family-specific constant).
record!("no_hardcoded_offset__economic_range_matches_unrestricted",
    economic_dual_range(mfctx) == economic_dual_range(ufctx),
    "economic_dual_range(mock)=$(economic_dual_range(mfctx)) economic_dual_range(unrestricted)=$(economic_dual_range(ufctx))")

# 2. Restriction range is disjoint from and strictly after the economic range
#    (no manual arithmetic assumed by a caller -- read directly off the accessor).
rr = restriction_dual_ranges(mfctx)
record!("restriction_range_disjoint_and_after_economic",
    length(rr) == 1 && isempty(intersect(rr[1].range, economic_dual_range(mfctx))) && minimum(rr[1].range) > maximum(economic_dual_range(mfctx)),
    "restriction_dual_ranges(mock)=$rr")

# 3. validate_family_layout_contract succeeds on a well-formed context.
ok = try
    validate_family_layout_contract(ufctx); validate_family_layout_contract(mfctx)
    true
catch e
    println("  unexpected throw: ", e); false
end
record!("validate_family_layout_contract_passes_wellformed", ok, "unrestricted + mock flexible_CM")

# 4. Incorrect layout checksum throws.
threw = false
try
    validate_family_layout_contract(mfctx_bad)
catch e
    global threw = occursin("layout_checksum mismatch", sprint(showerror, e))
end
record!("incorrect_layout_checksum_throws", threw, "bad_checksum=true adapter")

# 5. Wrong dual length throws: economic_dual_range longer than beta.
struct _BadRangeCtx
    base::UnrestrictedFamilyCtx
end
profiled_economic_layout(f::_BadRangeCtx) = profiled_economic_layout(f.base)
economic_dual_range(f::_BadRangeCtx) = 1:(profiled_economic_layout(f.base).total_reduced_economic_moments + 1000)  # deliberately too long
restriction_dual_ranges(::_BadRangeCtx) = RestrictionDualRange[]
profiled_anchor_spec(f::_BadRangeCtx) = profiled_anchor_spec(f.base)
profiled_outer_coordinate_layout(f::_BadRangeCtx) = profiled_outer_coordinate_layout(f.base)
family_kind(::_BadRangeCtx) = :bad_range_test
layout_checksum(f::_BadRangeCtx) = structural_checksum(profiled_economic_layout(f), profiled_anchor_spec(f), profiled_outer_coordinate_layout(f), RestrictionDualRange[])
restriction_contrib0(::_BadRangeCtx, ev) = zeros(ev.st.cf.W)

badf = _BadRangeCtx(ufctx)
threw2 = false
try
    validate_family_layout_contract(badf)
catch e
    global threw2 = true
end
threw3 = false
try
    build_shared_profiled_lfix_cache(w_calib, badf, ctx, ev)
catch e
    # validate_family_layout_contract (called first, inside the cache builder) already
    # rejects the mismatched range length before the cache builder's own "exceeds beta
    # length" defense-in-depth check is ever reached -- either message is an acceptable
    # PASS here, since the point of this test is that a wrong dual length is NEVER
    # silently used to index beta, by any layer.
    msg = sprint(showerror, e)
    global threw3 = occursin("exceeds beta length", msg) || occursin("total_reduced_economic_moments", msg)
end
record!("wrong_dual_length__contract_validator_throws", threw2, "economic_dual_range too long vs total_reduced_economic_moments")
record!("wrong_dual_length__cache_builder_throws", threw3, "economic_dual_range exceeds actual beta length (caught by validator or cache builder)")

# 6. Anchor coordinate cannot appear: assert_no_factual_price_index_moment
#    (called inside validate_family_layout_contract) fails loudly if an
#    anchor cell is smuggled into retained_full_factual_j. Prove this by
#    constructing a deliberately corrupted layout (anchor cell re-inserted).
bad_layout = let L = ufctx.layout
    bad_retained = vcat(L.retained_full_factual_j, [1])  # cell 1 may or may not be anchor; use the KNOWN anchor cell instead
    anchor_j = active_cell_index(ctx, L.anchor_origin_by_slot[1], 1)
    bad_retained2 = vcat(L.retained_full_factual_j, [anchor_j])
    ProfiledEconomicMomentLayout(L.D, L.Ddest, L.destination_ids, L.anchor_origin_by_slot,
        bad_retained2, vcat(L.retained_origin, [ufctx.spec.anchor_origin[1]]), vcat(L.retained_slot, [1]),
        L.full_factual_to_reduced, L.reduced_to_full_factual, L.france_ratio_reduced_j, L.total_reduced_economic_moments + 1)
end
threw4 = false
try
    assert_no_factual_price_index_moment(bad_layout)
catch e
    global threw4 = true
end
record!("anchor_coordinate_cannot_appear__corrupted_layout_throws", threw4, "anchor cell reinserted into retained_full_factual_j")

# 7. Shared-method identity assertion passes.
ok7 = try
    assert_shared_gradient_method_identity()
    true
catch e
    println("  unexpected throw: ", e); false
end
record!("shared_gradient_method_identity", ok7, "profiled_composite_gradient_from_cache has exactly one applicable method")

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_RESTRICTED_OUTER_GRADIENT_PREINTEGRATION_GATE_2026-08-01.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)
all_pass = all(r.pass for r in rows)
println("\nINTERFACE TESTS: ", all_pass ? "PASS" : "FAIL")
all_pass || error("interface tests failed")
