# ============================================================================
# Production outer bridge task (2026-08-01), §4 test: validate_family_layout_
# against_evaluation. Real D4 ctx/ev (unrestricted), plus the mock restricted
# family for the "extended beta" cases. Positive case + 6 negative cases,
# each proving a real throw (not just "no crash").
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
include(joinpath(@__DIR__, "profiled_stable_layout_digest_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_evaluation_aware_validator_2026-08-01.jl"))
using Test, CSV, DataFrames

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
ev = evaluate_profiled_point(w_calib, ctx, spec, pe)

ufctx = build_unrestricted_family_ctx(ctx, spec, pe, ev)
mfctx = build_mock_restricted_family_ctx(ufctx, :flexible_CM; n_restriction = 5)
ev_mock = ev_with_beta(ev, mock_extended_beta(ev, 5))

rows = NamedTuple[]
function record!(name::String, pass::Bool, detail::String)
    push!(rows, (test = name, pass = pass, detail = detail))
    println(pass ? "PASS  " : "FAIL  ", name, "  -- ", detail)
end

# 1. Positive: well-formed unrestricted fctx/ev passes.
ok = try
    r = validate_family_layout_against_evaluation(ufctx, ev)
    r.n_beta == length(ev.result.beta)
catch e
    println("  unexpected throw: ", e); false
end
record!("positive_unrestricted_passes", ok, "real D4 ctx/ev")

# 2. Positive: well-formed mock restricted fctx/ev_mock passes.
ok = try
    r = validate_family_layout_against_evaluation(mfctx, ev_mock)
    r.n_beta == length(ev_mock.result.beta)
catch e
    println("  unexpected throw: ", e); false
end
record!("positive_mock_restricted_passes", ok, "real D4 economic beta + 5 synthetic restriction entries")

# 3. Negative: economic range out of bounds (ev truncated below what economic_dual_range needs).
struct _TruncatedCtx
    base::Any
end
Main.profiled_economic_layout(f::_TruncatedCtx) = profiled_economic_layout(f.base)
Main.economic_dual_range(f::_TruncatedCtx) = economic_dual_range(f.base)
Main.restriction_dual_ranges(f::_TruncatedCtx) = restriction_dual_ranges(f.base)
Main.profiled_anchor_spec(f::_TruncatedCtx) = profiled_anchor_spec(f.base)
Main.profiled_outer_coordinate_layout(f::_TruncatedCtx) = profiled_outer_coordinate_layout(f.base)
Main.family_kind(f::_TruncatedCtx) = family_kind(f.base)
Main.layout_checksum(f::_TruncatedCtx) = layout_checksum(f.base)
tctx = _TruncatedCtx(ufctx)
ev_truncated = ev_with_beta(ev, ev.result.beta[1:(end - 1)])   # one entry short
threw = false
try
    validate_family_layout_against_evaluation(tctx, ev_truncated)
catch e
    global threw = occursin("out of bounds", sprint(showerror, e)) || occursin("do not cover", sprint(showerror, e))
end
record!("truncated_beta_throws", threw, "ev.result.beta one entry short of economic_dual_range's own length")

# 4. Negative: restriction range out of bounds (mock ctx against unextended real ev).
threw = false
try
    validate_family_layout_against_evaluation(mfctx, ev)   # ev has NO restriction entries in beta
catch e
    global threw = occursin("out of bounds", sprint(showerror, e))
end
record!("restriction_range_out_of_bounds_throws", threw, "mock restriction range points past real (unextended) beta")

# 5. Negative: gap in coverage (extend beta by MORE than n_restriction, leaving an unexplained trailing entry).
ev_gap = ev_with_beta(ev, mock_extended_beta(ev, 6))   # 6 raw entries but mfctx only claims 5
threw = false
try
    validate_family_layout_against_evaluation(mfctx, ev_gap)
catch e
    global threw = occursin("do not cover", sprint(showerror, e))
end
record!("unexplained_trailing_entry_throws", threw, "beta has 6 restriction entries, fctx only accounts for 5")

# 6. Negative: family_kind mismatch when ev exposes one.
ev_wrong_family = merge(ev_mock, (family_kind = :common_Frechet,))
threw = false
try
    validate_family_layout_against_evaluation(mfctx, ev_wrong_family)
catch e
    global threw = occursin("family_kind", sprint(showerror, e))
end
record!("family_kind_mismatch_throws", threw, "fctx claims flexible_CM, ev claims common_Frechet")

# 7. Negative: require_manifest_digest=true but ev has no manifest field -> throws (no silent skip).
threw = false
try
    validate_family_layout_against_evaluation(ufctx, ev; require_manifest_digest = true)
catch e
    global threw = occursin("layout_digest_manifest", sprint(showerror, e))
end
record!("required_manifest_digest_missing_throws", threw, "require_manifest_digest=true, ev has no :layout_digest_manifest field")

# 8. Negative: manifest digest mismatch when ev supplies a WRONG one.
ev_bad_digest = merge(ev, (layout_digest_manifest = "0"^64,))
threw = false
try
    validate_family_layout_against_evaluation(ufctx, ev_bad_digest; require_manifest_digest = true)
catch e
    global threw = occursin("layout_digest_manifest", sprint(showerror, e))
end
record!("manifest_digest_mismatch_throws", threw, "ev supplies deliberately wrong all-zero digest")

# 9. Positive: manifest digest matches when ev supplies the TRUE digest.
true_digest = stable_layout_digest(ufctx)
ev_good_digest = merge(ev, (layout_digest_manifest = true_digest,))
ok = try
    validate_family_layout_against_evaluation(ufctx, ev_good_digest; require_manifest_digest = true)
    true
catch e
    println("  unexpected throw: ", e); false
end
record!("manifest_digest_match_passes", ok, "ev supplies the true stable_layout_digest(fctx)")

df = DataFrame(rows)
out_csv = joinpath(dirname(dirname(@__DIR__)), "PROFILED_EVALUATION_AWARE_VALIDATOR_GATE_2026-08-01.csv")
CSV.write(out_csv, df)
println("\nWrote $out_csv")
show(df, allrows = true, allcols = true)
println()

all_pass = all(r.pass for r in rows)
println("\nEVALUATION-AWARE VALIDATOR GATE: ", all_pass ? "PASS" : "FAIL")
@assert all_pass
