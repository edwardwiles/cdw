# ============================================================================
# Production outer bridge task (2026-08-01), §13-15 test: production outer
# runner scaffold (both modes), A/B comparability hardening, checkpoint/
# plumbing manifest. Real D4 KNITRO, unrestricted family only (the only real
# family/evaluator this worktree has -- restricted families remain
# BLOCKED_PENDING_INNER, consistent with every other gate on this branch).
# Both KNITRO runs use tiny maxit_override so this stays a fast smoke test,
# not a real outer-search campaign (task: "Do not launch a production
# campaign").
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
include(joinpath(@__DIR__, "PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_production_outer_runner_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_ab_comparability_and_plumbing_2026-08-01.jl"))
using Test, CSV, DataFrames

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)

evaluate_fn, family_ctx_builder = unrestricted_ab_arm(ctx, spec, pe)

rows = NamedTuple[]
function record!(name::String, pass::Bool, detail::String)
    push!(rows, (test = name, pass = pass, detail = detail))
    println(pass ? "PASS  " : "FAIL  ", name, "  -- ", detail)
end

# 1. invalid mode throws
threw = false
try
    run_profiled_production_outer(:bogus_mode, "t", w_calib; ctx, evaluate_fn, family_ctx_builder)
catch e
    global threw = occursin("not in", sprint(showerror, e))
end
record!("invalid_mode_throws", threw, "mode=:bogus_mode")

# 2. :fixed_gp_parameterization_ab smoke -- gp never in free-var manifest.
res_ab, man_ab = run_profiled_production_outer(:fixed_gp_parameterization_ab, "smoke_ab", w_calib;
    ctx, evaluate_fn, family_ctx_builder, maxit_override = 4)
record!("fixed_gp_ab_runs", res_ab.n_eval > 0, "n_eval=$(res_ab.n_eval) status=$(res_ab.knitro_status)")
record!("fixed_gp_ab_manifest_excludes_gp", !man_ab.gp_free && length(man_ab.free_coordinate_names) == man_ab.n_free,
    "gp_free=$(man_ab.gp_free) n_free=$(man_ab.n_free) names[1:2]=$(man_ab.free_coordinate_names[1:2])")

# 3. :production_bound_search smoke -- gp IS a free KNITRO variable.
res_pb, man_pb = run_profiled_production_outer(:production_bound_search, "smoke_pb", w_calib;
    ctx, evaluate_fn, family_ctx_builder, maxit_override = 4, gp_bounds_halfwidth = 0.02)
record!("production_bound_search_runs", res_pb.n_eval > 0, "n_eval=$(res_pb.n_eval) status=$(res_pb.knitro_status)")
record!("production_bound_search_manifest_includes_gp", man_pb.gp_free && man_pb.n_free == man_ab.n_free + 1,
    "gp_free=$(man_pb.gp_free) n_free=$(man_pb.n_free) (ab n_free=$(man_ab.n_free))")

# For tests 4-5, override production_subsystems with a CONCRETE (non-placeholder) NamedTuple on
# both arms -- otherwise the placeholder guard (test 6 below) correctly fires first, since neither
# man_ab nor man_pb has a real dual_bank_policy/etc recorded yet (task §14/§7's own honesty
# requirement: this branch never fabricates a concrete subsystem value it hasn't actually wired).
concrete_subsystems = (dual_bank_policy = :cold_start, obj_x_reuse_policy = :inherited_from_ab_harness,
    exact_cache_policy = :disabled, screen_set = :default, restriction_backend = :none, solver_options_file = man_ab.knitro_opt_file)
man_ab_concrete = OuterRunManifest(man_ab.mode, man_ab.family, man_ab.label, man_ab.free_coordinate_names,
    man_ab.n_free, man_ab.gp_free, man_ab.hessopt_tag, man_ab.knitro_opt_file, man_ab.stable_layout_digest,
    man_ab.economic_parameterization, concrete_subsystems, man_ab.timestamp)
man_pb_concrete = OuterRunManifest(man_pb.mode, man_pb.family, man_pb.label, man_pb.free_coordinate_names,
    man_pb.n_free, man_pb.gp_free, man_pb.hessopt_tag, man_pb.knitro_opt_file, man_pb.stable_layout_digest,
    man_pb.economic_parameterization, concrete_subsystems, man_pb.timestamp)

# 4. assert_ab_comparable: two manifests differing only in mode/gp_free -- should PASS with allow_gp_free_diff=true.
ok = try
    assert_ab_comparable(man_pb_concrete, man_ab_concrete; allow_gp_free_diff = true)
    true
catch e
    println("  unexpected throw: ", e); false
end
record!("ab_comparable_passes_with_allowed_gp_diff", ok, "man_pb vs man_ab (concrete subsystems), allow_gp_free_diff=true")

# 5. assert_ab_comparable: same pair WITHOUT allow_gp_free_diff -- should THROW (gp_free differs).
threw = false
try
    assert_ab_comparable(man_pb_concrete, man_ab_concrete)  # default allow_gp_free_diff=false
catch e
    global threw = occursin("gp_free differs", sprint(showerror, e))
end
record!("ab_comparable_throws_on_unlabeled_gp_diff", threw, "default allow_gp_free_diff=false")

# 6. assert_ab_comparable: placeholder subsystem fields -> throws (not_yet_wired vs not_yet_wired
#    must NOT be treated as "equal").
threw = false
try
    assert_ab_comparable(man_ab, man_ab)  # identical manifest, but dual_bank_policy=:not_yet_wired on both
catch e
    global threw = occursin("placeholder", sprint(showerror, e))
end
record!("ab_comparable_throws_on_unrecorded_placeholder_fields", threw, "both arms share :not_yet_wired subsystem fields")

# 7. build_profiled_production_config: valid parameterization succeeds, namespace deterministic.
ufctx = family_ctx_builder(ctx, evaluate_fn(w_calib, ctx))
cfg1 = build_profiled_production_config(ufctx; economic_parameterization = :profiled_destination_scales)
cfg2 = build_profiled_production_config(ufctx; economic_parameterization = :profiled_destination_scales)
record!("production_config_namespace_deterministic", cfg1.checkpoint_namespace == cfg2.checkpoint_namespace,
    "cfg1=$(cfg1.checkpoint_namespace) cfg2=$(cfg2.checkpoint_namespace)")

# 8. build_profiled_production_config: invalid parameterization throws.
threw = false
try
    build_profiled_production_config(ufctx; economic_parameterization = :bogus_param)
catch e
    global threw = occursin("not in", sprint(showerror, e))
end
record!("production_config_invalid_parameterization_throws", threw, "economic_parameterization=:bogus_param")

# 9. build_profiled_production_config: full vs profiled parameterization get DIFFERENT namespaces.
cfg_full = build_profiled_production_config(ufctx; economic_parameterization = :full_gamma_normalized)
record!("production_config_namespaces_differ_by_parameterization", cfg_full.checkpoint_namespace != cfg1.checkpoint_namespace,
    "full=$(cfg_full.checkpoint_namespace) profiled=$(cfg1.checkpoint_namespace)")

# 10. assert_checkpoint_compatible: positive + negative.
ok = try
    assert_checkpoint_compatible(cfg1, cfg1.checkpoint_namespace)
    true
catch e
    println("  unexpected throw: ", e); false
end
record!("checkpoint_compatible_positive", ok, "loaded_namespace == cfg1.checkpoint_namespace")

threw = false
try
    assert_checkpoint_compatible(cfg1, cfg_full.checkpoint_namespace)
catch e
    global threw = occursin("mismatch", sprint(showerror, e))
end
record!("checkpoint_compatible_negative_throws", threw, "loading a full-formulation namespace into a profiled cfg")

df = DataFrame(rows)
out_csv = joinpath(dirname(dirname(@__DIR__)), "PROFILED_PRODUCTION_RUNNER_AB_HARDENING_PLUMBING_GATE_2026-08-01.csv")
CSV.write(out_csv, df)
println("\nWrote $out_csv")
show(df, allrows = true, allcols = true)
println()

all_pass = all(r.pass for r in rows)
println("\nPRODUCTION RUNNER + AB HARDENING + PLUMBING GATE: ", all_pass ? "PASS" : "FAIL")
@assert all_pass
