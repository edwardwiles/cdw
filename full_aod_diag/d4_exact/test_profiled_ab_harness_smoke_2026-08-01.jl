# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream): SMOKE test
# for PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl -- proves the
# family-generic harness mechanically works (short D4 run, maxit capped) via
# the unrestricted arm. NOT a campaign (task §19: "Do not launch a production
# campaign") -- a handful of outer iterations only, to confirm the harness
# wiring (evaluate_fn/family_ctx_builder/shared_family_outer_gradient/KNITRO
# callback plumbing) is correct before the inner branch's restricted-family
# evaluators are ready to plug in.
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
include(joinpath(@__DIR__, "PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl"))
using Random

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)

Random.seed!(11)
w_start = copy(w_calib); w_start[2:end] .+= 0.005 .* randn(length(w_calib) - 1)

evaluate_fn, family_ctx_builder = unrestricted_ab_arm(ctx, spec, pe)

result = run_profiled_family_outer_search("smoke_unrestricted", w_start; ctx = ctx,
    evaluate_fn = evaluate_fn, family_ctx_builder = family_ctx_builder,
    maxtime_real = 60.0, maxit_override = 8)

println("\nSMOKE RESULT: family=$(result.family) status=$(result.knitro_status) n_eval=$(result.n_eval) n_grad=$(result.n_grad_calls) wall=$(result.wall_ext)s")
ok = result.n_eval > 0 && result.n_grad_calls > 0 && result.family == :unrestricted
println("HARNESS SMOKE TEST: ", ok ? "PASS" : "FAIL")

# Also confirm a not-ready restricted arm fails LOUDLY, not silently.
threw = false
try
    ev_fn2, fctx_fn2 = flexible_cm_ab_arm()
    ev_fn2(w_start, ctx)
catch e
    global threw = occursin("not yet exposed", sprint(showerror, e))
end
println("restricted-arm-not-ready throws loudly: ", threw ? "PASS" : "FAIL")

ok || error("harness smoke test failed")
threw || error("restricted arm should have thrown a loud not-ready error")
