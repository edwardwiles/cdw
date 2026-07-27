# verification-defaults task (2026-07-27), Section 6.1: full comparison gate for unrestricted's
# NEWLY-WIRED evaluate_fullA_fast_compressed(...; verification_backend=:operator) against the
# pre-existing :dense_reference path, at D=4 and (SCALE=d20) real D=20/W=80,000. Unlike the 4
# CM-family gates (which call each family's own *_verified_state directly), unrestricted's
# verification lives inside evaluate_fullA_fast_compressed's own result-tuple tail -- both calls
# below go through THAT function (cache=nothing, so neither run can hit a cache populated by the
# other), and `compare_verify_tuples` is run directly against each call's full `result` NamedTuple
# (it already carries every field classify_inner_result/is_cacheable_result/is_verified_success
# need). Draw-level dual index / complete dual gradient are checked via the standalone
# verify_inner_solution_operator_unrestricted! call against a fresh dense obj.moments! recompute,
# mirroring test_operator_verification_unrestricted.jl's own established pattern for this family
# (it has no archC-style verified-state helper to piggyback on).
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_verification_backend_default_unrestricted.jl [d4|d20]
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "compressed_moments.jl", "compressed_cc_inner.jl", "compressed_factual_buffer_reuse.jl",
          "core_exact_hessian.jl", "structured_moment_build.jl", "compressed_live.jl",
          "operator_verification.jl", "verification_gate_utils.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

function run_gate(ctx, x_free0; label)
    W = size(ctx.obj.U, 1)

    result_d, _ = evaluate_fullA_fast_compressed(x_free0, ctx; cache = nothing, use_cache = false, verification_backend = :dense_reference)
    result_o, _ = evaluate_fullA_fast_compressed(x_free0, ctx; cache = nothing, use_cache = false, verification_backend = :operator)

    gcheck("$label: same inner-solve dual point (zeta/lambda unaffected by verification_backend)",
           result_d.zeta == result_o.zeta && result_d.lambda == result_o.lambda)

    compare_verify_tuples(label, result_d, result_o)
    @printf "  %s  dense_Delta_dual=%.10f  operator_Delta_dual=%.10f\n" label result_d.Delta_dual result_o.Delta_dual

    # Draw-level dual index + complete dual gradient: standalone operator verifier against a fresh
    # independent dense obj.moments! recompute (unrestricted has no archC-style G already built to
    # reuse -- same recipe test_operator_verification_unrestricted.jl already validated).
    base = solve_base_state(x_free0, ctx)
    obj = ctx.obj
    cf = cf_build(base.θ_full0, ctx; check_ties = true)
    ov = verify_inner_solution_operator_unrestricted!(base.ζstar, base.λstar, cf, obj, W)
    ncore1 = length(base.λstar)
    K = zeros(W); Gfull = zeros(W, obj.d)
    obj.moments!(K, Gfull, base.θ_full0, obj.U, obj)
    r_dense = -base.ζstar .- Gfull[:, 1:ncore1] * base.λstar
    maxerr_r = maximum(abs.(ov.r .- r_dense))
    gcheck("$label: draw-level dual index (r) agrees (max|Δr|=$maxerr_r)", maxerr_r < 1e-9)

    dPsi_dense = similar(r_dense); obj.dPsi!(dPsi_dense, r_dense)
    g_dense = -(1.0 / W) .* (Gfull[:, 1:ncore1]' * dPsi_dense)
    maxerr_g = maximum(abs.(ov.g_lambda .- g_dense))
    gcheck("$label: complete dual gradient (full vector) agrees (max|Δg|=$maxerr_g)", maxerr_g < 1e-9)
end

if SCALE == "d4"
    println("=== D=4 unrestricted verification_backend gate (evaluate_fullA_fast_compressed) ===")
    flush(stdout)
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    run_gate(ctx, x_free_calib; label = "unrestricted D4 calib")
elseif SCALE == "d20"
    println("=== Real D=20/W=80,000 unrestricted verification_backend gate (evaluate_fullA_fast_compressed) ===")
    flush(stdout)
    ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    run_gate(ctx, x_free_calib; label = "unrestricted D20/W=80000 calib")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

println()
println("Gate tally: ", GATE_TALLY.n_pass, "/", GATE_TALLY.n_checks, " checks passed")
GATE_TALLY.n_pass == GATE_TALLY.n_checks ? println("ALL PASS") : (println("SOME FAILURES"); exit(1))
