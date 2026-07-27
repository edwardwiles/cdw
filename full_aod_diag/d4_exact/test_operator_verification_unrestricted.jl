# shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: validates the
# newly-built verify_inner_solution_operator_unrestricted! (G=E only, no restriction block) against
# an INDEPENDENT dense recompute (obj.moments! -> fresh Gfull -> r_dense/g_dense), at D=4 and real
# D=20/W=80,000. Unlike the CM-family verifiers (which have an existing archC_verified_state to
# compare against), unrestricted has no equivalent "verified state" helper in this codebase, so this
# gate builds the dense cross-check directly (same recipe compressed_live.jl's own dense tail uses,
# and the same self-validation pattern build_lfix_base_cache!'s own `validate_dense=true` uses).
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_operator_verification_unrestricted.jl [d4|d20]
# ============================================================================
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "compressed_moments.jl", "compressed_cc_inner.jl", "compressed_factual_buffer_reuse.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

all_pass = true
function run_gate(ctx, x_free0; label)
    base = solve_base_state(x_free0, ctx)
    obj = ctx.obj
    W = size(obj.U, 1)
    cf = cf_build(base.θ_full0, ctx; check_ties = true)
    ov = verify_inner_solution_operator_unrestricted!(base.ζstar, base.λstar, cf, obj, W)

    ncore1 = length(base.λstar)
    K = zeros(W); Gfull = zeros(W, obj.d)
    obj.moments!(K, Gfull, base.θ_full0, obj.U, obj)
    r_dense = -base.ζstar .- Gfull[:, 1:ncore1] * base.λstar
    maxerr_r = maximum(abs.(ov.r .- r_dense))

    dPsi_dense = similar(r_dense); obj.dPsi!(dPsi_dense, r_dense)
    g_dense = -(1.0 / W) .* (Gfull[:, 1:ncore1]' * dPsi_dense)
    maxerr_g = maximum(abs.(ov.g_lambda .- g_dense))

    @printf("  %s: max|Δr|=%.3e  max|Δg_lambda|=%.3e  operator_kkt=%.3e  operator_f=%.6f\n",
            label, maxerr_r, maxerr_g, ov.kkt_resid, ov.f)
    ok = maxerr_r < 1e-9 && maxerr_g < 1e-9 && isfinite(ov.f)
    global all_pass &= ok
    @test maxerr_r < 1e-9
    @test maxerr_g < 1e-9
    @test isfinite(ov.f)
    return ok
end

if SCALE == "d4"
    println("=== D=4 unrestricted operator verification gate ===")
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    run_gate(ctx, x_free_calib; label = "D4 calib")
elseif SCALE == "d20"
    println("=== Real D=20/W=80,000 unrestricted operator verification gate ===")
    ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    run_gate(ctx, x_free_calib; label = "D20/W=80000 calib")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

println("\nALL $SCALE unrestricted operator verification gates: ", all_pass ? "PASS" : "FAIL")
all_pass || error("unrestricted operator verification $SCALE gate FAILED")
