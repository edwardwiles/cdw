# Phase 9 gate (integration/profiled-all-five-production-closeout, 2026-08-02): real D=20
# unrestricted operator verification gate at a CONFIGURABLE W (ENV["UNRESTRICTED_D20_W"], default
# 20000), so the same script covers both required D20 points (W=20k, W=80k) without duplication.
# Purely additive -- test_operator_verification_unrestricted.jl (which hardcodes W=80,000 for its
# own "d20" mode) is left completely unmodified; this file reuses the exact same `run_gate` method
# (dense recompute cross-check against `verify_inner_solution_operator_unrestricted!`), just with
# W read from the environment instead of hardcoded.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "compressed_moments.jl", "compressed_cc_inner.jl", "compressed_factual_buffer_reuse.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra
flush(stdout)

const W_VAL = parse(Int, get(ENV, "UNRESTRICTED_D20_W", "20000"))

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

    @printf("  %s: nStatus=%d  max|Δr|=%.3e  max|Δg_lambda|=%.3e  operator_kkt=%.3e  operator_f=%.6f\n",
            label, base.inner_status, maxerr_r, maxerr_g, ov.kkt_resid, ov.f)
    flush(stdout)
    ok = maxerr_r < 1e-9 && maxerr_g < 1e-9 && isfinite(ov.f) && base.inner_status == 0
    global all_pass &= ok
    @test maxerr_r < 1e-9
    @test maxerr_g < 1e-9
    @test isfinite(ov.f)
    @test base.inner_status == 0
    return ok
end

println("="^90); println("Real D=20/W=$W_VAL unrestricted operator verification gate"); println("="^90); flush(stdout)
t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]
t_solve = @elapsed run_gate(ctx, x_free_calib; label = "D20/W=$W_VAL calib")
@printf("solve+verify: %.2fs\n", t_solve); flush(stdout)

println("\nALL D20/W=$W_VAL unrestricted operator verification gates: ", all_pass ? "PASS" : "FAIL")
all_pass || error("unrestricted operator verification D20/W=$W_VAL gate FAILED")
