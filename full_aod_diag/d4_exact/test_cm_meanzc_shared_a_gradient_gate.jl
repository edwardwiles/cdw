# ============================================================================
# Shared-FG-verification-and-A-gradient release (2026-07-27): correctness gate for CM+ZC's
# shared-backend wiring (cm_meanzc_production_gradient's gradient_backend=:shared_inplace_pooled
# default vs :legacy_unbuffered reference), at D=4 and real D=20/W=80,000. Structure mirrors
# test_originzc_shared_a_gradient_gate.jl's own established gate pattern; the (x_free_calib,
# x_free_pert, νvec0) construction is lifted verbatim from test_meanzc_operator_correctness.jl's
# own proven-working pattern (a naive from-scratch (x_free0, nu) construction for this family is a
# known trap -- see test_cm_meanzc_regression.jl's own header note on this exact point).
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_cm_meanzc_shared_a_gradient_gate.jl [d4|d20]
# ============================================================================
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "gradient_workspace.jl",
          "shared_a_gradient.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

all_pass = true
function run_gate(ctx, pe, x_free_calib; L::Int, K_configs, W_label)
    @testset "CM+ZC ($W_label): :shared_inplace_pooled (default) vs :legacy_unbuffered full-gradient equivalence" begin
        for (K_mean, K_pair) in K_configs
            νvec0 = [Float64(factorial(k)) for k in 1:K_mean]
            pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
                                                       contrasts = :anchored, meanzc_basis = :direct)
            _, base0, verify0 = cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx)
            @test is_verified_success(verify0)

            g_shared, meta_shared = cm_meanzc_production_gradient(x_free_calib, νvec0, pcx, ctx, pe; base = base0, verify = verify0,
                threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())  # default backend
            g_legacy, meta_legacy = cm_meanzc_production_gradient(x_free_calib, νvec0, pcx, ctx, pe; base = base0, verify = verify0,
                gradient_backend = :legacy_unbuffered, threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

            D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
            @test length(g_shared) == D * Ddest + K_mean
            @test length(g_legacy) == D * Ddest + K_mean
            ok = g_shared == g_legacy
            global all_pass &= ok
            maxdiff = maximum(abs.(g_shared .- g_legacy))
            @printf("  %s K=%d/%d  max|Δg|=%.3e  bit-identical=%s\n", W_label, K_mean, K_pair, maxdiff, ok)
            @test ok
        end
    end
end

if SCALE == "d4"
    println("=== D=4 CM+ZC shared-backend wiring gate ===")
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    pe = build_pivot_elimination(ctx)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    run_gate(ctx, pe, x_free_calib; L = 10, K_configs = [(1, 0), (1, 1), (2, 2)], W_label = "D4")
elseif SCALE == "d20"
    println("=== Real D=20/W=80,000 CM+ZC shared-backend wiring gate ===")
    W = 80000
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                                 draw_seed = 20260719, destination_sample = :exclude_row)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; Ddest = ctx.D_dest
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0 * 1.01, zfree0)
    x_free_calib = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe))))
    run_gate(ctx, pe, x_free_calib; L = 50, K_configs = [(1, 1)], W_label = "D20/W=$W")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

println("\nALL $SCALE CM+ZC shared-backend gates: ", all_pass ? "PASS" : "FAIL")
all_pass || error("CM+ZC shared-backend $SCALE gate FAILED")
