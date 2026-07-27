# ============================================================================
# Shared-FG-verification-and-A-gradient release (2026-07-27): correctness gate for flexible-CM's
# shared-backend wiring (cm_production_gradient's gradient_backend=:shared_inplace_pooled default
# vs :legacy_unbuffered reference), at D=4 and real D=20/W=80,000. Structure mirrors
# test_cm_meanzc_shared_a_gradient_gate.jl / test_originzc_shared_a_gradient_gate.jl; the
# (x_free_calib, x_free_pert) construction is lifted from test_phaseB1_cmlookup_production_
# correctness.jl's own proven-working pattern.
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_cm_shared_a_gradient_gate.jl [d4|d20]
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "shared_a_gradient.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Test, Printf, LinearAlgebra, Random

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

all_pass = true
function run_gate(ctx, pe, x_free_calib, x_free_pert; L_list, W_label)
    @testset "flexible-CM ($W_label): :shared_inplace_pooled (default) vs :legacy_unbuffered full-gradient equivalence" begin
        for L in L_list
            pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
            for (label, xf) in (("calib", x_free_calib), ("perturbed", x_free_pert))
                base, verify = archC_verified_state(xf, pcx.ctx_cm, pcx.cctx)
                @test is_verified_success(verify)

                g_shared, meta_shared = cm_production_gradient(xf, pcx, ctx, pe; base = base,
                    threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())  # default backend
                g_legacy, meta_legacy = cm_production_gradient(xf, pcx, ctx, pe; base = base,
                    gradient_backend = :legacy_unbuffered, threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

                D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
                @test length(g_shared) == D * Ddest
                @test length(g_legacy) == D * Ddest
                ok = g_shared == g_legacy
                global all_pass &= ok
                maxdiff = maximum(abs.(g_shared .- g_legacy))
                @printf("  %s L=%d %s  max|Δg|=%.3e  bit-identical=%s\n", W_label, L, label, maxdiff, ok)
                @test ok
            end
        end
    end
end

if SCALE == "d4"
    println("=== D=4 flexible-CM shared-backend wiring gate ===")
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    pe = build_pivot_elimination(ctx)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    Random.seed!(778)
    x_free_pert = copy(x_free_calib)
    x_free_pert[2:end] .*= exp.(0.04 .* randn(length(x_free_pert) - 1))
    run_gate(ctx, pe, x_free_calib, x_free_pert; L_list = (10, 50), W_label = "D4")
elseif SCALE == "d20"
    println("=== Real D=20/W=80,000 flexible-CM shared-backend wiring gate ===")
    ctx = d20_real_setup_design(W = 80_000, δ = 1.0, find_smallest = true,
                                 draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
    pe = build_pivot_elimination(ctx)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    Random.seed!(778)
    x_free_pert = copy(x_free_calib)
    x_free_pert[2:end] .*= exp.(0.02 .* randn(length(x_free_pert) - 1))
    run_gate(ctx, pe, x_free_calib, x_free_pert; L_list = (50,), W_label = "D20/W=80000")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

println("\nALL $SCALE flexible-CM shared-backend gates: ", all_pass ? "PASS" : "FAIL")
all_pass || error("flexible-CM shared-backend $SCALE gate FAILED")
