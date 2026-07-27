# ============================================================================
# Shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: correctness
# gate for common-Frechet's shared-backend wiring (cm_frechet_production_gradient's
# gradient_backend=:shared_inplace_pooled default vs :legacy_unbuffered reference), at D=4 and
# real D=20/W=80,000. Structure mirrors test_cm_shared_a_gradient_gate.jl (flexible-CM) exactly --
# common-Frechet was the last CM-family restricted wrapper still hardcoded to the allocating
# reference gradient (see PERSISTENT_LFIX_BASE_CACHE_RELEASE_2026-07-27.md /
# SHARED_FG_AND_VERIFICATION_RELEASE_2026-07-27.md "what did NOT change" list).
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_cm_frechet_shared_a_gradient_gate.jl [d4|d20]
# ============================================================================
const D4X = @__DIR__
for f in ["context.jl","context_real_d20.jl","draw_design.jl","winners.jl","oracle.jl",
          "common_marginals_moments.jl","common_marginals_interval.jl","instrumentation.jl","oracle_fast.jl",
          "gravity_elimination.jl","three_way_derivatives.jl","lfix_incremental.jl",
          "composite_gradient.jl","composite_gradient_fast.jl","gradient_workspace.jl","shared_a_gradient.jl",
          "cm_lookup_kernels.jl","lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "nested_quantile_grids.jl","lfix_factorized.jl","lfix_factorized_workspace.jl","lfix_cm_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

all_pass = true
function run_gate(ctx, pe, x_free_calib, x_free_pert; L_list, W_label)
    @testset "common-Frechet ($W_label): :shared_inplace_pooled (default) vs :legacy_unbuffered full-gradient equivalence" begin
        for L in L_list
            pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, cm_hessian_backend = :structured)
            for (label, xf) in (("calib", x_free_calib), ("perturbed", x_free_pert))
                base, verify = archC_frechet_verified_state(xf, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
                @test is_verified_success(verify)

                g_shared, meta_shared = cm_frechet_production_gradient(xf, pcx, ctx, pe; base = base,
                    threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())  # default backend
                g_legacy, meta_legacy = cm_frechet_production_gradient(xf, pcx, ctx, pe; base = base,
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
    println("=== D=4 common-Frechet shared-backend wiring gate ===")
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    pe = build_pivot_elimination(ctx)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    Random.seed!(778)
    x_free_pert = copy(x_free_calib)
    x_free_pert[2:end] .*= exp.(0.04 .* randn(length(x_free_pert) - 1))
    run_gate(ctx, pe, x_free_calib, x_free_pert; L_list = (10, 50), W_label = "D4")
elseif SCALE == "d20"
    println("=== Real D=20/W=80,000 common-Frechet shared-backend wiring gate ===")
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

println("\nALL $SCALE common-Frechet shared-backend gates: ", all_pass ? "PASS" : "FAIL")
all_pass || error("common-Frechet shared-backend $SCALE gate FAILED")
