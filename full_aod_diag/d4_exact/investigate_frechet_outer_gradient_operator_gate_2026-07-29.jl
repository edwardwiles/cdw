# Investigation (2026-07-29), part 2: does common-Frechet's PRODUCTION OUTER GRADIENT path
# (cm_frechet_production_gradient_cplus -> archC_frechet_base_state -> build_lfix_base_cache_cm_frechet_C!
# -> composite_gradient_at_Cplus_from_cache) produce the SAME gradient under moment_representation=
# :operator as under :dense_reference, at real D=20 scale, at the calibration AND non-calibration
# outer points? The inner dual solve itself was already confirmed exact
# (investigate_frechet_operator_nonlocal_gate_2026-07-29.jl); this closes the remaining question of
# whether the OUTER coordinate-gradient loop (never previously tested under :operator for this
# family) introduces any dense-H/G dependency or numerical discrepancy.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "context_real_d20.jl",
          "gradient_workspace.jl", "lfix_factorized_workspace.jl", "lfix_factorized.jl", "cm_screen_bridge.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Random

lp(xs...) = (println(xs...); flush(stdout))

lp("Building real D=20 context (W=80000, delta=1.0, destination_sample=:exclude_row)...")
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D; W = size(ctx.U, 1)
Random.seed!(2026)
L = 50

x_free_near = x_free_calib .+ vcat(0.005, 0.01 .* randn(length(x_free_calib) - 1))
x_free_hard = x_free_calib .* 1.01

ALL_OK = Ref(true)

for contrasts in (:anchored, :orthonormal)
    println("="^90); println("contrasts=$contrasts L=$L"); println("="^90); flush(stdout)

    pcx_d = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup,
        moment_representation = :dense_reference)
    pcx_o = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup,
        moment_representation = :operator)

    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, Ddest, W)

    for (label, x_free) in (("calib", x_free_calib), ("near_delta1_perturbed", x_free_near), ("hard_point_x1.01", x_free_hard))
        print("  $label: dense gradient...  "); flush(stdout)
        t0 = time()
        g_d, meta_d = cm_frechet_production_gradient_cplus(x_free, pcx_d, ctx, pe, pool, ws;
            threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        dt_d = time() - t0
        @printf "done (%.1fs)\n" dt_d

        print("  $label: operator gradient...  "); flush(stdout)
        t0 = time()
        g_o, meta_o = cm_frechet_production_gradient_cplus(x_free, pcx_o, ctx, pe, pool, ws;
            threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        dt_o = time() - t0
        @printf "done (%.1fs)\n" dt_o

        gdiff = maximum(abs.(g_d .- g_o))
        grel = gdiff / max(1.0, maximum(abs.(g_d)))
        ok = gdiff < 1e-6
        ALL_OK[] &= ok
        @printf "    max|Δg|=%.3e  rel=%.3e  n_g=%d  pass=%s\n" gdiff grel length(g_d) ok
        flush(stdout)
    end
end

println()
println(ALL_OK[] ? "ALL OUTER-GRADIENT OPERATOR-VS-DENSE CHECKS PASS" : "SOME OUTER-GRADIENT CHECKS FAILED")
