# Phase 5.2 remediation (2026-07-26) performance gate: common-Frechet inner_fg_backend=
# :cm_frechet_lookup vs :dense_reference at real D=20/W=80,000/L=50, through the real production
# call path (archC_frechet_verified_state). Mirrors test_phaseB1_performance_gate.jl exactly.
#
# Usage: julia --project=. -t N full_aod_diag/d4_exact/test_phase52_frechet_performance_gate.jl
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, Statistics

lp(xs...) = (println(xs...); flush(stdout))

const W = 80_000
const L = 50
const NREP = 5

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

function bench_backend(backend::Symbol; nrep = NREP)
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored,
        cm_hessian_backend = :structured, inner_fg_backend = backend)
    base, verify = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    @assert base.inner_status == 0 "warm-up solve failed for backend=$backend"
    times = Float64[]
    allocs = Int[]
    for i in 1:nrep
        t0 = time()
        b = @allocated (base, verify) = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
        push!(times, time() - t0)
        push!(allocs, b)
    end
    return (times = times, allocs = allocs, Delta_dual = verify.Delta_dual)
end

lp(">>> Benchmarking inner_fg_backend=:dense_reference (production baseline) at D=20/W=$W/L=$L ...")
r_dense = bench_backend(:dense_reference)
lp(">>> Benchmarking inner_fg_backend=:cm_frechet_lookup (Phase 5.2 candidate) at D=20/W=$W/L=$L ...")
r_lookup = bench_backend(:cm_frechet_lookup)

@printf "\n==================== PERFORMANCE GATE RESULT (D=20, W=%d, L=%d, calib point) ====================\n" W L
@printf "dense_reference    : median=%.4fs  mean=%.4fs  min=%.4fs  max=%.4fs  median_alloc=%.1fMB  Delta_dual=%.8f\n" median(r_dense.times) mean(r_dense.times) minimum(r_dense.times) maximum(r_dense.times) median(r_dense.allocs)/1e6 r_dense.Delta_dual
@printf "cm_frechet_lookup  : median=%.4fs  mean=%.4fs  min=%.4fs  max=%.4fs  median_alloc=%.1fMB  Delta_dual=%.8f\n" median(r_lookup.times) mean(r_lookup.times) minimum(r_lookup.times) maximum(r_lookup.times) median(r_lookup.allocs)/1e6 r_lookup.Delta_dual
speedup = median(r_dense.times) / median(r_lookup.times)
alloc_ratio = median(r_dense.allocs) / max(1, median(r_lookup.allocs))
@printf "speedup (dense/lookup) = %.3fx   allocation ratio (dense/lookup) = %.3fx\n" speedup alloc_ratio
@printf "Delta_dual agreement: %.3e\n" abs(r_dense.Delta_dual - r_lookup.Delta_dual)
lp(">>> GATE_DONE")
