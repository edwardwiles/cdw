# Phase B4 remediation (production-audit continuation, 2026-07-26): performance gate for
# inner_fg_backend=:cm_lookup vs :dense_reference at real D=20/W=80,000/L=50 production scale.
# Measures complete inner-solve wall time and allocations (archC_verified_state, the real
# production call path) with warm-up, both backends, several repetitions at the calibration point.
#
# Usage: julia --project=. -t 8 full_aod_diag/d4_exact/test_phaseB1_performance_gate.jl
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
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, Statistics

lp(xs...) = (println(xs...); flush(stdout))

const W = 80_000
const L = 50
const NREP = 5

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

function bench_backend(backend::Symbol; nrep = NREP)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, inner_fg_backend = backend)
    # warm-up (JIT)
    base, verify = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
    @assert base.inner_status == 0 "warm-up solve failed for backend=$backend"
    times = Float64[]
    allocs = Int[]
    for i in 1:nrep
        t0 = time()
        b = @allocated (base, verify) = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
        push!(times, time() - t0)
        push!(allocs, b)
    end
    return (times = times, allocs = allocs, Delta_dual = verify.Delta_dual)
end

lp(">>> Benchmarking inner_fg_backend=:dense_reference (production baseline) at D=20/W=$W/L=$L ...")
r_dense = bench_backend(:dense_reference)
lp(">>> Benchmarking inner_fg_backend=:cm_lookup (Phase B1 candidate) at D=20/W=$W/L=$L ...")
r_lookup = bench_backend(:cm_lookup)

@printf "\n==================== PERFORMANCE GATE RESULT (D=20, W=%d, L=%d, calib point) ====================\n" W L
@printf "dense_reference : median=%.4fs  mean=%.4fs  min=%.4fs  max=%.4fs  median_alloc=%.1fMB  Delta_dual=%.8f\n" median(r_dense.times) mean(r_dense.times) minimum(r_dense.times) maximum(r_dense.times) median(r_dense.allocs)/1e6 r_dense.Delta_dual
@printf "cm_lookup       : median=%.4fs  mean=%.4fs  min=%.4fs  max=%.4fs  median_alloc=%.1fMB  Delta_dual=%.8f\n" median(r_lookup.times) mean(r_lookup.times) minimum(r_lookup.times) maximum(r_lookup.times) median(r_lookup.allocs)/1e6 r_lookup.Delta_dual
speedup = median(r_dense.times) / median(r_lookup.times)
alloc_ratio = median(r_dense.allocs) / max(1, median(r_lookup.allocs))
@printf "speedup (dense/lookup) = %.3fx   allocation ratio (dense/lookup) = %.3fx\n" speedup alloc_ratio
@printf "Delta_dual agreement: %.3e\n" abs(r_dense.Delta_dual - r_lookup.Delta_dual)
lp(">>> GATE_DONE")
