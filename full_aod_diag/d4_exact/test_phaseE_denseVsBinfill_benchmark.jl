# Phase E remediation (production-audit continuation, 2026-07-26), part 2: benchmark CM+ZC's
# persistent-dense-CM-columns approach (precalc_common_marginals_cdf, cm_meanzc_moments.jl:425,
# "CM, z, origins = precalc_common_marginals_cdf(...)" then "G[:,cm_cols] .= CM[1:n,:]") vs plain
# flexible-CM's bin-recompute/in-place-fill approach (fill_cm_columns_from_bins!,
# cm_hessian_architectures.jl) at real D=20/W=80,000/L=50 production scale. Task's own explicit
# instruction: "retain the measured winner" for the representation that is actually faster; "do
# not change that representation merely for stylistic uniformity" if benchmarking doesn't show a
# real difference.
#
# Times obj.moments! DIRECTLY (the actual per-outer-point column-construction closure both
# families' inner solve depends on), not the whole inner solve -- isolates the construction cost
# this task is actually asking about. CM+ZC's own moments! call necessarily does strictly more
# work than flexible-CM's (extra mean/pair columns on top of the same CM block), so this is NOT a
# perfectly isolated "CM-columns-only" comparison -- reported honestly as such -- but it IS the
# real, actual cost each family's production driver actually pays per outer point.
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
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, Statistics

lp(xs...) = (println(xs...); flush(stdout))

const W = 80_000
const L = 50
const NREP = 8

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

function bench_moments(obj, θ_full, H, nrep)
    G = CS.select_G_from_H(obj, H)
    times = Float64[]; allocs = Int[]
    obj.moments!(@view(H[:, 1]), G, θ_full, obj.U, obj)   # warm-up
    for i in 1:nrep
        t0 = time()
        b = @allocated obj.moments!(@view(H[:, 1]), G, θ_full, obj.U, obj)
        push!(times, time() - t0)
        push!(allocs, b)
    end
    return times, allocs
end

lp(">>> Building plain flexible-CM context (bin-fill CM columns) L=$L W=$W ...")
pcx_flex = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
θ_full_flex = CS.reconstruct_full(x_free_calib, pcx_flex.ctx_cm.m)
t_flex, a_flex = bench_moments(pcx_flex.ctx_cm.obj, θ_full_flex, pcx_flex.ctx_cm.obj.H, NREP)

lp(">>> Building CM+ZC context (persistent dense CM columns) K_mean=1 K_pair=1 L=$L W=$W ...")
pcx_zc = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = 1, K_pair = 1, contrasts = :anchored)
θ_full_zc = CS.reconstruct_full(x_free_calib, pcx_zc.ctx_cm.m)
νvec = [1.0]   # K_mean=1
θ_ext_zc = vcat(θ_full_zc, νvec)
t_zc, a_zc = bench_moments(pcx_zc.ctx_cm.obj, θ_ext_zc, pcx_zc.ctx_cm.obj.H, NREP)

@printf "\n==================== DENSE-vs-BINFILL BENCHMARK (D=20, W=%d, L=%d) ====================\n" W L
@printf "flexible_cm (bin-fill CM columns)      : median=%.5fs  mean=%.5fs  median_alloc=%.2fMB\n" median(t_flex) mean(t_flex) median(a_flex)/1e6
@printf "cm_meanzc   (dense-copy CM + mean/pair) : median=%.5fs  mean=%.5fs  median_alloc=%.2fMB\n" median(t_zc) mean(t_zc) median(a_zc)/1e6
lp(">>> NOTE: cm_meanzc's moments! call necessarily does strictly MORE work (extra mean/pair columns")
lp(">>> on top of the same CM block) -- this is the real production cost each family pays, not a")
lp(">>> perfectly isolated CM-columns-only comparison. See release doc for interpretation.")
lp(">>> BENCHMARK_DONE")
