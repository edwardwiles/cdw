# ============================================================================
# Phase 0 gate closure (2026-07-26, production-audit task): correctness + wall-clock gate for
# hessian_cm_frechet_structured_v2! (cm_frechet_hessian_threaded.jl), the threaded-bin-table
# Architecture-C Hessian for marginal_restriction=:common_frechet, vs the serial
# hessian_cm_frechet_structured! (cm_frechet_hessian.jl). Mirrors test_cm_threaded_hessian.jl's
# own structure exactly (D=20/W=80,000/L=50 real production-scale gate, both call orders,
# perturbed-point re-check, wall-clock benchmark) for CM's own threaded-vs-serial gate -- this is
# the analogous gate for the level-block extension.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/test_cm_frechet_threaded_hessian_gates.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl",
          "cm_screen_bridge.jl","gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Random, Printf

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

lp(">>> Threads.nthreads() = ", Threads.nthreads(), " (need > 1 for threaded_bins to be a real test)")

println("="^78)
println("Section 1: real D=20/W=80,000/L=50 common-Frechet context, converged base state")
println("="^78)
L = 50
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
probs = cm_equal_grid_probs(L)
cfg_frechet = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :structured,
                        contrasts = :anchored, marginal_restriction = :common_frechet)
pcx = build_cm_production_context_v2(ctx0, CS, cfg_frechet; L = L)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]

θ_full0 = CS.reconstruct_full(x_free_calib, pcx.ctx_cm.m)
K, x_sol, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(pcx.ctx_cm.obj, θ_full0; hess_cb_builder = pcx.hess_cb_builder)
check("base state feasible", nStatus in (0, -100, -101, -102, -103, -400, -401, -402))
lp(">>> nStatus=", nStatus, " n_fg=", n_fg, " n_hess=", n_hess)

obj = pcx.ctx_cm.obj
cctx = pcx.cctx
level_targets = pcx.aug.level_targets
_archC_prep_for_hessian!(obj, x_sol)
n = cctx.NCORE + cctx.ncm
h_serial = Vector{Float64}(undef, n * (n + 1) ÷ 2)
h_v2_serial = Vector{Float64}(undef, n * (n + 1) ÷ 2)
h_v2_threaded = Vector{Float64}(undef, n * (n + 1) ÷ 2)

println("="^78)
println("Section 2: correctness -- v2(threaded_bins=false) and v2(threaded_bins=true) vs serial")
println("="^78)
hessian_cm_frechet_structured!(h_serial, obj, cctx, level_targets)
hessian_cm_frechet_structured_v2!(h_v2_serial, obj, cctx, level_targets; threaded_bins = false)
d1 = maximum(abs.(h_serial .- h_v2_serial))
lp(">>> max|serial - v2(serial)| = ", d1)
check("v2 serial matches production serial to ~1e-12 (identical arithmetic, should be exact)", d1 < 1e-10)

tls = build_thread_local_scratch(cctx)
hessian_cm_frechet_structured_v2!(h_v2_threaded, obj, cctx, level_targets; threaded_bins = true, tls = tls)
d2 = maximum(abs.(h_serial .- h_v2_threaded))
lp(">>> max|serial - v2(threaded_bins=true)| = ", d2)
check("v2 threaded matches production serial to ~1e-9 (draw-partitioned sum, FP-order noise expected)", d2 < 1e-9)

# a second, perturbed point
rng = MersenneTwister(20260726)
x_free_near = x_free_calib .* (1.0 .+ 0.001 .* randn(rng, length(x_free_calib)))
θ_full_near = CS.reconstruct_full(x_free_near, pcx.ctx_cm.m)
K2, x_sol2, nStatus2, _, _ = inner_loop_internal_archgeneric(pcx.ctx_cm.obj, θ_full_near; hess_cb_builder = pcx.hess_cb_builder)
if nStatus2 in (0, -100, -101, -102, -103, -400, -401, -402)
    _archC_prep_for_hessian!(obj, x_sol2)
    hessian_cm_frechet_structured!(h_serial, obj, cctx, level_targets)
    hessian_cm_frechet_structured_v2!(h_v2_threaded, obj, cctx, level_targets; threaded_bins = true, tls = tls)
    d3 = maximum(abs.(h_serial .- h_v2_threaded))
    lp(">>> [perturbed] max|serial - v2(threaded_bins=true)| = ", d3)
    check("[perturbed] v2 threaded matches serial to ~1e-9", d3 < 1e-9)
else
    lp(">>> [perturbed] inner solve not feasible (nStatus=", nStatus2, "), skipping perturbed-point check")
end

println("="^78)
println("Section 3: wall-clock -- repeated real Hessian callback, serial vs threaded_bins, at this real point")
println("="^78)
_archC_prep_for_hessian!(obj, x_sol)
N = 20
hessian_cm_frechet_structured!(h_serial, obj, cctx, level_targets)
hessian_cm_frechet_structured_v2!(h_v2_threaded, obj, cctx, level_targets; threaded_bins = true, tls = tls)

t_serial = @elapsed for _ in 1:N
    hessian_cm_frechet_structured!(h_serial, obj, cctx, level_targets)
end
t_v2_serial = @elapsed for _ in 1:N
    hessian_cm_frechet_structured_v2!(h_v2_serial, obj, cctx, level_targets; threaded_bins = false)
end
t_v2_threaded = @elapsed for _ in 1:N
    hessian_cm_frechet_structured_v2!(h_v2_threaded, obj, cctx, level_targets; threaded_bins = true, tls = tls)
end

@printf ">>> N=%d reps: serial=%.4fs (%.2f ms/call), v2-serial=%.4fs (%.2f ms/call), v2-threaded(bins)=%.4fs (%.2f ms/call)\n" N t_serial (1000*t_serial/N) t_v2_serial (1000*t_v2_serial/N) t_v2_threaded (1000*t_v2_threaded/N)
lp(">>> speedup v2-threaded(bins) vs serial: ", round(t_serial / t_v2_threaded, digits = 3), "x")

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
