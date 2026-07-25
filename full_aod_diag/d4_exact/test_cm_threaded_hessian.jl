# ============================================================================
# Correctness + wall-clock benchmark for sections 6.1/6.2: hessian_cm_structured_v2!
# (cm_hessian_threaded.jl), the threaded-bin-table Architecture-C Hessian, vs the (now also
# Hraw_CC-fixed) serial hessian_cm_structured! (cm_hessian_architectures.jl).
#
# Both operate on the SAME CMBinHessCtx real production uses (built via build_cm_bin_ctx) --
# unlike cm_hessian_architecture_threaded.jl's separate hessian_cm_structured_threaded!, which
# works with a DIFFERENT, non-production CMConfig/build_cm_production_context_v2 system (confirmed
# NOT what run_cm_upper_checkpointed calls -- see this task's own investigation).
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/test_cm_threaded_hessian.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl",
          "cm_screen_bridge.jl","lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl","cm_checkpoint.jl"]
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
println("Section 1: real D=20/W=80,000/L=50 CM context, converged base state")
println("="^78)
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[50]
pcx = build_cm_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, probs = probs)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
base, verify = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
check("base state feasible", base.inner_status in (0, -100, -101, -103))

obj = pcx.ctx_cm.obj
cctx = pcx.cctx
_archC_prep_for_hessian!(obj, vcat(base.ζstar, base.λstar))
n = cctx.NCORE + cctx.ncm
h_serial = Vector{Float64}(undef, n * (n + 1) ÷ 2)
h_v2_serial = Vector{Float64}(undef, n * (n + 1) ÷ 2)
h_v2_threaded = Vector{Float64}(undef, n * (n + 1) ÷ 2)

println("="^78)
println("Section 2: correctness -- v2(threaded_bins=false) and v2(threaded_bins=true) vs serial")
println("="^78)
hessian_cm_structured!(h_serial, obj, cctx)
hessian_cm_structured_v2!(h_v2_serial, obj, cctx; threaded_bins = false, use_syrk = true)
d1 = maximum(abs.(h_serial .- h_v2_serial))
lp(">>> max|serial - v2(serial,syrk)| = ", d1)
check("v2 serial (syrk) matches production serial (gemm) to ~1e-10", d1 < 1e-10)

tls = build_thread_local_scratch(cctx)
hessian_cm_structured_v2!(h_v2_threaded, obj, cctx; threaded_bins = true, tls = tls, use_syrk = true)
d2 = maximum(abs.(h_serial .- h_v2_threaded))
lp(">>> max|serial - v2(threaded_bins=true)| = ", d2)
check("v2 threaded matches production serial to ~1e-9 (draw-partitioned sum, FP-order noise expected)", d2 < 1e-9)

# a second, perturbed point
rng = MersenneTwister(20260725)
x_free_near = x_free_calib .* (1.0 .+ 0.001 .* randn(rng, length(x_free_calib)))
base2, verify2 = archC_verified_state(x_free_near, pcx.ctx_cm, pcx.cctx)
if base2.inner_status in (0, -100, -101, -103)
    _archC_prep_for_hessian!(obj, vcat(base2.ζstar, base2.λstar))
    hessian_cm_structured!(h_serial, obj, cctx)
    hessian_cm_structured_v2!(h_v2_threaded, obj, cctx; threaded_bins = true, tls = tls, use_syrk = true)
    d3 = maximum(abs.(h_serial .- h_v2_threaded))
    lp(">>> [perturbed] max|serial - v2(threaded_bins=true)| = ", d3)
    check("[perturbed] v2 threaded matches serial to ~1e-9", d3 < 1e-9)
end

println("="^78)
println("Section 3: wall-clock -- repeated real Hessian callback, serial vs threaded_bins, at this real point")
println("="^78)
_archC_prep_for_hessian!(obj, vcat(base.ζstar, base.λstar))
N = 20
# warm-up (JIT) not counted
hessian_cm_structured!(h_serial, obj, cctx)
hessian_cm_structured_v2!(h_v2_threaded, obj, cctx; threaded_bins = true, tls = tls, use_syrk = true)

t_serial = @elapsed for _ in 1:N
    hessian_cm_structured!(h_serial, obj, cctx)
end
t_v2_serial = @elapsed for _ in 1:N
    hessian_cm_structured_v2!(h_v2_serial, obj, cctx; threaded_bins = false, use_syrk = true)
end
t_v2_threaded = @elapsed for _ in 1:N
    hessian_cm_structured_v2!(h_v2_threaded, obj, cctx; threaded_bins = true, tls = tls, use_syrk = true)
end

@printf ">>> N=%d reps: serial(gemm)=%.4fs (%.2f ms/call), v2-serial(syrk)=%.4fs (%.2f ms/call), v2-threaded(bins)=%.4fs (%.2f ms/call)\n" N t_serial (1000*t_serial/N) t_v2_serial (1000*t_v2_serial/N) t_v2_threaded (1000*t_v2_threaded/N)
lp(">>> speedup v2-serial(syrk) vs serial(gemm): ", round(t_serial / t_v2_serial, digits = 3), "x")
lp(">>> speedup v2-threaded(bins) vs serial(gemm): ", round(t_serial / t_v2_threaded, digits = 3), "x")
lp(">>> speedup v2-threaded(bins) vs v2-serial(syrk) (isolates JUST the threading, not the syrk swap): ", round(t_v2_serial / t_v2_threaded, digits = 3), "x")

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
