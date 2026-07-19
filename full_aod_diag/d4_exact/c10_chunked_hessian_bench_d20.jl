# ============================================================================
# Continuation 10, Part 1: D=20/W=80,000 real-data benchmark of the chunked-
# BLAS Hessian (chunked_hessian.jl) vs the existing dense baseline
# (oracle_fast.jl's inner_loop_internal_profiled, hessopt=exact + full
# H_copy materialize-then-gemm). Chunk sizes 5000/10000/20000/40000/80000.
#
# Per chunk size, measures (at the SAME calibration point, SAME W=80,000
# draws, SAME θ_full -- everything held fixed except the Hessian-forming step):
#   - Hessian-callback-ONLY wall time (isolated via a direct call, not through
#     KNITRO -- avoids conflating callback cost with KNITRO's own SQP overhead)
#   - Cold complete inner-solve time (fresh obj.x = NaN, no warm start)
#   - Warm complete inner-solve time (re-solve from the just-converged point)
#   - n_fg_calls / n_hess_calls (should be IDENTICAL to baseline; a difference
#     would indicate the chunked Hessian is producing a different Hessian that
#     changes KNITRO's own iteration path -- flagged as a bug if seen)
#   - Correctness: Hessian bit-identical (or ~1e-10) vs baseline, dual solution
#     max|dx|, Delta_dual match
#   - Allocations (@allocated) for the isolated Hessian-callback-only call
#   - Process VmHWM (whole-process; this is a shared machine, isolates only
#     what THIS process does, not other users' jobs)
#
# Followed by ONE W=800,000 microbenchmark (a single chunk size chosen from
# the W=80,000 sweep's best/most-competitive result, plus the baseline for
# comparison) -- per the standing memory-safety discipline, preceded by an
# analytic + Base.summarysize probe of the chunk buffer's actual footprint
# BEFORE running anything at that scale.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "chunked_hessian.jl"))
using Statistics, Printf, Dates, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c10_chunked_hessian_bench_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c10_chunked_hessian_bench_d20.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end
vmhwm_gb() = vmhwm_kb() / 1e6

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
D = ctx.D; obj = ctx.obj
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_gb(), digits = 2), " GB")

xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)
n_inner = obj.outer_constr_index
ncol = n_inner - 1
W = size(obj.U, 1)
logprint("D=", D, "  W=", W, "  n_inner (incl zeta)=", n_inner, "  ncol (excl zeta)=", ncol)

# ---- reference: baseline dense-Hessian solve ----
logprint("\n---- REFERENCE: dense baseline (oracle_fast.jl inner_loop_internal_profiled) ----")
obj.x .= NaN
t0 = time()
K_ref, x_ref, status_ref, nfg_ref, nhess_ref = inner_loop_internal_profiled(obj, θ_full)
t_cold_ref = time() - t0
status_ref in (0, -100, -101, -103) || error("reference cold solve failed, status=$status_ref")
t0 = time()
K_refw, x_refw, status_refw, nfg_refw, nhess_refw = inner_loop_internal_profiled(obj, θ_full)
t_warm_ref = time() - t0
logprint(@sprintf("  cold: status=%d n_fg=%d n_hess=%d wall=%.3fs K=%.10f", status_ref, nfg_ref, nhess_ref, t_cold_ref, K_ref))
logprint(@sprintf("  warm: status=%d n_fg=%d n_hess=%d wall=%.3fs K=%.10f", status_refw, nfg_refw, nhess_refw, t_warm_ref, K_refw))
logprint("  VmHWM after reference solves = ", round(vmhwm_gb(), digits = 2), " GB")

# ---- isolated Hessian-callback-only timing: baseline ----
_prep_for_hessian!(obj, x_ref)
n = obj.outer_constr_index
ntri = div(n * (n + 1), 2)
h_ref = zeros(ntri)
CS.hessian!(h_ref, obj)   # warm up / compile
N_HESS_REPS = 10
hess_times_ref = Float64[]
allocs_ref = Int[]
for _ in 1:N_HESS_REPS
    _prep_for_hessian!(obj, x_ref)
    a = @allocated CS.hessian!(h_ref, obj)
    t0 = time_ns()
    CS.hessian!(h_ref, obj)
    push!(hess_times_ref, (time_ns() - t0) / 1e9)
    push!(allocs_ref, a)
end
med_hess_ref = median(hess_times_ref)
logprint(@sprintf("  isolated hessian! (baseline): median=%.5fs  alloc(1 call)=%.2f MB", med_hess_ref, allocs_ref[end] / 1e6))

# ---- chunk-size sweep ----
chunk_sizes = [5000, 10000, 20000, 40000, 80000]
rows = NamedTuple[]

for cs in chunk_sizes
    logprint("\n---- chunk_size=", cs, " ----")

    # correctness: Hessian formula, same x_ref
    _prep_for_hessian!(obj, x_ref)
    h_c = zeros(ntri)
    hessian_chunked!(h_c, obj, cs)
    maxabsdiff = maximum(abs.(h_c .- h_ref))
    maxreldiff = maximum(abs.(h_c .- h_ref) ./ max.(abs.(h_c), abs.(h_ref), 1.0))
    bit_identical = h_c == h_ref
    logprint(@sprintf("  Hessian correctness: bit_identical=%s max|diff|=%.3e max_rel_diff=%.3e", bit_identical, maxabsdiff, maxreldiff))

    # isolated Hessian-callback-only timing
    _prep_for_hessian!(obj, x_ref)
    hessian_chunked!(h_c, obj, cs)   # warm up
    hess_times_c = Float64[]
    allocs_c = Int[]
    zbuf = Matrix{Float64}(undef, min(cs, W), n)
    for _ in 1:N_HESS_REPS
        _prep_for_hessian!(obj, x_ref)
        a = @allocated hessian_chunked!(h_c, obj, cs; zbuf = zbuf)
        t0 = time_ns()
        hessian_chunked!(h_c, obj, cs; zbuf = zbuf)
        push!(hess_times_c, (time_ns() - t0) / 1e9)
        push!(allocs_c, a)
    end
    med_hess_c = median(hess_times_c)
    logprint(@sprintf("  isolated hessian_chunked!: median=%.5fs  speedup_vs_dense=%.3fx  alloc(1 call, zbuf reused)=%.2f MB",
              med_hess_c, med_hess_ref / med_hess_c, allocs_c[end] / 1e6))

    # complete cold/warm inner solve
    obj.x .= NaN
    t0 = time()
    K_c, x_c, status_c, nfg_c, nhess_c = inner_loop_internal_chunked(obj, θ_full, cs)
    t_cold_c = time() - t0
    t0 = time()
    K_cw, x_cw, status_cw, nfg_cw, nhess_cw = inner_loop_internal_chunked(obj, θ_full, cs)
    t_warm_c = time() - t0

    dK = abs(K_c - K_ref); dx = maximum(abs.(x_c .- x_ref))
    logprint(@sprintf("  cold: status=%d(ref %d) n_fg=%d(ref %d) n_hess=%d(ref %d) wall=%.3fs(ref %.3fs) speedup=%.3fx |dK|=%.3e max|dx|=%.3e",
              status_c, status_ref, nfg_c, nfg_ref, nhess_c, nhess_ref, t_cold_c, t_cold_ref, t_cold_ref / t_cold_c, dK, dx))
    logprint(@sprintf("  warm: status=%d(ref %d) n_fg=%d(ref %d) n_hess=%d(ref %d) wall=%.3fs(ref %.3fs) speedup=%.3fx",
              status_cw, status_refw, nfg_cw, nfg_refw, nhess_cw, nhess_refw, t_warm_c, t_warm_ref, t_warm_ref / t_warm_c))
    logprint("  VmHWM = ", round(vmhwm_gb(), digits = 2), " GB")

    push!(rows, (chunk_size = cs, bit_identical = bit_identical, max_abs_diff = maxabsdiff, max_rel_diff = maxreldiff,
        hess_only_median_s = med_hess_c, hess_only_speedup = med_hess_ref / med_hess_c, hess_only_alloc_bytes = allocs_c[end],
        cold_wall_s = t_cold_c, cold_speedup = t_cold_ref / t_cold_c, warm_wall_s = t_warm_c, warm_speedup = t_warm_ref / t_warm_c,
        status_cold = status_c, n_fg_cold = nfg_c, n_hess_cold = nhess_c, dK = dK, dx = dx, vmhwm_gb = vmhwm_gb()))
end

logprint("\n", "="^100)
logprint("SUMMARY (W=80,000, reference dense: cold=", round(t_cold_ref, digits = 3), "s warm=", round(t_warm_ref, digits = 3),
          "s hess_only=", round(med_hess_ref, digits = 5), "s)")
logprint(@sprintf("  %10s %8s %10s | %10s %8s | %10s %8s | %10s", "chunk", "bit_id", "hess_s", "hess_spd", "cold_s", "cold_spd", "warm_s", "warm_spd"))
for r in rows
    @printf("  %10d %8s %10.5f | %10.3fx %8.3f | %10.3fx %8.3f | %10.3fx\n",
            r.chunk_size, r.bit_identical, r.hess_only_median_s, r.hess_only_speedup, r.cold_wall_s, r.cold_speedup, r.warm_wall_s, r.warm_speedup)
end

write_csv_rows(joinpath(OUTDIR, "chunked_hessian_w80k_summary.csv"),
    vcat([(chunk_size = 0, bit_identical = true, max_abs_diff = 0.0, max_rel_diff = 0.0,
           hess_only_median_s = med_hess_ref, hess_only_speedup = 1.0, hess_only_alloc_bytes = allocs_ref[end],
           cold_wall_s = t_cold_ref, cold_speedup = 1.0, warm_wall_s = t_warm_ref, warm_speedup = 1.0,
           status_cold = status_ref, n_fg_cold = nfg_ref, n_hess_cold = nhess_ref, dK = 0.0, dx = 0.0, vmhwm_gb = vmhwm_gb())],
          rows))

logprint("\nVmHWM at end of W=80k sweep = ", round(vmhwm_gb(), digits = 2), " GB")
logprint("\nc10_chunked_hessian_bench_d20.jl (W=80k portion) COMPLETE at ", now())
close(LOGIO)
