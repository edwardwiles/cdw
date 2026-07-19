# ============================================================================
# Continuation 10, Part 1: W=800,000 memory-safety probe + single benchmark run.
#
# Per this investigation's standing memory-safety discipline (a REAL
# server-wide memory incident hit at this exact W=800,000/D=20 scale earlier
# in this investigation, docs/fullA_continuation9_handoff.md's "A live
# server-wide memory incident" -- a mis-defaulted flag scaled a diagnostic
# tensor to 780GB+ at this W before being killed): compute the EXPECTED byte
# footprint of the chunk buffer analytically AND cross-check with
# Base.summarysize on a REAL allocated chunk buffer BEFORE running anything
# at W=800,000. Only after that probe passes does this script proceed to
# d20_real_setup(W=800000) (itself immediately VmHWM-checked) and ONE cold+
# warm solve at a single chosen chunk size (40000 -- a mid-range, good-balance
# choice per the W=80,000 sweep, where every chunk size performed
# similarly), NOT a full chunk-size sweep (per this task's explicit
# instruction to do only one microbenchmark at this scale).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "chunked_hessian.jl"))
using Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c10_chunked_hessian_w800k")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_chunked_hessian_w800k_probe.jl starting at ", now())

function vmhwm_gb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2]) / 1e6
    end
    return -1.0
end

const W_TARGET = 800_000
const NCOL = 402          # outer_constr_index at D=20 (confirmed from the W=80k runs: n_inner=402)
const CHUNK_SIZE = 40_000
const SAFETY_CEILING_GB = 25.0   # matches this investigation's established self-imposed kill threshold (see docs/fullA_fully_compressed_inner_report.md, "20GB self-imposed kill threshold")

logprint("\n---- Step 0: analytic + empirical memory probe (BEFORE touching W=800,000) ----")
bytes_chunk_analytic = CHUNK_SIZE * NCOL * 8
logprint(@sprintf("  analytic chunk buffer footprint: chunk_size=%d x ncol=%d x 8 bytes = %.2f MB",
          CHUNK_SIZE, NCOL, bytes_chunk_analytic / 1e6))
probe_buf = Matrix{Float64}(undef, CHUNK_SIZE, NCOL)
fill!(probe_buf, 1.0)
bytes_chunk_actual = Base.summarysize(probe_buf)
logprint(@sprintf("  Base.summarysize of a REAL chunk buffer: %.2f MB (matches analytic: %s)",
          bytes_chunk_actual / 1e6, bytes_chunk_actual == bytes_chunk_analytic))
probe_buf = nothing; GC.gc()

bytes_H_analytic = W_TARGET * (NCOL + 2) * 8   # obj.H's own shape (W x (d+2)), for comparison -- UNCHANGED by chunking
logprint(@sprintf("  for comparison, obj.H itself at W=%d: %.2f GB (unchanged by this task's chunking -- Part 2 addresses building this faster, not smaller)",
          W_TARGET, bytes_H_analytic / 1e9))
logprint(@sprintf("  baseline's H_copy (the SECOND full-size buffer chunking ELIMINATES): another %.2f GB avoided",
          bytes_H_analytic / 1e9))

bytes_chunk_actual < 1e9 || error("SAFETY ABORT: chunk buffer unexpectedly large ($(bytes_chunk_actual/1e9) GB) -- not proceeding to W=800,000")
logprint("  Probe PASSED: chunk buffer is ", round(bytes_chunk_actual / 1e6, digits = 2), " MB, nowhere near the ", SAFETY_CEILING_GB, " GB safety ceiling. Proceeding.")

logprint("\n---- Step 1: d20_real_setup(W=800,000) ----")
t0 = time()
ctx = d20_real_setup(W = W_TARGET)
t_setup = time() - t0
vm1 = vmhwm_gb()
logprint(@sprintf("  d20_real_setup(W=%d) wall=%.2fs  VmHWM=%.2f GB", W_TARGET, t_setup, vm1))
vm1 < SAFETY_CEILING_GB || error("SAFETY ABORT: VmHWM after setup ($vm1 GB) exceeds ceiling ($SAFETY_CEILING_GB GB) -- aborting before any solve")

obj = ctx.obj
D = ctx.D; W = size(obj.U, 1)
xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)
logprint("  D=", D, " W=", W, " n_inner=", obj.outer_constr_index)

logprint("\n---- Step 2: ONE baseline (dense) cold+warm solve ----")
obj.x .= NaN
t0 = time()
K_ref, x_ref, status_ref, nfg_ref, nhess_ref = inner_loop_internal_profiled(obj, θ_full)
t_cold_ref = time() - t0
vm2 = vmhwm_gb()
logprint(@sprintf("  baseline cold: status=%d n_fg=%d n_hess=%d wall=%.3fs VmHWM=%.2f GB K=%.10f",
          status_ref, nfg_ref, nhess_ref, t_cold_ref, vm2, K_ref))
status_ref in (0, -100, -101, -103) || error("baseline cold solve failed at W=800000, status=$status_ref")
vm2 < SAFETY_CEILING_GB || error("SAFETY ABORT: VmHWM after baseline cold solve ($vm2 GB) exceeds ceiling")

t0 = time()
K_refw, x_refw, status_refw, nfg_refw, nhess_refw = inner_loop_internal_profiled(obj, θ_full)
t_warm_ref = time() - t0
logprint(@sprintf("  baseline warm: status=%d n_fg=%d n_hess=%d wall=%.3fs", status_refw, nfg_refw, nhess_refw, t_warm_ref))

logprint("\n---- Step 3: ONE chunked (chunk_size=", CHUNK_SIZE, ") cold+warm solve ----")
obj.x .= NaN
t0 = time()
K_c, x_c, status_c, nfg_c, nhess_c = inner_loop_internal_chunked(obj, θ_full, CHUNK_SIZE)
t_cold_c = time() - t0
vm3 = vmhwm_gb()
dK = abs(K_c - K_ref); dx = maximum(abs.(x_c .- x_ref))
logprint(@sprintf("  chunked cold: status=%d(ref %d) n_fg=%d(ref %d) n_hess=%d(ref %d) wall=%.3fs(ref %.3fs) speedup=%.3fx VmHWM=%.2f GB |dK|=%.3e max|dx|=%.3e",
          status_c, status_ref, nfg_c, nfg_ref, nhess_c, nhess_ref, t_cold_c, t_cold_ref, t_cold_ref / t_cold_c, vm3, dK, dx))
status_c == status_ref || error("SAFETY/CORRECTNESS: chunked status differs from baseline at W=800000")
vm3 < SAFETY_CEILING_GB || error("SAFETY ABORT: VmHWM after chunked cold solve ($vm3 GB) exceeds ceiling")

t0 = time()
K_cw, x_cw, status_cw, nfg_cw, nhess_cw = inner_loop_internal_chunked(obj, θ_full, CHUNK_SIZE)
t_warm_c = time() - t0
logprint(@sprintf("  chunked warm: status=%d(ref %d) n_fg=%d(ref %d) n_hess=%d(ref %d) wall=%.3fs(ref %.3fs) speedup=%.3fx",
          status_cw, status_refw, nfg_cw, nfg_refw, nhess_cw, nhess_refw, t_warm_c, t_warm_ref, t_warm_ref / t_warm_c))

logprint("\n---- Isolated Hessian-callback-only timing at W=800,000 (baseline vs chunked, both JIT-warm from the solves above) ----")
_prep_for_hessian!(obj, x_ref)
n = obj.outer_constr_index
ntri = div(n * (n + 1), 2)
h_ref = zeros(ntri)
CS.hessian!(h_ref, obj)   # warmup
t_hess_ref = @elapsed CS.hessian!(h_ref, obj)
_prep_for_hessian!(obj, x_ref)
h_c = zeros(ntri)
zbuf = Matrix{Float64}(undef, min(CHUNK_SIZE, W), n)
hessian_chunked!(h_c, obj, CHUNK_SIZE; zbuf = zbuf)   # warmup
t_hess_c = @elapsed hessian_chunked!(h_c, obj, CHUNK_SIZE; zbuf = zbuf)
maxreldiff = maximum(abs.(h_c .- h_ref) ./ max.(abs.(h_c), abs.(h_ref), 1.0))
logprint(@sprintf("  hessian! (baseline): %.5fs   hessian_chunked!(cs=%d): %.5fs   speedup=%.3fx   max_rel_diff=%.3e",
          t_hess_ref, CHUNK_SIZE, t_hess_c, t_hess_ref / t_hess_c, maxreldiff))

logprint("\nFinal VmHWM = ", round(vmhwm_gb(), digits = 2), " GB")
logprint("\nc10_chunked_hessian_w800k_probe.jl COMPLETE at ", now())
close(LOGIO)
