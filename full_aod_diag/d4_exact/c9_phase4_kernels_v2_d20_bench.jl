# ============================================================================
# Continuation 9, Phase 4: D=20/W=80,000 timing of the destination-major _v2
# kernels (compressed_cc_kernels_v2.jl) vs the originals -- the primitives
# EVERY compressed FG/HVP call bottoms out in. Already verified bit-identical
# at D=4 (test_compressed_cc_kernels_v2.jl). Sanity check first (context
# build is already known-safe from Phase 3), then N-rep timing.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_cc_kernels_v2.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase4_kernels_v2_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase4_kernels_v2_d20_bench.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
D = ctx.D; W = size(ctx.U, 1)
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)
base = solve_base_state(xf_nat, ctx)
cf = build_compressed_factual(θ_full, ctx)
ncol = ctx.obj.outer_constr_index - 1
logprint("D=", D, "  ncol=", ncol, "  W=", W)

# ---- correctness cross-check IN THIS PROCESS too (belt and suspenders) ----
Random.seed!(999)
β = randn(ncol); weights = randn(W)
d1 = maximum(abs.(compressed_dual_contraction(β, cf) .- compressed_dual_contraction_v2(β, cf)))
d2 = maximum(abs.(compressed_transpose_contraction(weights, cf) .- compressed_transpose_contraction_v2(weights, cf)))
logprint("D=20 correctness cross-check: dual_contraction maxdiff=", d1, "  transpose_contraction maxdiff=", d2,
         "  ", (d1 == 0.0 && d2 == 0.0) ? "BIT-IDENTICAL" : "DIFFERS (investigate before trusting timing)")

ζ = base.ζstar; λ = collect(base.λstar)

# ---- warm-up (untimed): pay JIT for both original and v2 paths ----
for _ in 1:3
    compressed_dual_contraction(β, cf); compressed_dual_contraction_v2(β, cf)
    compressed_transpose_contraction(weights, cf); compressed_transpose_contraction_v2(weights, cf)
    compressed_cc_value_grad(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
    compressed_cc_value_grad_v2(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
end

N = 30
function time_reps(f, N)
    times = Float64[]
    for _ in 1:N
        push!(times, @elapsed f())
    end
    return median(times), times
end

logprint("\n---- compressed_dual_contraction: original vs v2, N=", N, " ----")
med_orig1, _ = time_reps(() -> compressed_dual_contraction(β, cf), N)
med_v2_1, _ = time_reps(() -> compressed_dual_contraction_v2(β, cf), N)
logprint(@sprintf("  original: median=%.4fms   v2: median=%.4fms   speedup=%.3fx",
          med_orig1 * 1000, med_v2_1 * 1000, med_orig1 / med_v2_1))

logprint("\n---- compressed_transpose_contraction: original vs v2, N=", N, " ----")
med_orig2, _ = time_reps(() -> compressed_transpose_contraction(weights, cf), N)
med_v2_2, _ = time_reps(() -> compressed_transpose_contraction_v2(weights, cf), N)
logprint(@sprintf("  original: median=%.4fms   v2: median=%.4fms   speedup=%.3fx",
          med_orig2 * 1000, med_v2_2 * 1000, med_orig2 / med_v2_2))

logprint("\n---- compressed_cc_value_grad: original vs v2, N=", N, " ----")
med_orig3, _ = time_reps(() -> compressed_cc_value_grad(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!), N)
med_v2_3, _ = time_reps(() -> compressed_cc_value_grad_v2(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!), N)
logprint(@sprintf("  original: median=%.4fms   v2: median=%.4fms   speedup=%.3fx",
          med_orig3 * 1000, med_v2_3 * 1000, med_orig3 / med_v2_3))

logprint("\n---- compressed_cc_hvp: original vs v2, N=", N, " ----")
_, _, _, q0, _ = compressed_cc_value_grad(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
p = randn(1 + ncol); p ./= norm(p)
med_orig4, _ = time_reps(() -> compressed_cc_hvp(q0, p[1], p[2:end], cf; ddPsi! = ctx.obj.ddPsi!), N)
med_v2_4, _ = time_reps(() -> compressed_cc_hvp_v2(q0, p[1], p[2:end], cf; ddPsi! = ctx.obj.ddPsi!), N)
logprint(@sprintf("  original: median=%.4fms   v2: median=%.4fms   speedup=%.3fx",
          med_orig4 * 1000, med_v2_4 * 1000, med_orig4 / med_v2_4))

write_csv_rows(joinpath(OUTDIR, "kernels_v2_speedup.csv"),
    [(kernel = "dual_contraction", orig_ms = med_orig1*1000, v2_ms = med_v2_1*1000, speedup = med_orig1/med_v2_1),
     (kernel = "transpose_contraction", orig_ms = med_orig2*1000, v2_ms = med_v2_2*1000, speedup = med_orig2/med_v2_2),
     (kernel = "cc_value_grad", orig_ms = med_orig3*1000, v2_ms = med_v2_3*1000, speedup = med_orig3/med_v2_3),
     (kernel = "cc_hvp", orig_ms = med_orig4*1000, v2_ms = med_v2_4*1000, speedup = med_orig4/med_v2_4)])

logprint("\nVmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase4_kernels_v2_d20_bench.jl COMPLETE at ", now())
close(LOGIO)
