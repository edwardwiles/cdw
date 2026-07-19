# ============================================================================
# Continuation 9, Phase 4: "batch by destination" -- benchmarks whether
# PROCESSING the FD-gradient's per-coordinate loop in DESTINATION-SORTED
# order (all coordinates touching the same destination handled consecutively)
# reduces memory traffic vs the natural (pivot-reduced-index) coordinate
# order, at D=20/W=80,000.
#
# SCOPE (deliberately lighter-touch per the brief's own framing "Benchmark...
# Test whether..."): this does NOT build a new batched kernel that shares
# per-destination base-score/winner/top-k work ACROSS coordinates (that would
# require restructuring select_bandwidth/a_block_fd_component's internals,
# out of this task's remaining time budget after the kernels_v2 work above).
# It tests the CHEAPER, still-informative question: does simply REORDERING
# the SAME existing per-coordinate calls (select_bandwidth + a_block_fd_
# component, unchanged, from lfix_incremental.jl/composite_gradient.jl) so
# that same-destination coordinates run consecutively change wall time via
# better cache locality on cache.price0[:,:,d]/pTσ0[:,:,d] (indexed by
# destination) -- a memory-traffic question, answered empirically rather than
# assumed. Serial (threaded=false) only, to isolate memory-locality effects
# from thread-scheduling noise.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase4_dest_batch_order")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase4_dest_batch_order_bench.jl starting at ", now(), "  commit=", COMMIT)

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
D = ctx.D; D2 = D^2
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

pe = build_pivot_elimination(ctx)
xf0 = ctx.θ0_up[ctx.free_idx]
base = solve_base_state(xf0, ctx)
cache = build_lfix_base_cache(xf0, ctx, base)
z0 = log.(reshape(xf0[2:end], D, D)); w0 = vcat(xf0[1], pivot_reduce(z0, pe))

# ---- coordinate orderings ----
natural_order = collect(2:D2)   # coord 1 = gamma, handled separately (analytic, O(1)) in the real gradient
dest_of(k) = last(first(affected_cells(pe, k)))   # primary destination touched by A-block coordinate k
dest_sorted_order = sort(natural_order; by = dest_of)

logprint("D=", D, "  D^2=", D2, "  n_A_block_coords=", length(natural_order))
logprint("first 10 natural-order destinations: ", [dest_of(k) for k in natural_order[1:10]])
logprint("first 10 dest-sorted-order destinations: ", [dest_of(k) for k in dest_sorted_order[1:10]])

"Serial (unthreaded) full A-block sweep using the EXISTING select_bandwidth/a_block_fd_component (unchanged), in a caller-specified coordinate ORDER only."
function run_sweep(order; h_mode::Symbol = :fixed, h0::Float64 = 0.01, multi_method::Symbol = :top3)
    g = zeros(D2)
    for k in order
        if h_mode == :fixed
            h = h0
        else
            h, _, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
        end
        Lp = a_block_fd_component(cache, ctx, pe, w0, k, w0[k] + h; multi_method = multi_method)
        Lm = a_block_fd_component(cache, ctx, pe, w0, k, w0[k] - h; multi_method = multi_method)
        g[k] = (Lp - Lm) / (2h)
    end
    return g
end

function time_min(f, N)
    f()
    ts = Float64[@elapsed f() for _ in 1:N]
    return minimum(ts), sum(ts) / N
end

N = 4
logprint("\n---- h_mode=:fixed, serial, N=", N, " ----")
mn_nat, mu_nat = time_min(() -> run_sweep(natural_order; h_mode = :fixed), N)
mn_dst, mu_dst = time_min(() -> run_sweep(dest_sorted_order; h_mode = :fixed), N)
logprint(@sprintf("  natural order:      min=%.4fs  mean=%.4fs", mn_nat, mu_nat))
logprint(@sprintf("  dest-sorted order:  min=%.4fs  mean=%.4fs", mn_dst, mu_dst))
logprint(@sprintf("  speedup(min, natural/dest-sorted) = %.3fx", mn_nat / mn_dst))

logprint("\n---- h_mode=:adaptive, serial, N=", N, " ----")
mn_nat_a, mu_nat_a = time_min(() -> run_sweep(natural_order; h_mode = :adaptive), N)
mn_dst_a, mu_dst_a = time_min(() -> run_sweep(dest_sorted_order; h_mode = :adaptive), N)
logprint(@sprintf("  natural order:      min=%.4fs  mean=%.4fs", mn_nat_a, mu_nat_a))
logprint(@sprintf("  dest-sorted order:  min=%.4fs  mean=%.4fs", mn_dst_a, mu_dst_a))
logprint(@sprintf("  speedup(min, natural/dest-sorted) = %.3fx", mn_nat_a / mn_dst_a))

# ---- correctness: both orderings must produce the SAME gradient (order must not matter mathematically) ----
g_nat = run_sweep(natural_order; h_mode = :fixed)
g_dst = run_sweep(dest_sorted_order; h_mode = :fixed)
gdiff = maximum(abs.(g_nat .- g_dst))
logprint("\n  gradient order-independence check (h_mode=:fixed): max|g_natural - g_dest_sorted| = ", gdiff)

write_csv_rows(joinpath(OUTDIR, "dest_batch_order_speedup.csv"),
    [(h_mode = "fixed", natural_min_s = mn_nat, dest_sorted_min_s = mn_dst, speedup = mn_nat/mn_dst),
     (h_mode = "adaptive", natural_min_s = mn_nat_a, dest_sorted_min_s = mn_dst_a, speedup = mn_nat_a/mn_dst_a)])

logprint("\nVmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase4_dest_batch_order_bench.jl COMPLETE at ", now())
close(LOGIO)
