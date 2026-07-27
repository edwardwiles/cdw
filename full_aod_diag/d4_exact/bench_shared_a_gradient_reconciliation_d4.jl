# ============================================================================
# Shared outer-A-gradient task (2026-07-27), Section 2: source-level allocation
# reconciliation of composite_gradient_at_fast_pooled's residual bytes.
#
# D=4/W=8000 first (cheap, iterate fast); a D=20/W=80000 companion script
# (bench_shared_a_gradient_reconciliation_d20.jl) repeats the same breakdown
# at production scale to reconcile the codebase's own cited 756.5 MB/call
# figure (c10_d20_production_driver.jl:103).
# ============================================================================
using Printf, Random

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
println("D=$D, D2=$D2, W=$W")

const W_CAND = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf = x_free_from_w(W_CAND)

base = solve_base_state(xf, ctx)

function bytes_of(f, args...; kwargs...)
    f(args...; kwargs...)  # warm
    return @allocated f(args...; kwargs...)
end

println("\n--- Component-level breakdown (warmed @allocated) ---")

b_cache = bytes_of(build_lfix_base_cache, xf, ctx, base; validate_dense = false)
@printf("build_lfix_base_cache (fresh, once per gradient call):       %10.4f MB\n", b_cache / 1e6)

cache = build_lfix_base_cache(xf, ctx, base; validate_dense = false)
z0 = log.(reshape(xf[2:end], D, D))
w0 = vcat(xf[1], pivot_reduce(z0, pe))

# select_bandwidth on a COLD cache (one representative single-changed-origin coordinate
# and, separately, a coordinate that hits the 2-changed-origin top3 tier if one exists
# in this D=4 point's pivot structure).
cells_by_k = Dict(k => affected_cells(pe, k) for k in 2:D2)
dests_by_k = Dict(k => unique(last.(v)) for (k, v) in cells_by_k)
n_changed_by_k = Dict(k => length(v) for (k, v) in cells_by_k)
println("coord -> #affected (o,d) cells: ", sort(collect(n_changed_by_k)))

k1 = 2
b_bw_single = bytes_of(select_bandwidth, cache, ctx, pe, w0, k1)
@printf("select_bandwidth (single coordinate, cold, k=%d):             %10.4f MB\n", k1, b_bw_single / 1e6)

b_bw_all = 0.0
GC.gc()
b_bw_all = @allocated for k in 2:D2
    select_bandwidth(cache, ctx, pe, w0, k)
end
@printf("select_bandwidth summed over ALL %d coordinates (cold cache): %10.4f MB\n", D2 - 1, b_bw_all / 1e6)

# a_block_fd_component (allocating, original) vs the ws!-based pooled variant, at a fixed h.
h_fixed = 0.01
b_fd_orig_one = bytes_of(a_block_fd_component, cache, ctx, pe, w0, k1, h_fixed)
@printf("a_block_fd_component (ORIGINAL, allocating), 1 coordinate:    %10.4f MB\n", b_fd_orig_one / 1e6)

pool = build_grad_workspace_pool(W)
ws = pool.slots[1]
b_fd_ws_one = bytes_of(a_block_fd_component_ws!, ws, cache, ctx, pe, w0, k1, h_fixed)
@printf("a_block_fd_component_ws! (pooled), 1 coordinate:              %10.4f MB\n", b_fd_ws_one / 1e6)

println("\n--- Full-gradient totals, h_mode=:cached, COLD bandwidth_cache every call (worst case: forces select_bandwidth on every coordinate every call) ---")
# Warm the FUNCTION itself first (compile do_coord!/select_bandwidth/etc. for these argument
# types) with a throwaway Dict -- @allocated on a never-yet-called method includes one-time JIT
# compilation allocation, which would otherwise be misattributed as "gradient allocation."
composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
b_pooled_cold = @allocated begin
    bwc = Dict{Int,Float64}()
    composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc)
end
@printf("composite_gradient_at_fast_pooled, COLD bandwidth_cache (function pre-warmed, fresh Dict): %10.4f MB\n", b_pooled_cold / 1e6)

println("\n--- Full-gradient totals, h_mode=:cached, WARM bandwidth_cache reused across calls (steady-state production pattern) ---")
bwc_warm = Dict{Int,Float64}()
composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_warm)  # populate
b_pooled_warm = @allocated composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_warm)
@printf("composite_gradient_at_fast_pooled, WARM bandwidth_cache:      %10.4f MB\n", b_pooled_warm / 1e6)

println("\n--- Reconciliation: cold-call total should approx = build_lfix_base_cache + sum(select_bandwidth) + sum(a_block_fd_component_ws!) ---")
b_fd_ws_all = @allocated for k in 2:D2
    a_block_fd_component_ws!(ws, cache, ctx, pe, w0, k, h_fixed)
end
recon_total = b_cache + b_bw_all + b_fd_ws_all
@printf("build_lfix_base_cache:        %10.4f MB\n", b_cache/1e6)
@printf("sum(select_bandwidth), all k: %10.4f MB\n", b_bw_all/1e6)
@printf("sum(a_block_fd_component_ws!), all k: %10.4f MB\n", b_fd_ws_all/1e6)
@printf("RECONCILED SUM:               %10.4f MB\n", recon_total/1e6)
@printf("MEASURED pooled COLD total:   %10.4f MB\n", b_pooled_cold/1e6)
@printf("MEASURED pooled WARM total:   %10.4f MB\n", b_pooled_warm/1e6)

println("\ndone.")

println("\n--- Per-coordinate a_block_fd_component_ws! breakdown (diagnosing the 1.99MB/15-coord total) ---")
for k in 2:D2
    a_block_fd_component_ws!(ws, cache, ctx, pe, w0, k, h_fixed)  # warm this k specifically
    b = @allocated a_block_fd_component_ws!(ws, cache, ctx, pe, w0, k, h_fixed)
    cells = affected_cells(pe, k)
    @printf("k=%2d  cells=%-20s bytes=%8d (%.4f MB)\n", k, string(cells), b, b/1e6)
end
