# ============================================================================
# D=20/W=80000 real-data correctness + allocation gate for shared_a_gradient.jl
# (shared outer-A-gradient task, 2026-07-27).
# ============================================================================
using Test, Printf, Random

include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "bandwidth_quantile.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "shared_a_gradient.jl"))

println("Building real D=20 context (W=80000, delta=1.0)...")
flush(stdout)
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D; D2 = D * Ddest; W = ctx.W
println("D=$D, Ddest=$Ddest, D2=$D2, W=$W")
flush(stdout)

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe)
gp0 = ctx.θ0_up[3+D]
xf_calib = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

println("Solving base state at calibration point...")
flush(stdout)
base = solve_base_state(xf_calib, ctx)
println("Base state solved, nStatus=", base.inner_status)
flush(stdout)

ws = EconomicAGradientWorkspace(W)

println("\n=== D=20 correctness: economic_A_gradient! vs composite_gradient_at_fast (reference), h_mode=:cached ===")
flush(stdout)
bwc_ref = Dict{Int,Float64}()
@time g_ref, meta_ref = composite_gradient_at_fast(xf_calib, ctx, pe; base = base, h_mode = :cached, bandwidth_cache = bwc_ref)
flush(stdout)

grad_A = zeros(D2)
@time meta_shared = economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)
flush(stdout)

maxdiff = maximum(abs.(g_ref .- grad_A))
@printf("max|Δg| = %.3e   bit-identical=%s\n", maxdiff, g_ref == grad_A)
@test g_ref == grad_A
@test meta_ref.h_used == meta_shared.h_used
flush(stdout)

println("\n=== D=20 allocation: composite_gradient_at_fast_pooled at the CURRENT real-D20 production default (destination_sample=:exclude_row, D=20/Ddest=19) ===")
flush(stdout)
pool = build_grad_workspace_pool(W)
pooled_crashes = false
try
    composite_gradient_at_fast_pooled(xf_calib, ctx, pe, pool; base = base, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
catch e
    global pooled_crashes = true
    println("composite_gradient_at_fast_pooled CRASHED at the real D=20 production default: ", sprint(showerror, e)[1:min(200,end)])
end
flush(stdout)

println("\n=== D=20 allocation: composite_gradient_at_fast_buffered at the CURRENT real-D20 production default ===")
buffered_crashes = false
try
    composite_gradient_at_fast_buffered(xf_calib, ctx, pe; base = base, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
catch e
    global buffered_crashes = true
    println("composite_gradient_at_fast_buffered CRASHED at the real D=20 production default: ", sprint(showerror, e)[1:min(200,end)])
end
flush(stdout)

@printf("PRE_EXISTING_BUG_CONFIRMED: composite_gradient_at_fast_pooled crashes under :exclude_row = %s\n", pooled_crashes)
@printf("PRE_EXISTING_BUG_CONFIRMED: composite_gradient_at_fast_buffered crashes under :exclude_row = %s\n", buffered_crashes)
flush(stdout)

println("\n=== D=20 allocation: UNBUFFERED (composite_gradient_at_fast, the reference) vs SHARED (economic_A_gradient!), COLD bandwidth cache ===")
flush(stdout)
composite_gradient_at_fast(xf_calib, ctx, pe; base = base, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())  # warm
b_unbuffered_cold = @allocated composite_gradient_at_fast(xf_calib, ctx, pe; base = base, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
@printf("unbuffered COLD: %.2f MB\n", b_unbuffered_cold/1e6)
flush(stdout)

empty!(ws.bandwidth_cache)
economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)  # warm
empty!(ws.bandwidth_cache)
b_shared_cold = @allocated economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)
@printf("shared COLD: %.2f MB\n", b_shared_cold/1e6)
@printf("reduction vs unbuffered (cold): %.1f%%\n", 100*(1 - b_shared_cold/b_unbuffered_cold))
flush(stdout)

println("\n=== D=20 allocation: UNBUFFERED vs SHARED, WARM bandwidth cache (steady state) ===")
flush(stdout)
bwc_unbuf_warm = Dict{Int,Float64}()
composite_gradient_at_fast(xf_calib, ctx, pe; base = base, h_mode = :cached, bandwidth_cache = bwc_unbuf_warm)
b_unbuffered_warm = @allocated composite_gradient_at_fast(xf_calib, ctx, pe; base = base, h_mode = :cached, bandwidth_cache = bwc_unbuf_warm)
@printf("unbuffered WARM: %.2f MB\n", b_unbuffered_warm/1e6)
flush(stdout)

empty!(ws.bandwidth_cache)
economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)
b_shared_warm = @allocated economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)
@printf("shared WARM: %.2f MB\n", b_shared_warm/1e6)
@printf("reduction vs unbuffered (warm): %.1f%%\n", 100*(1 - b_shared_warm/b_unbuffered_warm))
flush(stdout)

println("\nDONE.")
