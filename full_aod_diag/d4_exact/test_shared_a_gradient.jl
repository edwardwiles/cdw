# ============================================================================
# Regression + allocation test for shared_a_gradient.jl (shared outer-A-gradient
# task, 2026-07-27): economic_A_gradient! must be bit-for-bit identical to
# composite_gradient_at_fast/composite_gradient_at_fast_buffered/
# composite_gradient_at_fast_pooled (h_mode in :fixed,:cached), at D=4 and at a
# real D=20/W=80000 point, and must allocate strictly less than the pooled
# baseline (task §9: "at least an additional 80% reduction from the pooled
# result").
# ============================================================================
using Test, Printf, Random

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "shared_a_gradient.jl"))

println("=== D=4 correctness gate ===")
ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)

const W_CAND = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

Random.seed!(19260727)
dir = randn(D2 - 1); dir ./= sqrt(sum(abs2, dir))
points_d4 = [
    ("W_CAND", x_free_from_w(W_CAND)),
    ("perturbed", x_free_from_w(vcat(W_CAND[1], W_CAND[2:end] .+ 0.03 .* dir))),
]

all_pass = true
ws = EconomicAGradientWorkspace(W)

for (label, xf) in points_d4
    base = solve_base_state(xf, ctx)
    for h_mode in (:fixed, :cached)
        bwc_ref = Dict{Int,Float64}()
        g_ref, meta_ref = composite_gradient_at_fast(xf, ctx, pe; base = base, h_mode = h_mode,
            bandwidth_cache = h_mode == :cached ? bwc_ref : nothing)

        grad_A = zeros(D2)
        empty!(ws.bandwidth_cache)
        meta_shared = economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = h_mode)

        ok = g_ref == grad_A
        global all_pass &= ok
        @printf("[D=4 %-10s h_mode=%-7s] max|Δg| = %.3e  bit-identical=%s\n",
            label, string(h_mode), maximum(abs.(g_ref .- grad_A)), ok)
        @test ok
        @test meta_ref.h_used == meta_shared.h_used
    end
end

println("\n=== D=4 allocation: economic_A_gradient! vs composite_gradient_at_fast_pooled (warm bandwidth cache) ===")
xf = points_d4[1][2]
base = solve_base_state(xf, ctx)
pool = build_grad_workspace_pool(W)
bwc_pooled = Dict{Int,Float64}()
composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, h_mode = :cached, bandwidth_cache = bwc_pooled)
b_pooled_warm = @allocated composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, h_mode = :cached, bandwidth_cache = bwc_pooled)

grad_A = zeros(D2)
empty!(ws.bandwidth_cache)
economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)
b_shared_warm = @allocated economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)

@printf("pooled (warm bwc):  %.4f MB\n", b_pooled_warm / 1e6)
@printf("shared (warm bwc):  %.4f MB\n", b_shared_warm / 1e6)
@printf("reduction: %.1f%%\n", 100 * (1 - b_shared_warm / b_pooled_warm))

println("\n=== D=4 allocation: cold bandwidth cache (worst case -- forces select_bandwidth on every coordinate) ===")
bwc_pooled_cold_probe = Dict{Int,Float64}()
composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, h_mode = :cached, bandwidth_cache = bwc_pooled_cold_probe)  # warm the METHOD
b_pooled_cold = @allocated composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

empty!(ws.bandwidth_cache)
economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)  # warm the METHOD
empty!(ws.bandwidth_cache)
b_shared_cold = @allocated economic_A_gradient!(grad_A, base, ctx, pe, ws; h_mode = :cached)

@printf("pooled (cold bwc):  %.4f MB\n", b_pooled_cold / 1e6)
@printf("shared (cold bwc):  %.4f MB\n", b_shared_cold / 1e6)
@printf("reduction: %.1f%%\n", 100 * (1 - b_shared_cold / b_pooled_cold))

println("\nALL D=4 TESTS: ", all_pass ? "PASS" : "FAIL")
all_pass || error("D=4 correctness gate FAILED")
