# 2026-07-25 continuation, task §9: decompose CM's build_bin_tables! cost into its Ttab (CM-CM,
# H_CC ingredient) vs Stab (core-CM cross, H_EC ingredient) halves. Non-blocking side task -- does
# NOT gate the shared H_EE merge decision.
#
# build_bin_tables!'s actual loop body (cm_hessian_architectures.jl) is TWO STRUCTURALLY SEPARATE
# per-draw blocks inside one `for s in 1:W` loop (Stab accumulation, O(D*NCORE) per draw, THEN
# Ttab accumulation, O(D^2) per draw) -- not fused -- so they can be timed independently by
# replaying each block as its own standalone pass, without touching production code.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "core_exact_hessian.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, LinearAlgebra, Statistics

"Stab-only pass (core-CM cross ingredient, H_EC's raw material)."
function stab_only!(cctx::CMBinHessCtx, E::AbstractMatrix{Float64}, w::AbstractVector{Float64})
    D = cctx.D; NCORE = cctx.NCORE; Bidx = cctx.Bidx
    S = cctx.Stab
    fill!(S, 0.0)
    W = size(E, 1)
    @inbounds for s in 1:W
        ws = w[s]
        for x in 1:D
            bx = Bidx[s, x]
            for j in 1:NCORE
                S[x, j, bx] += ws * E[s, j]
            end
        end
    end
    return nothing
end

"Ttab-only pass (CM-CM ingredient, H_CC's raw material)."
function ttab_only!(cctx::CMBinHessCtx, w::AbstractVector{Float64})
    D = cctx.D; Bidx = cctx.Bidx
    T = cctx.Ttab
    fill!(T, 0.0)
    W = length(w)
    @inbounds for s in 1:W
        ws = w[s]
        for x in 1:D
            bx = Bidx[s, x]
            for y in 1:D
                by = Bidx[s, y]
                T[x, y, bx, by] += ws
            end
        end
    end
    return nothing
end

W = 80_000
lp(xs...) = (println(xs...); flush(stdout))
lp("Building D=20 real context: :exclude_row, W=$W, seed=20260719 ..."); flush(stdout)
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_seed = 20260719, destination_sample = :exclude_row)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]

pcx = build_cm_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = false)
base = archC_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
lp("nStatus=", base.inner_status)

obj = pcx.ctx_cm.obj
cctx = pcx.cctx
x = vcat(base.ζstar, base.λstar)
_archC_prep_for_hessian!(obj, x)
@unpack H, arg0, arg2, ddPsi! = obj
ddPsi!(arg2, arg0)
w = arg2
NCORE = cctx.NCORE
E = @view H[:, 2:1+NCORE]
M = obj.M

# warm-up (JIT)
stab_only!(cctx, E, w); ttab_only!(cctx, w); build_bin_tables!(cctx, E, w); prefix_sum_tables!(cctx)

n_rep = 5
t_stab = minimum([@elapsed stab_only!(cctx, E, w) for _ in 1:n_rep])
t_ttab = minimum([@elapsed ttab_only!(cctx, w) for _ in 1:n_rep])
t_combined = minimum([@elapsed build_bin_tables!(cctx, E, w) for _ in 1:n_rep])
t_prefix = minimum([@elapsed prefix_sum_tables!(cctx) for _ in 1:n_rep])

# full Hessian callback (dense-reference core, for an apples-to-apples "whole callback" denominator)
cctx.core_hessian_backend = :dense_reference
h = Vector{Float64}(undef, (cctx.NCORE + cctx.ncm) * (cctx.NCORE + cctx.ncm + 1) ÷ 2)
hessian_cm_structured!(h, obj, cctx)   # warm-up
t_full_callback = minimum([@elapsed hessian_cm_structured!(h, obj, cctx) for _ in 1:n_rep])

b_stab = @allocated stab_only!(cctx, E, w)
b_ttab = @allocated ttab_only!(cctx, w)

lp("="^90)
@printf("Stab-only (core-CM cross ingredient, O(W*D*NCORE)): %.4fs, %.2f MB\n", t_stab, b_stab/1e6)
@printf("Ttab-only (CM-CM ingredient, O(W*D^2)):              %.4fs, %.2f MB\n", t_ttab, b_ttab/1e6)
@printf("Combined build_bin_tables! (production, both):      %.4fs\n", t_combined)
@printf("prefix_sum_tables!:                                  %.4fs\n", t_prefix)
@printf("Full Hessian callback (dense-reference core):        %.4fs\n", t_full_callback)
@printf("\nShare of Stab within combined build_bin_tables!: %.1f%%\n", 100*t_stab/t_combined)
@printf("Share of Ttab within combined build_bin_tables!: %.1f%%\n", 100*t_ttab/t_combined)
@printf("Share of Stab within FULL Hessian callback: %.1f%%\n", 100*t_stab/t_full_callback)
@printf("Share of combined bin-table build within FULL Hessian callback: %.1f%%\n", 100*t_combined/t_full_callback)
@printf("D=%d, NCORE=%d, L=%d, W=%d -- D*NCORE=%d vs D^2=%d (per-draw op-count ratio Stab:Ttab = %.1f:1)\n",
    cctx.D, NCORE, cctx.L, W, cctx.D*NCORE, cctx.D^2, (cctx.D*NCORE)/(cctx.D^2))
