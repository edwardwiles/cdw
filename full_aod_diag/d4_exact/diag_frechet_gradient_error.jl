# Diagnostic: reproduce the cm_frechet_production_gradient error OUTSIDE the
# KNITRO callback wrapper (which swallows the real Julia stacktrace and only
# reports a generic -500 to KNITRO) to see the actual root cause.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
using Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
t0 = time()
const W = 80_000; const L = 50; const DELTA = 1.0
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
lp("[$(round(time()-t0,digits=1))s] ctx built")
pe = build_pivot_elimination(ctx)
cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
lp("[$(round(time()-t0,digits=1))s] fpcx built")

x_free_calib = ctx.θ0_up[ctx.free_idx]
z_star = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
zfree_star = pivot_reduce(z_star, pe)
σ = ctx.σ
κ_star = 1 - x_free_calib[1]^(σ / (σ - 1))
gp_target = (1 - (κ_star + 1e-4))^((σ - 1) / σ)
x_free0 = vcat(gp_target, vec(exp.(pivot_expand(zfree_star, pe))))
lp("[$(round(time()-t0,digits=1))s] solving base state at (gp_target, zfree*)...")
base = cm_frechet_base_state(x_free0, fpcx)
lp("[$(round(time()-t0,digits=1))s] base solved, inner_status=$(base.inner_status)")

lp("[$(round(time()-t0,digits=1))s] calling cm_frechet_production_gradient directly (no KNITRO wrapper)...")
bandwidth_cache = Dict{Int,Float64}()
try
    g, meta = cm_frechet_production_gradient(x_free0, fpcx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
    lp("[$(round(time()-t0,digits=1))s] SUCCESS: gradient length=$(length(g))  norm=$(norm(g))")
    lp("[$(round(time()-t0,digits=1))s] calling a SECOND time (reusing base) to measure warm gradient cost...")
    g2, meta2 = cm_frechet_production_gradient(x_free0, fpcx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
    lp("[$(round(time()-t0,digits=1))s] SUCCESS (2nd call): max|g-g2|=$(maximum(abs.(g.-g2)))")
catch e
    lp("[$(round(time()-t0,digits=1))s] GRADIENT CALL FAILED with:")
    showerror(stdout, e, catch_backtrace())
    println()
end
