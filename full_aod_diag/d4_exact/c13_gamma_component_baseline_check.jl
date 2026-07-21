# Isolates whether the gamma'_focal analytic-vs-FD gap seen in c13_validate_cm_aware_lfix.jl
# is specific to the CM-augmented path, or a pre-existing property of gamma_component_analytic
# vs a fully-optimized-value FD of the PLAIN (non-CM) delta_star -- run against ctx/obj alone,
# no CM code touched at all.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Printf, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]

function delta_star_plain(x_free)
    r = evaluate_fullA(x_free, ctx; use_cache = false, warm = false)
    r.inner_status in (0, -100, -101, -103) || error("inner solve failed, nStatus=$(r.inner_status)")
    return -r.zeta
end

D = ctx.D; D2 = D^2
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0 = vcat(x_free_calib[1], pivot_reduce(z0, pe))

base = solve_base_state(x_free_calib, ctx)
g_analytic, _ = composite_gradient_at_fast(x_free_calib, ctx, pe; base = base)

for hfd in (0.02, 0.01, 0.005, 0.0025, 0.001)
    wp = copy(w0); wp[1] += hfd
    wm = copy(w0); wm[1] -= hfd
    xfp = vcat(wp[1], vec(exp.(z0)))
    xfm = vcat(wm[1], vec(exp.(z0)))
    g_fd_gamma = (delta_star_plain(xfp) - delta_star_plain(xfm)) / (2hfd)
    @printf "PLAIN (non-CM)  h=%.4f  gamma analytic=%.6e  FD=%.6e  ratio=%.4f\n" hfd g_analytic[1] g_fd_gamma g_analytic[1]/g_fd_gamma
end

println()
println("="^80)
println("Now the CM-augmented path, same shrinking-h sweep on gamma alone")
println("="^80)
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))

aug = build_cm_augmented_obj(ctx, CS; L = 10, contrasts = :anchored)
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
bins = cm_bin_indices_for(ctx, aug)
function delta_star_cm2(x_free)
    r = evaluate_fullA(x_free, ctx_cm; use_cache = false, warm = false)
    r.inner_status in (0, -100, -101, -103) || error("inner solve failed, nStatus=$(r.inner_status)")
    return -r.zeta
end
base_cm = solve_base_state(x_free_calib, ctx_cm)
cache_cm = build_lfix_base_cache_cm(x_free_calib, ctx_cm, base_cm, ctx, aug, bins)
g_analytic_cm, _ = composite_gradient_at_fast_cm(x_free_calib, ctx_cm, pe, ctx, aug, bins; base = base_cm, cache = cache_cm)
for hfd in (0.02, 0.01, 0.005, 0.0025, 0.001)
    wp = copy(w0); wp[1] += hfd
    wm = copy(w0); wm[1] -= hfd
    xfp = vcat(wp[1], vec(exp.(z0)))
    xfm = vcat(wm[1], vec(exp.(z0)))
    g_fd_gamma = (delta_star_cm2(xfp) - delta_star_cm2(xfm)) / (2hfd)
    @printf "CM-augmented    h=%.4f  gamma analytic=%.6e  FD=%.6e  ratio=%.4f\n" hfd g_analytic_cm[1] g_fd_gamma g_analytic_cm[1]/g_fd_gamma
end
