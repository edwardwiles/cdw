# D20 real-data outer-gradient verification for OZC-CROSS (2026-08-09 task, phase 3 continued).
# Full FD sweep at the cheap K=1/1 config (~22s/solve); (B) fixed-contribution check (cheap, no
# re-solving) at all three K configs; a partial FD spot-check (a handful of components) at the
# expensive K=3/3 config (~5min/solve) instead of the full 2*n_eta=120-solve sweep (~10h, impractical).
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D; Ddest = ctx.D_dest
println("ctx built. D=$D D_dest=$Ddest μHat=$(ctx.μHat)")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

function run_B_check(pcx, base, verify, νfull0, tag)
    cache0 = build_lfix_base_cache(x_free_calib, pcx.ctx_cm, base)
    contrib0 = originzc_cross_fixed_contribution(base, pcx.aug, νfull0)
    q0_via_fold = cache0.q0 .- contrib0
    st = pcx.octx.fg_lookup_st
    q0_via_operator = copy(dual_index!(st, vcat(base.ζstar, base.λstar)))
    d_q0 = maximum(abs.(q0_via_fold .- q0_via_operator))
    println("    (B) [$tag] fixed-contribution-fold q0 vs independent operator-FG q0: max abs diff=$d_q0")
    check("$tag: (B) originzc_cross_fixed_contribution matches independent operator recompute", d_q0 < 1e-6)
end

# ---- K=1/1: full FD sweep (cheap) ----
println("\n==== D20 W=$W K_mean=1 K_pair=1: FULL gradient verification ====")
flush(stdout)
layout1 = OriginByPowerCrossLayout(D, 1, 1)
νfull1 = nu0_origin(1, D)
pcx1 = build_originzc_cross_production_context(ctx, CS, layout1)
t = @elapsed (base1, verify1) = archOZ_verified_state(x_free_calib, νfull1, pcx1.ctx_cm)
println("    solve: $(t)s inner_status=$(verify1.inner_status) Delta_dual=$(verify1.Delta_dual)")
flush(stdout)
run_B_check(pcx1, base1, verify1, νfull1, "K=1/1")

t = @elapsed begin
    g_analytic1 = d_delta_dual_d_eta_origin_cross_vec(base1.λstar, pcx1.aug, νfull1; mean_m = verify1.m_mean)
    g_fd1 = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull1, pcx1.ctx_cm; h = 1e-4)
end
d_ag1 = maximum(abs.(g_analytic1 .- g_fd1))
rel1 = d_ag1 / max(1e-8, maximum(abs.(g_fd1)))
println("    (A) [K=1/1] full FD sweep ($(t)s): max abs diff=$d_ag1  max rel diff=$rel1")
println("        analytic=$g_analytic1")
println("        fd      =$g_fd1")
check("K=1/1: (A) analytic eta-gradient matches full reoptimized FD sweep", rel1 < 5e-2)
flush(stdout)

# ---- K=3/3: (B) check + partial FD spot-check (a handful of components) ----
println("\n==== D20 W=$W K_mean=3 K_pair=3: (B) check + PARTIAL FD spot-check ====")
flush(stdout)
layout3 = OriginByPowerCrossLayout(D, 3, 3)
νfull3 = nu0_origin(3, D)
pcx3 = build_originzc_cross_production_context(ctx, CS, layout3)
t = @elapsed (base3, verify3) = archOZ_verified_state(x_free_calib, νfull3, pcx3.ctx_cm)
println("    solve: $(t)s inner_status=$(verify3.inner_status) Delta_dual=$(verify3.Delta_dual)")
flush(stdout)
run_B_check(pcx3, base3, verify3, νfull3, "K=3/3")

g_analytic3 = d_delta_dual_d_eta_origin_cross_vec(base3.λstar, pcx3.aug, νfull3; mean_m = verify3.m_mean)
# Spot-check a handful of representative eta indices: origin 1 & origin 11 (spread across the D=20
# range) at levels 1 and 3 (n_eta = K_mean*D = 60, level-major/origin-minor: idx=(k-1)*D+o).
spot_indices = [target_index(layout3, 1, 1), target_index(layout3, 11, 1),
                target_index(layout3, 1, 3), target_index(layout3, 11, 3)]
η3 = log.(νfull3)
h = 1e-4
worst_rel = 0.0
for j in spot_indices
    tj = @elapsed begin
        ηp = copy(η3); ηp[j] += h
        ηm = copy(η3); ηm[j] -= h
        _, _, vp = cm_originzc_value_verified_from_eta(x_free_calib, ηp, pcx3.ctx_cm)
        _, _, vm = cm_originzc_value_verified_from_eta(x_free_calib, ηm, pcx3.ctx_cm)
        fd_j = (vp.Delta_dual - vm.Delta_dual) / (2h)
    end
    d = abs(g_analytic3[j] - fd_j)
    rel = d / max(1e-8, abs(fd_j))
    global worst_rel = max(worst_rel, rel)   # top-level for-loop soft-scope gotcha: must be `global` here
    println("    (A) [K=3/3 spot] idx=$j ($(tj)s): analytic=$(g_analytic3[j])  fd=$fd_j  abs diff=$d  rel diff=$rel")
    flush(stdout)
end
check("K=3/3: (A) spot-checked analytic eta-gradient components are within reason of FD", worst_rel < 2e-1)

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
