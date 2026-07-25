# Correctness gate for cm_frechet_cdf_only_gradient.jl's new frechet_fixed_contribution_archB
# lookup kernel, against the EXISTING dense reference frechet_fixed_contribution (used unchanged by
# the :cdf_power path). Task: FIXED_FRECHET_INNER_SOLVER production-feasibility 2026-07-24
# (CDF-only addendum) -- fills the "only :cdf_power is wired" outer-gradient gap.
include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_cdf_only_gradient.jl"))
using Printf, LinearAlgebra

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

println("="^100); println("D=4 SETUP"); println("="^100)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; W = size(ctx.U, 1)
const L = 8
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)

cfg_frec = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_only, frechet_basis = :cumulative)
targets = build_frechet_reference_targets(ctx, cfg_frec; L = L)

# ---- archB (fast, no dense CM matrix) -- this is what production/:cdf_only actually uses ----
aug_archB = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = :orthonormal)
fctx_archB = build_cm_frechet_bin_ctx(ctx, aug_archB)
K, x_sol, nStatus, _, _ = inner_loop_internal_archgeneric(aug_archB.obj_cm, θ_full0;
    hess_cb_builder = _o -> archC_frechet_hess_cb_builder(fctx_archB))
check("archB CDF-only solve feasible", nStatus in (0,-100,-101,-103))
base_archB = BaseDualState(collect(x_free_calib), θ_full0, x_sol[1], collect(x_sol[2:end]), copy(aug_archB.obj_cm.arg1), nStatus)

# ---- dense (Architecture A) reference: SAME layout, different materialization ----
aug_dense = build_cm_frechet_augmented_obj_basis(ctx, CS, targets; basis = :cumulative, feature_set = :cdf_only, contrasts = :orthonormal)
check("dense aug ncm == archB aug ncm (same layout)", aug_dense.ncm == aug_archB.ncm)
check("dense aug ncore == archB aug ncore", aug_dense.ncore == aug_archB.ncore)

# ---- the actual gate: given the SAME base (base_archB), do the two evaluation mechanisms agree? ----
out_lookup = frechet_fixed_contribution_archB(base_archB, aug_archB, fctx_archB)
out_dense = frechet_fixed_contribution(base_archB, aug_dense)
maxerr = maximum(abs.(out_lookup .- out_dense))
relerr = maxerr / (maximum(abs.(out_dense)) + 1e-300)
println("max|out_lookup - out_dense| = $maxerr  (relative: $relerr,  W=$(length(out_lookup)))")
check("frechet_fixed_contribution_archB matches dense reference to 1e-8 abs", maxerr < 1e-8)

# ---- end-to-end: cm_frechet_production_gradient_cdf_only runs and returns a finite gradient,
#      cross-checked against a central finite difference of the SAME Delta*-style scalar the
#      CDF+POWER gate already trusts (delta_dual_from_base) ----
pe = build_pivot_elimination(ctx)
cfg2 = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_only, frechet_basis = :cumulative)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg2; L = L)
z_star = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D))
w0 = vcat(x_free_calib[1], pivot_reduce(z_star, pe))
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

g, meta = cm_frechet_production_gradient_cdf_only(x_free_from_w(w0), fpcx, pe)
check("cm_frechet_production_gradient_cdf_only returns finite gradient of correct length", all(isfinite, g) && length(g) == length(w0))
println("g[1:5] = ", g[1:min(5,end)])

function Delta_at_w(w)
    xf = x_free_from_w(w)
    base = cm_frechet_base_state(xf, fpcx)
    return delta_dual_from_base(fpcx.ctx_cm.obj, base)
end
idx_check = [1, 2, 3]
hs = [1e-3, 1e-4, 1e-5, 1e-6]
for i in idx_check
    println("coord $i: analytic g=$(g[i])")
    for h in hs
        wp = copy(w0); wp[i] += h
        wm = copy(w0); wm[i] -= h
        fd = (Delta_at_w(wp) - Delta_at_w(wm)) / (2h)
        println(@sprintf("  h=%.0e: finite-diff=%.8f  diff=%.2e  rel=%.2e", h, fd, abs(g[i]-fd), abs(g[i]-fd)/(abs(g[i])+1e-300)))
    end
end
# Use the BEST (smallest-error) h per coordinate for the pass/fail gate -- the point of the sweep
# above is to distinguish "wrong at every h" (bug) from "wrong only at bad h" (FD/solver-noise,
# since Delta_at_w re-solves the inner KNITRO problem at each perturbed point).
for i in idx_check
    best = minimum(abs(g[i] - (Delta_at_w(copy(w0) .+ h.*(1:length(w0).==i)) - Delta_at_w(copy(w0) .- h.*(1:length(w0).==i)))/(2h)) for h in hs)
    check("coord $i: analytic gradient matches SOME finite-diff h to 1e-3 abs (best-of-sweep=$best)", best < 1e-3)
end

# ---- ISOLATION CHECK: does the SAME z-direction mismatch appear on the EXISTING, unmodified
#      :cdf_power gradient path (cm_frechet_production_gradient, dense aug.CM matvec,
#      frechet_fixed_contribution -- code this session did NOT touch)? If yes, the discrepancy is
#      pre-existing/shared (composite_gradient_at_fast's handling of the Frechet "common" pinning
#      block generally), not something this session's new archB kernel introduced. If no, the bug
#      is specific to this session's new code and needs to be found. ----
println(); println("="^100); println("ISOLATION CHECK: same test against EXISTING :cdf_power gradient path"); println("="^100)
cfg_cp = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
fpcx_cp = build_cm_frechet_production_context(ctx, CS, cfg_cp; L = L)
g_cp, meta_cp = cm_frechet_production_gradient(x_free_from_w(w0), fpcx_cp, pe)
function Delta_at_w_cp(w)
    xf = x_free_from_w(w)
    base = cm_frechet_base_state(xf, fpcx_cp)
    return delta_dual_from_base(fpcx_cp.ctx_cm.obj, base)
end
for i in idx_check
    println("coord $i: analytic g_cdf_power=$(g_cp[i])")
    for h in hs
        wp = copy(w0); wp[i] += h
        wm = copy(w0); wm[i] -= h
        fd = (Delta_at_w_cp(wp) - Delta_at_w_cp(wm)) / (2h)
        println(@sprintf("  h=%.0e: finite-diff=%.8f  diff=%.2e", h, fd, abs(g_cp[i]-fd)))
    end
end

println()
println("="^100)
println("RESULT: $n_pass passed, $n_fail failed")
println("="^100)
exit(n_fail == 0 ? 0 : 1)
