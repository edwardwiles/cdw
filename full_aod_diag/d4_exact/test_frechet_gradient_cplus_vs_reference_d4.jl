# D=4 Part IV gate: common-Frechet C+ (factorized envelope) gradient vs Reference (non-factorized)
# envelope gradient, at the SAME solved dual state -- task's primary implementation gate ("machine
# precision agreement"). Also runs the identical comparison for flexible CM as a control, so the two
# can be reported side by side (the two backends' agreement should be comparable in magnitude,
# not materially worse for common Frechet).
const _D4E = @__DIR__
include(joinpath(_D4E, "context.jl"))
for f in ["winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl",
          "gradient_workspace.jl","lfix_factorized_workspace.jl","lfix_cm_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_cplus.jl","cm_config.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free0 = ctx.θ0_up[ctx.free_idx]
D = ctx.D
Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
W = size(ctx.U, 1)
L = 10
pe = build_pivot_elimination(ctx)
pool = build_grad_workspace_pool(W)
ws = build_lfix_factorized_workspace(D, Ddest, W)

function report_agreement(label, g_ref, g_cp)
    diffvec = g_cp .- g_ref
    absdiff = abs.(diffvec)
    reldiff = absdiff ./ (abs.(g_ref) .+ 1e-300)
    maxabs, imax = findmax(absdiff)
    maxrel, imaxrel = findmax(reldiff)
    @printf("  %s: length=%d  max|diff|=%.3e (at k=%d)  max relative=%.3e (at k=%d)\n",
            label, length(g_ref), maxabs, imax, maxrel, imaxrel)
    return maxabs
end

println("== FLEXIBLE CM control ==")
aug_cm = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
ctx_cm = merge(ctx, (obj = aug_cm.obj_cm,))
bins_cm = cm_bin_indices_for(ctx, aug_cm)
base_cm = solve_base_state(x_free0, ctx_cm)
check("CM base solve feasible", base_cm.inner_status in (0, -100, -101, -103))
g_ref_cm, _ = composite_gradient_at_fast_cm(x_free0, ctx_cm, pe, ctx, aug_cm, bins_cm; base = base_cm,
    h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
g_cp_cm, _ = composite_gradient_at_Cplus_cm(x_free0, ctx_cm, pe, ctx, aug_cm, bins_cm, pool, ws, nothing;
    base = base_cm, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
maxabs_cm = report_agreement("CM Reference vs C+", g_ref_cm, g_cp_cm)
check("CM Reference vs C+ agree to < 1e-8", maxabs_cm < 1e-8)

println()
println("== COMMON FRECHET (dense/Architecture-A obj, contrasts=:anchored) ==")
aug_f = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
ctx_f = merge(ctx, (obj = aug_f.obj_cm,))
bins_f = cm_bin_indices_for(ctx, aug_f)
base_f = solve_base_state(x_free0, ctx_f)
check("Frechet base solve feasible", base_f.inner_status in (0, -100, -101, -103))

g_ref_f, _ = composite_gradient_at_fast_frechet(x_free0, ctx_f, pe, ctx, aug_f, bins_f; base = base_f,
    h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
g_cp_f, _ = composite_gradient_at_Cplus_frechet(x_free0, ctx_f, pe, ctx, aug_f, bins_f, pool, ws, nothing;
    base = base_f, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
maxabs_f = report_agreement("Frechet Reference vs C+", g_ref_f, g_cp_f)
check("Frechet Reference vs C+ agree to < 1e-8", maxabs_f < 1e-8)

println()
println("== summary: is Frechet's C+ vs Reference agreement comparable to (not materially worse than) CM's? ==")
@printf("  CM max|diff|=%.3e   Frechet max|diff|=%.3e   ratio=%.2fx\n", maxabs_cm, maxabs_f, maxabs_f / max(maxabs_cm, 1e-300))
check("Frechet discrepancy not materially worse than CM's (ratio < 100x)", maxabs_f < 100 * max(maxabs_cm, 1e-14))

# ---- sanity: dM/dA=0 property -- the fixed marginal block contributes ONLY through the additive q0
# correction (constant across every outer-coordinate probe), never through a per-coordinate term of
# its own -- verified structurally by construction (frechet_cm_level_fixed_contribution is computed
# ONCE, outside the coordinate loop, in both build_lfix_base_cache_cm_frechet(_C!)) -- this check
# confirms the two gradient vectors' FIRST (gamma/analytic) component, which does NOT touch the CM/
# level tail at all, also agrees, as an additional consistency signal. ----
check("gamma-component (g[1]) agrees between CM Reference/C+", abs(g_ref_cm[1] - g_cp_cm[1]) < 1e-10)
check("gamma-component (g[1]) agrees between Frechet Reference/C+", abs(g_ref_f[1] - g_cp_f[1]) < 1e-10)

println()
println("==================================================")
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
