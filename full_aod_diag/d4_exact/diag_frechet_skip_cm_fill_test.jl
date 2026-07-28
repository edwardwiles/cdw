# ============================================================================
# Root-cause investigation, part 3 (shared-FG-verification-and-A-gradient release, 2026-07-27):
# archC_frechet_base_state USED TO set cctx.skip_cm_fill_ref[]=true whenever inner_fg_backend=
# :cm_frechet_lookup, based on a comment's claim that the level block's own dense fill is ALSO
# skippable under the same condition -- i.e. that archC_frechet_hess_cb_builder (the SAME Hessian
# callback used by BOTH dense and lookup, via a thin adapter) never reads obj.H's CM/level dense
# columns either. Part 2 of this investigation showed the lookup FG kernel's own math is correct
# (near-zero gradient at dense's own converged optimum) -- this script tested whether the SKIP
# itself was the bug, by forcibly disabling it (leaving the dense CM/level column fill ON even
# under the lookup FG backend) and re-running the exact failing solve.
#
# STATUS UPDATE (skip_cm_fill_ref removal, 2026-07-27): this investigation's conclusion (Test B
# below) was already applied as a permanent fix in cm_frechet_cplus.jl/cm_frechet_level.jl BEFORE
# this later session started -- archC_frechet_base_state no longer touches any skip ref at all, and
# `wrap_moments_with_cm_frechet_archB`'s dense CM/level fill is unconditional. The later session
# then REMOVED the `skip_cm_fill_ref` `Ref{Bool}` field/plumbing from `CMBinHessCtx` entirely (see
# docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md). Both tests below are updated to not
# reference the now-deleted field; Test A necessarily always "succeeds" now (there is no skip left
# to reproduce the original bug with) and Test B's forced-fill is simply the ONLY behavior that
# exists post-removal -- kept as a smoke check that both call paths still solve correctly, not as a
# live root-cause reproduction anymore.
# ============================================================================
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

lp("Building real D=20/W=80,000 context (destination_sample=:exclude_row)...")
W = 80000
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                             draw_seed = 20260719, destination_sample = :exclude_row)
pe_g = build_pivot_elimination(ctx)
D = ctx.D; Ddest = ctx.D_dest
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe_g)
gp0 = ctx.θ0_up[3+D]
x_free = vcat(gp0 * 1.01, vec(exp.(pivot_expand(zfree0, pe_g))))

L = 50
pcx_lookup = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored,
    cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup)

cctx = pcx_lookup.cctx
lp("cctx.skip_cm_fill_ref no longer exists (removed 2026-07-27) -- the dense CM/level fill is unconditional now")

lp("")
lp("=== Test A: baseline (archC_frechet_base_state, as it normally runs today -- fill always ON) ===")
try
    st = archC_frechet_base_state(x_free, pcx_lookup.ctx_cm, cctx, pcx_lookup.aug.level_targets)
    lp("  SUCCESS (expected -- fill is unconditional post-fix/post-removal): inner_status=", st.inner_status, "  zeta*=", st.ζstar)
catch e
    lp("  UNEXPECTED FAILURE (regression): ", sprint(showerror, e)[1:min(150,end)])
end

lp("")
lp("=== Test B: same solve, direct inner_loop_internal_cmfrechetlookup_production call (fill always ON, unconditionally, no ref/kwarg to force) ===")
obj_l = pcx_lookup.ctx_cm.obj
theta_full0 = CS.reconstruct_full(x_free, pcx_lookup.ctx_cm.m)
cctx.cmlookup_st = nothing
K, xsol, nStatus, n_fg, n_hess = inner_loop_internal_cmfrechetlookup_production(obj_l, theta_full0, cctx,
    pcx_lookup.aug.level_targets; hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, pcx_lookup.aug.level_targets))
lp("  nStatus = ", nStatus, "  (0/-100/-101/-103 = success, -400 = infeasible -- the original crash)")
if nStatus in (0, -100, -101, -103)
    lp("  SUCCESS with the fill unconditionally ON -- consistent with the fix already applied (Test A above uses the identical, now-only, code path)")
    lp("  zeta* = ", xsol[1])
else
    lp("  UNEXPECTED FAILURE -- regression, investigate")
end

lp("")
lp("Done.")
