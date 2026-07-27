# ============================================================================
# Root-cause investigation, part 2 (shared-FG-verification-and-A-gradient release, 2026-07-27):
# the x0=0 comparison in diag_frechet_lookup_vs_dense_callback.jl was degenerate (both f_dense and
# f_lookup trivially ~0 at the all-zero KNITRO starting guess). This script instead: (1) fully
# solves the DENSE reference at the exact failing perturbed point (known to succeed, nStatus=0),
# recovering its converged (zeta*, lambda*); (2) evaluates the LOOKUP kernel's own callable
# st(x,g) AT THAT EXACT (zeta*, lambda*) DENSE solved this claims is (near-)optimal; (3) compares
# f/g directly. If lookup disagrees materially with dense about the gradient AT DENSE'S OWN
# OPTIMUM, that is decisive evidence of a genuine kernel bug, not a trajectory/conditioning
# difference.
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
pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored,
    cm_hessian_backend = :structured, inner_fg_backend = :dense_reference)
pcx_lookup = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored,
    cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup)

lp("Solving DENSE reference at the exact failing perturbed point (known SUCCESS)...")
base_dense = archC_frechet_base_state(x_free, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
lp("  dense converged: inner_status=", base_dense.inner_status, "  zeta*=", base_dense.ζstar)

x_opt = vcat(base_dense.ζstar, base_dense.λstar)

theta_full0_lookup = CS.reconstruct_full(x_free, pcx_lookup.ctx_cm.m)
obj_l = pcx_lookup.ctx_cm.obj
obj_l.moments!(view(obj_l.H, :, 1), CS.select_G_from_H(obj_l, obj_l.H), theta_full0_lookup, obj_l.U, obj_l)
obj_l.H[:, 2] .= 1.0

cctx_l = pcx_lookup.cctx
if cctx_l.cmlookup_st === nothing
    bins_u = cctx_l.Bidx isa Matrix{UInt32} ? cctx_l.Bidx : Matrix{UInt32}(cctx_l.Bidx)
    ncm_cm = cctx_l.ncm - cctx_l.L
    cctx_l.cmlookup_st = CMFrechetLookupState(obj_l, cctx_l.NCORE, ncm_cm, cctx_l.L, cctx_l.L, cctx_l.D,
        cctx_l.origins, cctx_l.refIndex1, bins_u, cctx_l.R, pcx_lookup.aug.level_targets;
        nthreads_use = 1, core_cf_ref = cctx_l.core_cf_ref)
end
st = cctx_l.cmlookup_st
g_lookup = zeros(length(x_opt))
f_lookup = st(x_opt, g_lookup)

# dense reference formula (same as before, mathematically what KNITRO's dense path evaluates)
obj_d = pcx_dense.ctx_cm.obj
oci = obj_d.outer_constr_index
zeta_opt = x_opt[1]; lambda_opt = x_opt[2:end]
q_dense = [-zeta_opt - dot(lambda_opt, view(obj_d.H, s, 1:oci-1)) for s in 1:W]
Psi_dense = similar(q_dense); CS.Psi!(Psi_dense, q_dense)
f_dense_at_opt = -(sum(Psi_dense)/W + zeta_opt)

lp("")
lp("At DENSE's own converged (zeta*, lambda*):")
lp("  f_dense (should be ~0, this IS the KKT stationarity value at the optimum) = ", f_dense_at_opt)
lp("  f_lookup (evaluated at the SAME point) = ", f_lookup)
lp("  abs(f_dense - f_lookup) = ", abs(f_dense_at_opt - f_lookup))
lp("  |g_lookup| (should be ~0 at a true stationary point, like KNITRO's own gradient-norm convergence check) = ", norm(g_lookup))
lp("  max|g_lookup| = ", maximum(abs.(g_lookup)))
lp("  g_lookup[1:5] = ", g_lookup[1:5])

lp("")
lp("Done.")
