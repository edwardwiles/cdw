# ============================================================================
# Root-cause investigation (shared-FG-verification-and-A-gradient release, 2026-07-27): at the
# EXACT point where inner_fg_backend=:cm_frechet_lookup hard-fails (nStatus=-400) while
# :dense_reference succeeds, directly compare the two backends' own KNITRO FG callback at the SAME
# fixed dual iterate (KNITRO's own initial guess) BEFORE any KNITRO iteration runs. If (f,g) already
# disagree at iterate 0, the bug is in the lookup kernel's forward math itself, not in how KNITRO
# navigates a correct-but-differently-conditioned problem.
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

theta_full0_dense = CS.reconstruct_full(x_free, pcx_dense.ctx_cm.m)
theta_full0_lookup = CS.reconstruct_full(x_free, pcx_lookup.ctx_cm.m)
lp("theta_full0 agree between the two pcx own reconstruct_full: ", theta_full0_dense == theta_full0_lookup)

obj_d = pcx_dense.ctx_cm.obj
obj_l = pcx_lookup.ctx_cm.obj

lp("")
lp("--- Step 1: fire obj.moments! for both (this is what inner_loop_internal_* does first) ---")
obj_d.moments!(view(obj_d.H, :, 1), CS.select_G_from_H(obj_d, obj_d.H), theta_full0_dense, obj_d.U, obj_d)
obj_d.H[:, 2] .= 1.0
lp("  dense: obj.H[1,1]=", obj_d.H[1,1], "  obj.H[1,2]=", obj_d.H[1,2], "  size(obj.H)=", size(obj_d.H))

obj_l.moments!(view(obj_l.H, :, 1), CS.select_G_from_H(obj_l, obj_l.H), theta_full0_lookup, obj_l.U, obj_l)
obj_l.H[:, 2] .= 1.0
lp("  lookup: obj.H[1,1]=", obj_l.H[1,1], "  obj.H[1,2]=", obj_l.H[1,2], "  size(obj.H)=", size(obj_l.H))
lp("  obj.H[:,1] (K column) bit-identical between dense/lookup: ", obj_d.H[:,1] == obj_l.H[:,1])
lp("  max abs delta K: ", maximum(abs.(obj_d.H[:,1] .- obj_l.H[:,1])))

lp("")
lp("--- Step 2: core_cf_ref freshness for the lookup path ---")
cctx_l = pcx_lookup.cctx
cf_state = cctx_l.core_cf_ref[]
lp("  cctx.core_cf_ref[] isa: ", typeof(cf_state))
if cf_state isa Symbol
    lp("  *** core_cf_ref is a SENTINEL (", cf_state, "), not a real CompressedFactual -- lookup path fell back to a non-compressed branch! ***")
end

lp("")
lp("--- Step 3: initial dual guess + direct FG callback comparison at iterate 0 (BEFORE any KNITRO step) ---")
x0 = CS.inner_loop_initial_values(obj_d)
x0_l = CS.inner_loop_initial_values(obj_l)
lp("  x0 (dense) == x0 (lookup) initial dual guess bit-identical: ", x0 == x0_l, "  length=", length(x0))

function dense_q_and_L(obj, zeta, lambda)
    W_ = size(obj.U, 1)
    oci = obj.outer_constr_index
    q = [-zeta - dot(lambda, view(obj.H, s, 1:oci-1)) for s in 1:W_]
    return q
end

if cctx_l.cmlookup_st === nothing
    bins_u = cctx_l.Bidx isa Matrix{UInt32} ? cctx_l.Bidx : Matrix{UInt32}(cctx_l.Bidx)
    ncm_cm = cctx_l.ncm - cctx_l.L
    cctx_l.cmlookup_st = CMFrechetLookupState(obj_l, cctx_l.NCORE, ncm_cm, cctx_l.L, cctx_l.L, cctx_l.D,
        cctx_l.origins, cctx_l.refIndex1, bins_u, cctx_l.R, pcx_lookup.aug.level_targets;
        nthreads_use = 1, core_cf_ref = cctx_l.core_cf_ref)
end
st = cctx_l.cmlookup_st
g_lookup = zeros(length(x0))
f_lookup = st(x0, g_lookup)

zeta0 = x0[1]; lambda0 = collect(x0[2:end])
q_dense = dense_q_and_L(obj_d, zeta0, lambda0)
Psi_dense = similar(q_dense); CS.Psi!(Psi_dense, q_dense)
f_dense = -(sum(Psi_dense)/length(q_dense) + zeta0)
lp("  f_dense (Lfix objective at x0) = ", f_dense)
lp("  f_lookup (CMFrechetLookupState at x0) = ", f_lookup)
lp("  abs(f_dense - f_lookup) = ", abs(f_dense - f_lookup))

lp("")
lp("Done.")
