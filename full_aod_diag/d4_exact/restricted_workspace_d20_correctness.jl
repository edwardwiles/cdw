# ============================================================================
# Phase B (2026-07-24) B9: supplementary D=20/W=80,000/L=50 correctness gate.
#
# Written because the pre-existing production release gates
# (d20_meanzc_release_gates.jl / d20_originzc_fixedpoint_gates.jl) both load
# a "Point B" seed checkpoint (cm_campaign_2026-07-22/chain1/delta_1.0/
# cold_verified_seed.jls) that is now dimensionally INCOMPATIBLE with this
# context's pivot-elimination convention (DimensionMismatch: new dimensions
# (20,20) must be consistent with array length 380) -- a stale/incompatible
# checkpoint from another concurrent session, unrelated to this task's own
# change (the crash happens at seed-deserialization time, before any of
# this task's modified code runs). Rather than edit those shared production
# gate scripts (used by other concurrent sessions) to work around someone
# else's stale artifact, this script re-derives the SAME correctness content
# (dense vs cached moment/gradient/KKT agreement) using only Point A
# (ctx.θ0_up[ctx.free_idx], no external checkpoint dependency).
#
# D=4 gates (test_cm_meanzc_pure_moments.jl, test_cm_meanzc_d4_gates.jl,
# test_cm_originzc_pure_moments.jl) already exhaustively validate the
# dense/cached equivalence (moment columns, Delta_dual, KKT, Hessian,
# analytic-vs-FD envelope derivative) across K=1/K=2, direct/anchored basis.
# This script's job is only to confirm the SAME equivalence holds at real
# D=20/W=80,000 scale, which the D=4 gates cannot by themselves guarantee
# (different BLAS/threading/numerical regime).
#
# Usage: julia --project=. full_aod_diag/d4_exact/phaseB_b9_d20_correctness.jl
# ============================================================================
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
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
using Printf, LinearAlgebra, Statistics, Dates

lp(xs...) = (println(xs...); flush(stdout))
const W = 80_000
const L = 50
const DRAW_SEED = 20260719
const CONTRASTS = :orthonormal
ok_all = true

lp(">>> [", now(), "] building D=20/W=", W, " real-data context (draw_seed=", DRAW_SEED, ")...")
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = DRAW_SEED)
pe = build_pivot_elimination(ctx)
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]
x_free_A = ctx.θ0_up[ctx.free_idx]
lp(">>> context built. D=", ctx.D)

function build_dense_meanzc(ctx, aug)
    moments_dense! = wrap_moments_with_cm_meanzc_dense(ctx.obj.moments!, aug.ncore_econ, aug.CM,
        aug.Zraw_all, aug.Zpairraw_all; meanzc_basis = aug.meanzc_basis, refIndex1 = aug.refIndex1)
    obj0 = aug.obj_cm
    return CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_dense!, moments_jacobian! = error,
        d = obj0.d, outer_constr_index = obj0.outer_constr_index,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
end

function build_dense_originzc(ctx, aug, layout)
    moments_dense! = wrap_moments_with_originzc_dense(ctx.obj.moments!, aug.ncore_econ, aug.Zraw_all, aug.Zpairraw_all, layout)
    obj0 = aug.obj_cm
    return CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_dense!, moments_jacobian! = error,
        d = obj0.d, outer_constr_index = obj0.outer_constr_index,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
end

lp()
lp("="^100)
lp("CM+mean/ZC (K_mean=1, K_pair=1): dense vs cached, real D=20/W=80,000")
lp("="^100)
nu0 = [1.0]
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 1, contrasts = CONTRASTS,
    meanzc_basis = :direct, probs = probs)
ctx_cached = merge(ctx, (obj = aug.obj_cm,))
cctx_cached = build_cm_meanzc_bin_ctx(ctx, aug)
ctx_dense = merge(ctx, (obj = build_dense_meanzc(ctx, aug),))
cctx_dense = build_cm_meanzc_bin_ctx(ctx, aug)

base_c, verify_c = archC_meanzc_verified_state(x_free_A, nu0, ctx_cached, cctx_cached)
base_d, verify_d = archC_meanzc_verified_state(x_free_A, nu0, ctx_dense, cctx_dense)

lp("  Delta_dual   : cached=", verify_c.Delta_dual, "  dense=", verify_d.Delta_dual,
   "  |diff|=", abs(verify_c.Delta_dual - verify_d.Delta_dual))
lp("  gap          : cached=", verify_c.primal_dual_gap, "  dense=", verify_d.primal_dual_gap)
lp("  kkt_resid    : cached=", verify_c.max_abs_moment_kkt_resid, "  dense=", verify_d.max_abs_moment_kkt_resid)
lp("  lambda* diff : max|cached-dense|=", maximum(abs.(base_c.λstar .- base_d.λstar)))
ok1 = abs(verify_c.Delta_dual - verify_d.Delta_dual) < 1e-10 &&
      abs(verify_c.primal_dual_gap - verify_d.primal_dual_gap) < 1e-10 &&
      maximum(abs.(base_c.λstar .- base_d.λstar)) < 1e-8
global ok_all &= ok1
lp("  PASS(inner solve equivalence)=", ok1)

pcx_c = (ctx_cm = ctx_cached, aug = aug, cctx = cctx_cached, bins = cm_bin_indices_for(ctx, aug))
pcx_d = (ctx_cm = ctx_dense, aug = aug, cctx = cctx_dense, bins = cm_bin_indices_for(ctx, aug))
g_c, _ = cm_meanzc_production_gradient(x_free_A, nu0, pcx_c, ctx, pe; base = base_c, verify = verify_c)
g_d, _ = cm_meanzc_production_gradient(x_free_A, nu0, pcx_d, ctx, pe; base = base_d, verify = verify_d)
gmaxdiff = maximum(abs.(g_c .- g_d))
grelmax = maximum(abs.(g_c .- g_d) ./ max.(abs.(g_c), 1e-8))
lp("  full outer gradient (g_econ + eta_nu): max|diff|=", gmaxdiff, " max relative=", grelmax, " length=", length(g_c))
ok2 = gmaxdiff < 1e-6
global ok_all &= ok2
lp("  PASS(gradient equivalence)=", ok2)

lp()
lp("="^100)
lp("origin-ZC (K_mean=1, K_pair=1, SharedByPowerLayout): dense vs cached, real D=20/W=80,000")
lp("="^100)
layout1 = SharedByPowerLayout(1, 1)
nu0_oz = [mean(ctx.U .^ k) for k in 1:layout1.K_mean]
aug_oz = build_originzc_augmented_obj(ctx, CS, layout1)
ctx_cached_oz = merge(ctx, (obj = aug_oz.obj_cm,))
ctx_dense_oz = merge(ctx, (obj = build_dense_originzc(ctx, aug_oz, layout1),))

base_c2, verify_c2 = archOZ_verified_state(x_free_A, nu0_oz, ctx_cached_oz)
base_d2, verify_d2 = archOZ_verified_state(x_free_A, nu0_oz, ctx_dense_oz)
lp("  Delta_dual   : cached=", verify_c2.Delta_dual, "  dense=", verify_d2.Delta_dual,
   "  |diff|=", abs(verify_c2.Delta_dual - verify_d2.Delta_dual))
lp("  gap          : cached=", verify_c2.primal_dual_gap, "  dense=", verify_d2.primal_dual_gap)
lp("  kkt_resid    : cached=", verify_c2.max_abs_moment_kkt_resid, "  dense=", verify_d2.max_abs_moment_kkt_resid)
lp("  lambda* diff : max|cached-dense|=", maximum(abs.(base_c2.λstar .- base_d2.λstar)))
ok3 = abs(verify_c2.Delta_dual - verify_d2.Delta_dual) < 1e-10 &&
      abs(verify_c2.primal_dual_gap - verify_d2.primal_dual_gap) < 1e-10 &&
      maximum(abs.(base_c2.λstar .- base_d2.λstar)) < 1e-8
global ok_all &= ok3
lp("  PASS(inner solve equivalence)=", ok3)

pcx_c2 = (ctx_cm = ctx_cached_oz, aug = aug_oz)
pcx_d2 = (ctx_cm = ctx_dense_oz, aug = aug_oz)
g_c2, _ = cm_originzc_production_gradient(x_free_A, nu0_oz, pcx_c2, ctx, pe; base = base_c2, verify = verify_c2)
g_d2, _ = cm_originzc_production_gradient(x_free_A, nu0_oz, pcx_d2, ctx, pe; base = base_d2, verify = verify_d2)
gmaxdiff2 = maximum(abs.(g_c2 .- g_d2))
grelmax2 = maximum(abs.(g_c2 .- g_d2) ./ max.(abs.(g_c2), 1e-8))
lp("  full outer gradient (g_econ + eta): max|diff|=", gmaxdiff2, " max relative=", grelmax2, " length=", length(g_c2))
ok4 = gmaxdiff2 < 1e-6
global ok_all &= ok4
lp("  PASS(gradient equivalence)=", ok4)

lp()
lp("="^100)
lp(ok_all ? "ALL B9 D=20 SUPPLEMENTARY CORRECTNESS CHECKS PASSED" : "SOME B9 D=20 CHECKS FAILED -- SEE ABOVE")
lp("="^100)
lp(">>> DONE at ", now())
