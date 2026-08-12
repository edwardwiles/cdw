# ================================================================================================
# REAL D=20 exact-Hessian convergence + per-callback BLOCK PROFILE for the CM + pairwise-quantile
# family (family #7, 2026-08-12).
#
# Two questions, in this order, and no optimization is done before both are answered:
#   1. Does the inner solve CONVERGE at production scale under `hessopt=exact` (ek_inner_cmpq.opt)?
#   2. Where does the Hessian callback's time actually go, block by block?
#
# The block breakdown comes from the SAME opt-in `@cmhess_prof` instrumentation CM's own callback
# uses (`CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]`, off by default, one Ref check when off), reading
# the SAME `PROF_TIMES` store, so the numbers below are measured on the real production code path --
# not on a re-implementation of it.
#
# WALL-CLOCK CAVEAT, stated here rather than discovered later: this box runs other campaigns and has
# been at load 20-55 on 208 cores all session. An A/B of two runs is worthless here unless the arms
# are interleaved. What this script reports is therefore a WITHIN-RUN block SHARE (all blocks paying
# the same load) plus operation counts and dimensions, not a cross-run speed claim.
#
# EVERY SCIENTIFIC PARAMETER IS A REQUIRED ARGUMENT. Nothing here is defaulted.
#
# Usage:
#   julia --project=. --threads=N full_aod_diag/d4_exact/test_cm_pairwise_quantile_d20_hessian_solve.jl \
#         <W> <L> <G> <n_families> <contrasts> <sigmaHat> <gravity_mask:production|none>
#   e.g.  ... 20000 5 50 2 orthonormal 3.0 production
#
# `gravity_mask` IS REQUIRED, added 2026-08-12 after this file's first numbers turned out to have been
# produced under the SILENT DEFAULT. `d20_real_setup_design` defaults `exclude_diagonal_gravity=false`
# and `gravity_exclude_cells=Tuple{Int,Int}[]`, and this driver originally passed neither -- so it
# solved a DIFFERENT economic problem from the production one (Delta* 0.028738400455 vs
# 0.028782523524 at W=100,000) while reporting both as "the calibration point". Those two kwargs are
# named explicitly in CLAUDE.md's no-silent-defaults rule for exactly this reason.
# ================================================================================================

const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl", "draw_design.jl",
          "country_resolve.jl",   # default_gravity_exclude_cells_brazil_korea (the production mask)
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl",
          "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf

const USAGE = "usage: julia ... test_cm_pairwise_quantile_d20_hessian_solve.jl <W> <L> <G> " *
              "<n_families> <contrasts:anchored|orthonormal> <sigmaHat> <gravity_mask:production|none>"
length(ARGS) == 7 || error(USAGE)
const W_ARG      = parse(Int, ARGS[1])
const L_ARG      = parse(Int, ARGS[2])
const G_ARG      = parse(Int, ARGS[3])
const NFAM_ARG   = parse(Int, ARGS[4])
const CONTR_ARG  = Symbol(ARGS[5])
const SIGMA_ARG  = parse(Float64, ARGS[6])
const GRAVMASK   = Symbol(ARGS[7])
CONTR_ARG in (:anchored, :orthonormal) || error(USAGE)
GRAVMASK in (:production, :none) || error(USAGE)
# :production = the Brazil-Korea exclusion set + diagonal excluded, i.e. what every campaign runs.
# :none = d20_real_setup_design's own defaults, kept ONLY so an old number can be reproduced.
const EXCL_DIAG = GRAVMASK === :production
const GRAV_CELLS = GRAVMASK === :production ? default_gravity_exclude_cells_brazil_korea() : Tuple{Int,Int}[]

@printf("W=%d  L=%d  G=%d  families=%d  contrasts=%s  sigmaHat=%.4g  gravity_mask=%s (exclude_diagonal=%s, %d excluded cells)  julia_threads=%d  OPENBLAS=%s\n",
        W_ARG, L_ARG, G_ARG, NFAM_ARG, String(CONTR_ARG), SIGMA_ARG, String(GRAVMASK),
        string(EXCL_DIAG), length(GRAV_CELLS), Threads.nthreads(),
        get(ENV, "OPENBLAS_NUM_THREADS", "<unset>")); flush(stdout)

t_ctx = @elapsed begin
    global ctx = d20_real_setup_design(; W = W_ARG, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = 20260719,
        destination_sample = :exclude_row, σHat = SIGMA_ARG, inner_lower_limit = -10.0,
        exclude_diagonal_gravity = EXCL_DIAG, gravity_exclude_cells = GRAV_CELLS)
end
@printf("context build: %.2fs   D=%d  size(U)=%s  muHat=%.6g  refIndex1=%d  obj.d=%d\n",
        t_ctx, ctx.D, string(size(ctx.U)), ctx.μHat, ctx.γ.refIndex1, ctx.obj.d); flush(stdout)

const PROD_OPT = joinpath(dirname(D4X), "ek_inner_cmpq.opt")
cfg = CMPairwiseQuantileConfig(L = L_ARG, cm_grid_size = G_ARG, cm_moment_families = NFAM_ARG,
                               contrasts = CONTR_ARG, min_bin_count = 1, mass_start = :uniform)
t_fam = @elapsed begin
    global cmpq = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = PROD_OPT)
end
t_hess = @elapsed begin
    global ctx_cm = cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx = true)
end
octx = ctx_cm.cmpq_hess_ctx
n_x = cmpq.obj_cmpq.outer_constr_index
npacked = div(n_x * (n_x + 1), 2)
@printf("family context: %.2fs   hessian context: %.2fs\n", t_fam, t_hess)
@printf("DIMENSIONS  n_x=%d = NCORE %d + n_restr %d + ncm %d   npair=%d  Lcm=%d  nO=%d\n",
        n_x, cmpq.ncore_econ, cmpq.n_restr, cmpq.ncm, cmpq.op.npair, cmpq.Lcm, cmpq.nO)
@printf("            packed entries=%d   standalone-PQ rows=%d (this family reads %d, %.1f%% unread)\n",
        npacked, n_total_rows(ctx.D, L_ARG), cmpq.n_restr,
        100 * (1 - cmpq.n_restr / n_total_rows(ctx.D, L_ARG)))
@printf("            X-table build ~ W*npair*D = %.3g increments; X storage %.1f MB (x%d families)\n",
        Float64(W_ARG) * cmpq.op.npair * ctx.D,
        (G_ARG) * (L_ARG - 1)^2 * ctx.D * cmpq.op.npair * 8 / 2^20, NFAM_ARG)
@printf("            gates: bin cells checked=%d  min joint count=%d\n",
        cmpq.gates.bin_cells_checked, cmpq.gates.min_joint_count); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
raw_masses = cmpq.raw_start

# ---- one warm callback OUTSIDE the timing, so JIT is not attributed to a block ------------------
θ_econ0 = CS.reconstruct_full(x_free_calib, ctx_cm.m)
prime_operator!(ctx_cm.obj, θ_econ0, ctx, cmpq.core_cf_ref)
reset_for_solve!(ctx_cm.cmpq_fg_state, raw_masses)
x_warm = zeros(n_x); x_warm[1] = 0.1
g_warm = zeros(n_x)
t_fg1 = @elapsed ctx_cm.cmpq_fg_state(x_warm, g_warm)
t_h1 = @elapsed cmpq_fill_hessian_blocks!(octx, ctx_cm.obj)
hbuf = Vector{Float64}(undef, npacked)
HCCv = @view octx.cctx.Hfull[cmpq.ncore_econ+1 : cmpq.ncore_econ+cmpq.ncm,
                             cmpq.ncore_econ+1 : cmpq.ncore_econ+cmpq.ncm]
t_p1 = @elapsed pack_cmpq_hessian!(hbuf, octx.hee_packed, octx.HEQ_pq, octx.HEC, octx.HRR_pq,
                                   octx.HRC, HCCv, octx.sig, cmpq.ncore_econ, cmpq.n_restr, cmpq.ncm)
@printf("JIT/warm pass (DISCARDED): fg=%.2fs  hessian blocks=%.2fs  pack=%.2fs  finite=%s\n",
        t_fg1, t_h1, t_p1, string(all(isfinite, hbuf))); flush(stdout)

# ---- measured, warm: 3 repeats of the block fill + pack, profiled ------------------------------
prof_reset!()
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
const NREP = 3
t_warm = @elapsed for _ in 1:NREP
    cmpq_fill_hessian_blocks!(octx, ctx_cm.obj)
    pack_cmpq_hessian!(hbuf, octx.hee_packed, octx.HEQ_pq, octx.HEC, octx.HRR_pq, octx.HRC, HCCv,
                       octx.sig, cmpq.ncore_econ, cmpq.n_restr, cmpq.ncm)
end
@printf("\n=== per-callback BLOCK PROFILE (%d warm repeats, %.3fs total, %.3fs/callback) ===\n",
        NREP, t_warm, t_warm / NREP)
labels = sort([k for k in keys(PROF_TIMES) if startswith(k, "cmpq_")],
              by = k -> -sum(PROF_TIMES[k]))
tot = sum(sum(PROF_TIMES[k]) for k in labels; init = 0.0)
for k in labels
    s = sum(PROF_TIMES[k])
    @printf("  %-22s %8.4f s total  %8.4f s/callback  %5.1f%%  (%d calls)\n",
            k, s, s / NREP, 100 * s / max(tot, 1e-12), length(PROF_TIMES[k]))
end
@printf("  %-22s %8.4f s total  %8.4f s/callback\n", "SUM(labelled)", tot, tot / NREP)
flush(stdout)
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = false

# ---- the real inner solve ----------------------------------------------------------------------
println("\n=== REAL KNITRO inner solve, hessopt=exact (", basename(PROD_OPT), ") ===")
flush(stdout)
n_hess0 = octx.n_hess_calls
t_solve = @elapsed begin
    global nStatus, xsol, objb, n_fg, n_hess =
        archCMPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm;
                            hess_cb_builder = cmpq_hess_builder_for(ctx_cm))
end
gsol = zeros(n_x)
f_final = ctx_cm.cmpq_fg_state(xsol, gsol)
@printf("nStatus=%d  n_fg=%d  n_hess=%d  wall=%.2fs  f=%.12g  Delta_dual=%.12g  |grad|=%.4e\n",
        nStatus, n_fg, n_hess, t_solve, f_final, -f_final, norm(gsol))
@printf("per-callback share: %.3fs/hess-callback measured warm above; %d callbacks => %.1fs of the %.1fs solve\n",
        t_warm / NREP, n_hess, n_hess * t_warm / NREP, t_solve)
flush(stdout)

ok = nStatus in (0, -100, -101, -102, -103) && isfinite(f_final) && -f_final > 0.0 &&
     n_hess > 0 && octx.n_hess_calls > n_hess0 && all(isfinite, gsol)
println(ok ? "\nRESULT: PASS -- real D=20 inner solve converged under hessopt=exact" :
             "\nRESULT: FAIL -- see status/Delta above")
ok || error("test_cm_pairwise_quantile_d20_hessian_solve: inner solve did not reach an expected state")
