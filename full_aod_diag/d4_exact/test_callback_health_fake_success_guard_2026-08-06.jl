# Regression test (cm-meanzc-frechet-outer-production-closeout-2026-08-06, task section 3/12):
# reproduces the EXACT old missing-`Pow=` bug (fixed in commit 895b99b) by directly constructing a
# `CMMeanZCOperatorState` the OLD, broken way (no `Pow=` kwarg for a genuinely two-family cctx) and
# proves the new callback-health guard (cm_callback_health.jl) makes it impossible for that to come
# back as a silent, usable `nStatus=0` result -- it must now hard-error instead.
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf
lp(xs...) = (println(xs...); flush(stdout))
npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; lp("  PASS  ", name)
    else
        nfail += 1; lp("  FAIL  ", name)
    end
end

lp("="^90); lp("TEST: callback_health_guard catches the OLD missing-Pow= DimensionMismatch (D4, fast)"); lp("="^90)
ctx = d4_exact_setup()
pcx = build_cm_meanzc_production_context(ctx, CS; L = 5, K_mean = 1, K_pair = 0, contrasts = :orthonormal,
                                          include_truncated_moment = true, meanzc_basis = :direct,
                                          probs = cm_equal_grid_probs(5), moment_representation = :operator)
cctx = pcx.cctx
@assert cctx.n_families == 2 "test setup invalid: cctx must be genuinely two-family for this bug to reproduce"

x_free0 = ctx.θ0_up[ctx.free_idx]
nu1_guess = sum(pcx.aug.Zraw_all[1]) / length(pcx.aug.Zraw_all[1])   # calibration-consistent guess (verify4 method), not an arbitrary 1.0
lp("nu1_guess = ", nu1_guess)

# Reproduce the OLD broken construction directly (pre-895b99b): no Pow= kwarg at all.
bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
st_broken = CMMeanZCOperatorState(pcx.ctx_cm.obj, cctx.ncore_core - 1, cctx.meanzc_zc_op::ZCRestrictionOperator,
    cctx.meanzc_zc_layout, cctx.core_cf_ref, cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R;
    nthreads_use = Threads.nthreads())   # Pow= OMITTED ON PURPOSE -- reproduces the exact old bug
check("test setup: st_broken.Pow is nothing (the bug precondition)", st_broken.Pow === nothing)

reset_for_solve!(st_broken, [nu1_guess])
threw = false
threw_type = Nothing
threw_msg = ""
try
    inner_loop_KNITRO_meanzc_operator(pcx.ctx_cm.obj, st_broken; hess_cb_builder = obj -> archC_hess_cb_builder(cctx))
catch e
    global threw, threw_type, threw_msg
    threw = true
    threw_type = typeof(e)
    threw_msg = sprint(showerror, e)
end
lp("  threw=", threw, " type=", threw_type)
lp("  message[1:300]=", threw_msg[1:min(end, 300)])
check("guarded call THROWS (does not silently return a usable result)", threw)
check("thrown exception is a real ErrorException (fake-success guard), not swallowed as nStatus=0", threw_type === ErrorException)
check("error message identifies this as a masked callback exception, not ordinary infeasibility",
      occursin("callback threw a real Julia exception", threw_msg) || occursin("DimensionMismatch", threw_msg))
check("error message does NOT read like an ordinary CMExpectedSolveFailure (would let a caller silently swallow it)",
      !occursin("CMExpectedSolveFailure", threw_msg))
check("st_broken.n_fg_calls stayed 0 (the crash happened before the counter increment, exactly as in the real 2026-08-06 incident)",
      st_broken.n_fg_calls == 0)

lp("="^90); lp("CONTROL: the FIXED construction path (archC_meanzc_base_state, current HEAD) still succeeds normally"); lp("="^90)
base = archC_meanzc_base_state(x_free0, [nu1_guess], pcx.ctx_cm, cctx)
lp("  inner_status=", base.inner_status, " n_fg_calls(post-solve)=", cctx.cmlookup_st.n_fg_calls)
check("fixed construction path: no exception, real solve occurred (n_fg>0)", cctx.cmlookup_st.n_fg_calls > 0)
check("fixed construction path: cctx.cmlookup_st.Pow is NOT nothing (the actual fix)", cctx.cmlookup_st.Pow !== nothing)

lp(); lp("="^90); lp("TOTAL: $npass passed, $nfail failed"); lp("="^90)
exit(nfail == 0 ? 0 : 1)
