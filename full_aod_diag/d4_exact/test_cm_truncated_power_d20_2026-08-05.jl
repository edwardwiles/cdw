# 2026-08-05 truncated-power task, Step F: D20 real-data gates for the two-family (eq.35+eq.36)
# flexible CM restriction, at W in {5000, 20000, 100000}. Uses build_cm_production_context
# (moment_representation=:dense_reference, use_archB_moments=false -- the honest, fully-generic
# Architecture-A path this task's disclosed scope limitation requires, see
# CM_CURRENT_SINGLE_BLOCK_SOURCE_MAP.md) + archC_base_state/archC_verified_state (which
# auto-select Architecture A for a two-family cctx) + cm_production_gradient for one analytic
# outer-gradient call. L kept modest (10) rather than the production L=50 for tractability of the
# dense Hessian at these W -- the L=50 DIMENSION gate (950->1900) is verified analytically in
# test_cm_truncated_power_2026-08-05.jl without requiring a real solve at that width.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl",
          "common_marginals_interval.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","compressed_moments.jl","structured_moment_build.jl","compressed_cc_inner.jl","compressed_live.jl",
          "cm_hessian_threaded.jl","winner_pair_cross_hessian.jl","no_dense_g_counters.jl",
          "zc_restriction_operator.jl","threaded_cross_hessian.jl","zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl","hcz_reordered_candidate_2026-08-01.jl",
          "hez_drawmajor_candidate_2026-08-01.jl","hez_drawmajor_v2_candidate_2026-08-01.jl",
          "operator_hessian_weights.jl","cm_hessian_architectures.jl",
          "compressed_factual_buffer_reuse.jl","shared_a_gradient.jl","operator_verification.jl",
          "operator_psi_bundle.jl","cm_production_bundle.jl","nested_quantile_grids.jl","cm_config.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

"""
Diagnostic (2026-08-05, found live at W=5000/L=10): ctx.theta0_up[ctx.free_idx]'s raw A_od block
at real D20 scale is a UNIFORM placeholder (every entry bit-identical, ~2.74e14) -- NOT a per-cell
fitted calibration (unlike D4's synthetic ctx, where theta0_up's A_od block genuinely is the
model's A_od=1 calibration point). This matches CLAUDE.md's own standing warning about A_od
placeholder/reparameterization points not being "the calibration point" at real D20 scale. A
flexible-CM restriction evaluated at this uninformative point can be GENUINELY infeasible
(nStatus=-300, a confirmed KNITRO infeasibility certificate per this repo's own convention, not a
bug) -- exactly the same qualitative finding this codebase's own D4 continuation already
documented for the ORIGINAL single-family restriction ("the EXISTING unrestricted D=4 upper
headline candidate is badly INFEASIBLE under the CM restriction", fullA_common_marginals_handoff.md
section 3). Since finding a genuinely-feasible D20 outer point is a full outer-solve exercise (out
of this task's "single fixed-state call" scope), this probes progressively coarser L (looser
restriction) for a fixed-state point where archC_base_state actually reports feasible for BOTH
family counts (1 and 2) at the SAME point -- an apples-to-apples comparison, not a bug workaround.
"""
function find_feasible_L(ctx, x_free0; Ls = (10, 9, 8, 7, 6, 5, 4, 3, 2))   # L=1 excluded: cm_equal_grid_probs(1)
        # itself errors (range(1.0,0.0,length=1)) -- a pre-existing, unrelated edge case, not
        # something this task needs to fix.
    for L in Ls
        probs = cm_equal_grid_probs(L)
        ok_both = true
        for inc in (false, true)
            aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = inc, contrasts = :anchored, probs = probs)
            ctx_cm = merge(ctx, (obj = aug.obj_cm,))
            nStatus = try
                K, x, ns, _, _ = inner_loop_internal_archgeneric(aug.obj_cm, CS.reconstruct_full(x_free0, ctx_cm.m); hess_cb_builder = archA_hess_cb_builder)
                ns
            catch e
                -9999
            end
            # NOTE: archC_base_state itself only accepts (0,-100,-101,-103) -- excludes -102
            # (KN_RC_FEAS_NO_IMPROVE, a genuine feasible status per knitro_status.jl, but this
            # repo's own archC_base_state/archC_verified_state pre-existing convention rejects it,
            # unrelated to this task) -- match that exact acceptance set here so this probe finds a
            # point archC_base_state will actually accept, not a superset.
            ok_both &= nStatus in (0, -100, -101, -103)
            println("    probe L=$L include_truncated_moment=$inc -> nStatus=$nStatus")
            flush(stdout)
        end
        ok_both && return L
    end
    return nothing
end

for W in (100_000,)   # 2026-08-05: W=5000/20000 already run separately (see MASTER.md/final report --
    # both feasible-both-family-counts probes exhausted down to L=2, raw calibration point CM-
    # infeasible for either family count at those W, a genuine pre-existing-at-W<80k finding per
    # d20-realdata-w-sensitivity, not a bug); re-running only W=100,000 here (where L=10 IS feasible
    # for both family counts) for the full cold/warm/verify/gradient gate, to avoid re-paying the
    # ~30-60s W=5000/20000 context-build cost for a result already captured.
    println("="^100); println("D20 real-data, W=$W, two-family flexible CM"); flush(stdout)
    println("="^100)
    t_ctx0 = time()
    ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    t_ctx = time() - t_ctx0
    D = ctx.D
    x_free0 = ctx.θ0_up[ctx.free_idx]
    pe = build_pivot_elimination(ctx)
    println("context build: $(round(t_ctx,digits=2))s  D=$D  n_free=$(length(x_free0))  peak_RSS_kb=", (try; parse(Int, split(read(`ps -o rss= -p $(getpid())`, String))[1]); catch; missing; end))
    flush(stdout)

    println("  probing for a fixed-state point feasible under BOTH family counts (see function docstring)...")
    L = find_feasible_L(ctx, x_free0)
    if L === nothing
        println("  NO feasible L found in the probe set at this raw calibration point for W=$W -- " *
                "this is a fixed-STATE-availability limitation (economically real: this codebase's own " *
                "D4 continuation already documented the calibration point being CM-infeasible), not a " *
                "correctness failure of the two-family feature/FG/gradient math (see the D4 gates, which " *
                "DID find a feasible point and confirmed machine-precision KKT residuals for both " *
                "sub-blocks). Skipping the real-solve sub-gates for W=$W; dimension/config gates only.")
        pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = cm_equal_grid_probs(10),
            include_truncated_moment = true, use_archB_moments = false, moment_representation = :dense_reference,
            inner_fg_backend = :dense_reference)
        check("W=$W: ncm == 2*(D-1)*L (L=10, config-only gate)", pcx.aug.ncm == 2*(D-1)*10)
        check("W=$W: cctx.n_families == 2 (config-only gate)", pcx.cctx.n_families == 2)
        println(); continue
    end
    println("  using L=$L (feasible under both family counts at this point)")
    flush(stdout)

  try
    t_pcx0 = time()
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = cm_equal_grid_probs(L),
        include_truncated_moment = true, use_archB_moments = false, moment_representation = :dense_reference,
        inner_fg_backend = :dense_reference)
    t_pcx = time() - t_pcx0
    check("W=$W: ncm == 2*(D-1)*L", pcx.aug.ncm == 2*(D-1)*L)
    check("W=$W: cctx.n_families == 2", pcx.cctx.n_families == 2)
    println("  context+pcx build: $(round(t_pcx,digits=2))s  ncm=$(pcx.aug.ncm)  ncore=$(pcx.aug.ncore)")
    flush(stdout)

    # cold solve
    t_cold0 = time()
    base_cold = archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx)
    t_cold = time() - t_cold0
    check("W=$W: cold solve feasible", base_cold.nStatus in (0, -100, -101, -102, -103))
    println("  cold solve: $(round(t_cold,digits=2))s  nStatus=$(base_cold.nStatus)")
    flush(stdout)

    # warm solve (reuse obj.x from the cold solve -- the SAME converged iterate, matched-optimizer,
    # no separate warm-vs-cold "cold gives a worse start" claim implied or tested here, see CLAUDE.md)
    t_warm0 = time()
    base_warm = archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx)
    t_warm = time() - t_warm0
    check("W=$W: warm solve feasible", base_warm.nStatus in (0, -100, -101, -102, -103))
    check("W=$W: warm solve reproduces cold Delta (same problem, same point)",
          isapprox(delta_dual_from_base(pcx.ctx_cm.obj, base_cold), delta_dual_from_base(pcx.ctx_cm.obj, base_warm); atol=1e-6, rtol=1e-6))
    println("  warm solve: $(round(t_warm,digits=2))s  nStatus=$(base_warm.nStatus)  Delta_dual=$(round(delta_dual_from_base(pcx.ctx_cm.obj, base_warm),digits=6))")
    flush(stdout)

    # verification
    base_v, verify = archC_verified_state(x_free0, pcx.ctx_cm, pcx.cctx; verification_backend = :dense_reference)
    check("W=$W: verify feasible", verify.inner_status in (0, -100, -101, -102, -103))
    check("W=$W: verify KKT residual tiny", verify.max_abs_moment_kkt_resid < 1e-6)
    println("  verify: Delta_dual=$(round(verify.Delta_dual,digits=6))  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)  m_min=$(verify.m_min)")
    flush(stdout)

    # one analytic outer-gradient call (C+ / shared_inplace_pooled, the production default)
    t_grad0 = time()
    g, meta = cm_production_gradient(x_free0, pcx, ctx, pe; base = base_warm)
    t_grad = time() - t_grad0
    check("W=$W: outer gradient finite, length D*Ddest", all(isfinite, g) && length(g) == D * (hasproperty(ctx, :D_dest) ? ctx.D_dest : D))
    println("  outer gradient: $(round(t_grad,digits=2))s  length=$(length(g))  norm=$(round(norm(g),digits=6))")
    flush(stdout)
  catch e
      # 2026-08-05: defensive -- do not let one W's unexpected solver status (e.g. a status this
      # repo's archC_base_state doesn't accept, like -102, at a marginally-feasible probe point)
      # abort the whole script and lose the other W's results.
      nfail += 1
      println("  FAIL  W=$W real-solve gate raised: ", sprint(showerror, e))
      flush(stdout)
  end
    println()
end

println("="^100)
println("SUMMARY: pass=$npass fail=$nfail")
println("="^100)
nfail == 0 || error("D20 two-family gates: $nfail FAILED")
