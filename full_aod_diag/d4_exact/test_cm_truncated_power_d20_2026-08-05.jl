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
          "operator_psi_bundle.jl","cm_production_bundle.jl"]
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

L = 10
for W in (5_000, 20_000, 100_000)
    println("="^100); println("D20 real-data, W=$W, L=$L, two-family flexible CM"); flush(stdout)
    println("="^100)
    t_ctx0 = time()
    ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    t_ctx = time() - t_ctx0
    D = ctx.D
    x_free0 = ctx.θ0_up[ctx.free_idx]
    pe = build_pivot_elimination(ctx)
    println("context build: $(round(t_ctx,digits=2))s  D=$D  n_free=$(length(x_free0))  peak_RSS_kb=", (try; parse(Int, split(read(`ps -o rss= -p $(getpid())`, String))[1]); catch; missing; end))

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
    println()
end

println("="^100)
println("SUMMARY: pass=$npass fail=$nfail")
println("="^100)
nfail == 0 || error("D20 two-family gates: $nfail FAILED")
