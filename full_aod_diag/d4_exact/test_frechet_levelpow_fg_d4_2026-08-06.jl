# D4 gate: common-Fréchet TWO-FAMILY operator FG (CMFrechetLookupState + levelpow extension,
# cm_frechet_lookup_kernels.jl) vs a TRUE, independent dense-reference FG (PsiObjectiveBundleImplicit
# built directly by build_cm_frechet_level_augmented_obj -- NOT this same session's own new code) at
# the SAME real point, several random dual (zeta,lambda) vectors. Also checks
# _verify_inner_solution_operator_cm_core's own levelpow extension (operator_verification.jl)
# against the SAME dense reference, independently of CMFrechetLookupState.
const _D4E = @__DIR__
include(joinpath(_D4E, "context.jl"))
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
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
using LinearAlgebra, Random, Printf

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
L = 10

println("="^90); println("CONTROL: single-family (include_truncated_moment=false), harness sanity check"); println("="^90)
let contrasts = :anchored
    aug_dense1 = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = contrasts, include_truncated_moment = false)
    obj_dense1 = aug_dense1.obj_cm
    n1 = obj_dense1.outer_constr_index
    println("  n1=", n1, " ncore=", aug_dense1.ncore, " ncm_cm=", aug_dense1.ncm_cm, " ncm_level=", aug_dense1.ncm_level)
    # PRIME obj_dense1.H first -- the FG callable only READS H (BLAS.gemv! on H), it never calls
    # moments! itself; H is left undef/zero until this priming call fills it.
    θ_full0_dense1 = CS.reconstruct_full(x_free0, ctx.m)
    obj_dense1.moments!(@view(obj_dense1.H[:, 1]), CS.select_G_from_H(obj_dense1, obj_dense1.H), θ_full0_dense1, obj_dense1.U, obj_dense1)
    pcx1 = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = false,
        moment_representation = :operator)
    θ_full0_1 = CS.reconstruct_full(x_free0, pcx1.ctx_cm.m)
    obj_op1 = pcx1.ctx_cm.obj
    prime_operator!(obj_op1, θ_full0_1, pcx1.cctx.econ_ctx, pcx1.cctx.core_cf_ref; restriction_state = pcx1.cctx)
    bins_u1 = pcx1.cctx.Bidx isa Matrix{UInt32} ? pcx1.cctx.Bidx : Matrix{UInt32}(pcx1.cctx.Bidx)
    st1 = CMFrechetLookupState(obj_op1, pcx1.cctx.NCORE, pcx1.cctx.ncm - L, L, L, D,
        pcx1.cctx.origins, pcx1.cctx.refIndex1, bins_u1, pcx1.cctx.R, pcx1.aug.level_targets;
        core_cf_ref = pcx1.cctx.core_cf_ref)
    Random.seed!(1)
    x1 = 0.1 .* randn(n1)
    g_dense1 = zeros(n1); f_dense1 = obj_dense1(x1, g_dense1)
    g_op1 = zeros(n1); f_op1 = st1(x1, g_op1)
    rel_f1 = abs(f_dense1 - f_op1) / max(abs(f_dense1), abs(f_op1), 1e-10)
    rel_g1 = norm(g_dense1 - g_op1) / max(norm(g_dense1), norm(g_op1), 1e-10)
    check("CONTROL single-family f: dense=$f_dense1 op=$f_op1 rel=$rel_f1 < 1e-8", rel_f1 < 1e-8)
    check("CONTROL single-family g: rel=$rel_g1 < 1e-8", rel_g1 < 1e-8)
    println("    g_dense1[1:5]=", g_dense1[1:5])
    println("    g_op1[1:5]=", g_op1[1:5])
    println("    g_dense1[end-5:end]=", g_dense1[end-5:end])
    println("    g_op1[end-5:end]=", g_op1[end-5:end])
end

for contrasts in (:anchored, :orthonormal)
    println("="^90); println("contrasts = ", contrasts); println("="^90)

    # ---- TRUE dense reference (independent of this session's operator kernel changes) ----
    aug_dense = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = contrasts, include_truncated_moment = true)
    obj_dense = aug_dense.obj_cm
    n = obj_dense.outer_constr_index
    ncore1 = aug_dense.ncore - 1
    ncm_cm = aug_dense.ncm_cm      # 2*(D-1)*L
    ncm_level = aug_dense.ncm_level  # 2*L
    println("  n=", n, " ncore1=", ncore1, " ncm_cm=", ncm_cm, " ncm_level=", ncm_level,
        " total=", 1 + ncore1 + ncm_cm + ncm_level)
    θ_full0_dense = CS.reconstruct_full(x_free0, ctx.m)
    obj_dense.moments!(@view(obj_dense.H[:, 1]), CS.select_G_from_H(obj_dense, obj_dense.H), θ_full0_dense, obj_dense.U, obj_dense)

    # ---- operator context (this session's new levelpow extension) ----
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = true,
        moment_representation = :operator)
    check("cctx.n_families == 2", pcx.cctx.n_families == 2)
    check("cctx.Pow !== nothing", pcx.cctx.Pow !== nothing)

    θ_full0 = CS.reconstruct_full(x_free0, pcx.ctx_cm.m)
    obj_op = pcx.ctx_cm.obj
    prime_operator!(obj_op, θ_full0, pcx.cctx.econ_ctx, pcx.cctx.core_cf_ref; restriction_state = pcx.cctx)
    bins_u = pcx.cctx.Bidx isa Matrix{UInt32} ? pcx.cctx.Bidx : Matrix{UInt32}(pcx.cctx.Bidx)
    fam2 = pcx.cctx.n_families == 2
    ncm_level_total = fam2 ? 2L : L
    ncm_cm_op = pcx.cctx.ncm - ncm_level_total
    level_targets_cdf = pcx.aug.level_targets[1:L]
    levelpow_targets = pcx.aug.level_targets[L+1:2L]
    st = CMFrechetLookupState(obj_op, pcx.cctx.NCORE, ncm_cm_op, ncm_level_total, L, D,
        pcx.cctx.origins, pcx.cctx.refIndex1, bins_u, pcx.cctx.R, level_targets_cdf;
        core_cf_ref = pcx.cctx.core_cf_ref, Pow = pcx.cctx.Pow, levelpow_targets = levelpow_targets)
    check("st.ncm_cm == ncm_cm (dense)", st.ncm_cm == ncm_cm)
    check("st.ncm_level == ncm_level (dense)", st.ncm_level == ncm_level)

    cf = pcx.cctx.core_cf_ref[]

    Random.seed!(20260806)
    for trial in 1:5
        x = 0.1 .* randn(n)
        # TRUE dense FG (independent ground truth). g has length n (g[1]=d f/d zeta, g[2:end]=d f/d lambda).
        g_dense = zeros(n)
        f_dense = obj_dense(x, g_dense)

        # operator FG (CMFrechetLookupState, this session's extension)
        g_op = zeros(n)
        f_op = st(x, g_op)

        rel_f = abs(f_dense - f_op) / max(abs(f_dense), abs(f_op), 1e-10)
        rel_g = norm(g_dense - g_op) / max(norm(g_dense), norm(g_op), 1e-10)
        check("trial=$trial f: dense=$f_dense op=$f_op rel=$rel_f < 1e-8", rel_f < 1e-8)
        check("trial=$trial g: ||dense-op||/||.|| = $rel_g < 1e-8", rel_g < 1e-8)
        if trial == 1
            # block-by-block localization: [1]=g1, [2:1+ncore1]=E, [next ncm_cm/2]=CM_cdf,
            # [next ncm_cm/2]=CM_pow, [next L]=level_cdf, [next L]=level_pow
            off = 1
            println("    g1: dense=", g_dense[1], " op=", g_op[1])
            off += 1
            gE_d = g_dense[off:off+ncore1-1]; gE_o = g_op[off:off+ncore1-1]
            println("    E block: ||d-o||/||d|| = ", norm(gE_d - gE_o) / max(norm(gE_d), 1e-10))
            off += ncore1
            half = ncm_cm ÷ 2
            gCcdf_d = g_dense[off:off+half-1]; gCcdf_o = g_op[off:off+half-1]
            println("    CM_cdf block: ||d-o||/||d|| = ", norm(gCcdf_d - gCcdf_o) / max(norm(gCcdf_d), 1e-10))
            off += half
            gCpow_d = g_dense[off:off+half-1]; gCpow_o = g_op[off:off+half-1]
            println("    CM_pow block: ||d-o||/||d|| = ", norm(gCpow_d - gCpow_o) / max(norm(gCpow_d), 1e-10))
            off += half
            gLcdf_d = g_dense[off:off+L-1]; gLcdf_o = g_op[off:off+L-1]
            println("    level_cdf block: ||d-o||/||d|| = ", norm(gLcdf_d - gLcdf_o) / max(norm(gLcdf_d), 1e-10),
                " d=", gLcdf_d[1:3], " o=", gLcdf_o[1:3])
            off += L
            gLpow_d = g_dense[off:off+L-1]; gLpow_o = g_op[off:off+L-1]
            println("    level_pow block: ||d-o||/||d|| = ", norm(gLpow_d - gLpow_o) / max(norm(gLpow_d), 1e-10),
                " d=", gLpow_d[1:3], " o=", gLpow_o[1:3])
        end

        # independent verifier (operator_verification.jl's own levelpow extension). g_lambda excludes
        # the zeta component (matches _verify_inner_solution_operator_cm_core's own return contract).
        zeta = x[1]; lambda = x[2:end]
        ov = verify_inner_solution_operator_cm_frechet!(zeta, lambda, cf, L, pcx.cctx.nO, pcx.cctx.origins,
            pcx.cctx.refIndex1, bins_u, pcx.cctx.R, pcx.aug.level_targets, obj_op, size(obj_op.U, 1);
            Pow = pcx.cctx.Pow)
        rel_f_ov = abs(f_dense - ov.f) / max(abs(f_dense), abs(ov.f), 1e-10)
        rel_g_ov = norm(g_dense[2:end] - ov.g_lambda) / max(norm(g_dense[2:end]), norm(ov.g_lambda), 1e-10)
        check("trial=$trial verifier f: dense=$f_dense verify=$(ov.f) rel=$rel_f_ov < 1e-8", rel_f_ov < 1e-8)
        check("trial=$trial verifier g: rel = $rel_g_ov < 1e-8", rel_g_ov < 1e-8)
    end
end

println(); println("="^90); println("TOTAL: $npass passed, $nfail failed"); println("="^90)
exit(nfail == 0 ? 0 : 1)
