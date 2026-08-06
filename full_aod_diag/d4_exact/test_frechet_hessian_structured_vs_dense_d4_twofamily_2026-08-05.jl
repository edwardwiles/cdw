# D=4 gate (2026-08-05, paired-basis-preconditioning pilot): common-Fréchet TWO-FAMILY
# (include_truncated_moment=true) structured (Architecture C) Hessian vs dense reference
# (Architecture A) Hessian, at the SAME (theta, x) point, both contrast modes. Direct two-family
# analogue of test_frechet_hessian_structured_vs_dense_d4.jl -- see that file's own header for the
# base methodology (unchanged here). NEW here: the 6 new/extended blocks from this task's own
# _fill_frechet_level_blocks! extension -- H_E,levelpow, H_CM(cdf),level [existing], H_CM(pow),level
# [item 1], H_CM(cdf),levelpow, H_CM(pow),levelpow, H_level,levelpow, H_levelpow,levelpow.
const _D4E = @__DIR__
include(joinpath(_D4E, "context.jl"))
for f in ["winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","threaded_cross_hessian.jl",
          "cm_production_bundle.jl","winner_pair_cross_hessian.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_config.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra
using Printf

npass = 0
nfail = 0
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

for contrasts in (:anchored, :orthonormal)
    println("== contrasts = $contrasts (two-family, include_truncated_moment=true) ==")

    # ---- dense (Architecture A) reference ----
    aug_dense = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = contrasts, include_truncated_moment = true)
    ctx_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    r_dense = evaluate_fullA(x_free0, ctx_dense; use_cache = false, warm = false)
    check("dense solve feasible", r_dense.inner_status in (0, -100, -101, -102, -103, -400, -401, -402))

    n = aug_dense.obj_cm.outer_constr_index
    hbuf_dense = zeros(div(n * (n + 1), 2))
    aug_dense.obj_cm(aug_dense.obj_cm.x, h = hbuf_dense)

    # ---- structured (Architecture C), independent KNITRO run, threaded bins (production default) ----
    cfg_struct = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :structured,
                           contrasts = contrasts, marginal_restriction = :common_frechet, cm_moment_families = 2)
    pcx_struct = build_cm_production_context_v2(ctx, CS, cfg_struct; L = L)
    θ_full0_s = CS.reconstruct_full(x_free0, pcx_struct.ctx_cm.m)
    H_save_s, x_s, nStatus_s, n_fg_s, n_hess_s = inner_loop_internal_archgeneric(
        pcx_struct.ctx_cm.obj, θ_full0_s; hess_cb_builder = pcx_struct.hess_cb_builder)
    check("structured solve feasible", nStatus_s in (0, -100, -101, -102, -103, -400, -401, -402))
    @printf("  dense kappa proxy: obj_cm.H_save=%.10f   structured H_save=%.10f  (nStatus_s=%d, n_hess=%d)\n",
            aug_dense.obj_cm.H_save, H_save_s, nStatus_s, n_hess_s)
    check("dense and structured solves agree (H_save diff < 1e-6)", abs(aug_dense.obj_cm.H_save - H_save_s) < 1e-6)

    # ---- CROSS-EVALUATION: structured Hessian formula at the DENSE solve's own (theta,x) point ----
    obj_s = pcx_struct.ctx_cm.obj
    cctx = pcx_struct.aug.cctx
    x_dense = aug_dense.obj_cm.x
    θ_dense = r_dense.θ_full
    K_s = zeros(size(ctx.U, 1))
    obj_s.moments!(K_s, CS.select_G_from_H(obj_s, obj_s.H), θ_dense, obj_s.U, obj_s)
    obj_s.H[:, 2] .= 1.0
    _archC_prep_for_hessian!(obj_s, x_dense)
    hbuf_struct = zeros(div(n * (n + 1), 2))
    hessian_cm_frechet_structured!(hbuf_struct, obj_s, cctx, pcx_struct.aug.level_targets)

    max_diff = maximum(abs.(hbuf_struct .- hbuf_dense))
    check("structured Hessian == dense Hessian at shared (theta,x) point (max abs diff < 1e-8)", max_diff < 1e-8)
    @printf("  max|H_structured - H_dense| = %.3e   (dense max|H|=%.3e)\n", max_diff, maximum(abs.(hbuf_dense)))

    # ---- also check the THREADED bin-table variant (hessian_cm_structured_v2!) agrees with dense ----
    hbuf_struct_v2 = zeros(div(n * (n + 1), 2))
    hessian_cm_frechet_structured_v2!(hbuf_struct_v2, obj_s, cctx, pcx_struct.aug.level_targets; threaded_bins = true, tls = cctx.tls)
    max_diff_v2 = maximum(abs.(hbuf_struct_v2 .- hbuf_dense))
    check("threaded structured Hessian == dense Hessian (max abs diff < 1e-8)", max_diff_v2 < 1e-8)
    @printf("  max|H_structured_v2(threaded) - H_dense| = %.3e\n", max_diff_v2)

    # ---- unpack and check the NEW blocks separately for a clearer failure signal ----
    NCORE = aug_dense.ncore
    ncm_cdf = aug_dense.ncm_cdf
    ncm_cm = aug_dense.ncm_cm   # = 2*ncm_cdf
    L1 = L
    Hfull_dense = Matrix{Float64}(undef, n, n)
    k = 1
    for i in 1:n, j in i:n
        Hfull_dense[i, j] = hbuf_dense[k]; Hfull_dense[j, i] = hbuf_dense[k]
        k += 1
    end
    Hfull_struct = Matrix{Float64}(undef, n, n)
    k = 1
    for i in 1:n, j in i:n
        Hfull_struct[i, j] = hbuf_struct[k]; Hfull_struct[j, i] = hbuf_struct[k]
        k += 1
    end
    cm_cdf_off = NCORE
    cm_pow_off = NCORE + ncm_cdf
    level_off = NCORE + ncm_cm
    level_pow_off = level_off + L1

    blocks = [
        ("H_EE",              1:NCORE,                      1:NCORE),
        ("H_E,levelpow",      1:NCORE,                       level_pow_off+1:level_pow_off+L1),
        ("H_CM(cdf),level",   cm_cdf_off+1:cm_cdf_off+ncm_cdf, level_off+1:level_off+L1),
        ("H_CM(pow),level",   cm_pow_off+1:cm_pow_off+ncm_cdf, level_off+1:level_off+L1),
        ("H_CM(cdf),levelpow",cm_cdf_off+1:cm_cdf_off+ncm_cdf, level_pow_off+1:level_pow_off+L1),
        ("H_CM(pow),levelpow",cm_pow_off+1:cm_pow_off+ncm_cdf, level_pow_off+1:level_pow_off+L1),
        ("H_level,levelpow",  level_off+1:level_off+L1,       level_pow_off+1:level_pow_off+L1),
        ("H_levelpow,levelpow",level_pow_off+1:level_pow_off+L1, level_pow_off+1:level_pow_off+L1),
    ]
    for (name, rows, cols) in blocks
        d = maximum(abs.(Hfull_struct[rows, cols] .- Hfull_dense[rows, cols]))
        check("$name block matches (NEW)", d < 1e-8)
        @printf("    %-24s max|diff| = %.3e\n", name, d)
    end
    println()
end

println("==================================================")
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
