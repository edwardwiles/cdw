# D=20 real-data gate (2026-08-06, paired-basis-preconditioning pilot continuation): common-Fréchet
# TWO-FAMILY (include_truncated_moment=true) structured (Architecture C) Hessian vs dense reference
# (Architecture A) Hessian, at the SAME (theta, x) point, real D=20/W=80,000 draws, both contrast
# modes. Direct D20 analogue of test_frechet_hessian_structured_vs_dense_d4_twofamily_2026-08-05.jl
# (which PASSED 26/26 at D4) -- same methodology, real data instead of the D4 synthetic fixture.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","threaded_cross_hessian.jl",
          "cm_production_bundle.jl","cm_screen_bridge.jl","winner_pair_cross_hessian.jl",
          "gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

W = 80_000
L = 10
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row, inner_lower_limit = -10.0)
D = ctx.D
x_free0 = ctx.θ0_up[ctx.free_idx]

println("="^90); println("D=20 real-data two-family common-Fréchet Hessian gate (W=$W, L=$L)"); println("="^90)

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
    check("structured Hessian == dense Hessian at shared (theta,x) point (max abs diff < 1e-6)", max_diff < 1e-6)
    @printf("  max|H_structured - H_dense| = %.3e   (dense max|H|=%.3e)\n", max_diff, maximum(abs.(hbuf_dense)))

    hbuf_struct_v2 = zeros(div(n * (n + 1), 2))
    hessian_cm_frechet_structured_v2!(hbuf_struct_v2, obj_s, cctx, pcx_struct.aug.level_targets; threaded_bins = true, tls = cctx.tls)
    max_diff_v2 = maximum(abs.(hbuf_struct_v2 .- hbuf_dense))
    check("threaded structured Hessian == dense Hessian (max abs diff < 1e-6)", max_diff_v2 < 1e-6)
    @printf("  max|H_structured_v2(threaded) - H_dense| = %.3e\n", max_diff_v2)

    NCORE = aug_dense.ncore
    ncm_cdf = aug_dense.ncm_cdf
    ncm_cm = aug_dense.ncm_cm
    cm_cdf_off = NCORE
    cm_pow_off = NCORE + ncm_cdf
    level_off = NCORE + ncm_cm
    level_pow_off = level_off + L
    n_full = n
    Hfull_dense = Matrix{Float64}(undef, n_full, n_full)
    k = 1
    for i in 1:n_full, j in i:n_full
        Hfull_dense[i, j] = hbuf_dense[k]; Hfull_dense[j, i] = hbuf_dense[k]
        k += 1
    end
    Hfull_struct = Matrix{Float64}(undef, n_full, n_full)
    k = 1
    for i in 1:n_full, j in i:n_full
        Hfull_struct[i, j] = hbuf_struct[k]; Hfull_struct[j, i] = hbuf_struct[k]
        k += 1
    end
    blocks = [
        ("H_EE",              1:NCORE,                      1:NCORE),
        ("H_E,levelpow",      1:NCORE,                       level_pow_off+1:level_pow_off+L),
        ("H_CM(cdf),level",   cm_cdf_off+1:cm_cdf_off+ncm_cdf, level_off+1:level_off+L),
        ("H_CM(pow),level",   cm_pow_off+1:cm_pow_off+ncm_cdf, level_off+1:level_off+L),
        ("H_CM(cdf),levelpow",cm_cdf_off+1:cm_cdf_off+ncm_cdf, level_pow_off+1:level_pow_off+L),
        ("H_CM(pow),levelpow",cm_pow_off+1:cm_pow_off+ncm_cdf, level_pow_off+1:level_pow_off+L),
        ("H_level,levelpow",  level_off+1:level_off+L,       level_pow_off+1:level_pow_off+L),
        ("H_levelpow,levelpow",level_pow_off+1:level_pow_off+L, level_pow_off+1:level_pow_off+L),
    ]
    for (name, rows, cols) in blocks
        d = maximum(abs.(Hfull_struct[rows, cols] .- Hfull_dense[rows, cols]))
        check("$name block matches (NEW)", d < 1e-6)
        @printf("    %-24s max|diff| = %.3e\n", name, d)
    end
    println()
end

println("==================================================")
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
