# D=4 Part III gate: structured (Architecture C, winner-pair H_EE + level-block extension)
# Hessian vs dense reference (Architecture A) Hessian, at the SAME (theta, x) point, both contrast
# modes. This is the core Part III correctness gate -- the level Hessian blocks (H_E,level,
# H_CM,level, H_level,level) are new code (cm_frechet_hessian.jl); everything else in the callback
# (build_bin_tables!, prefix_sum_tables!, _fill_cm_HEE!, the H_EC/H_CC loops) is copied UNCHANGED
# from hessian_cm_structured!.
const _D4E = @__DIR__
include(joinpath(_D4E, "context.jl"))
for f in ["winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl",
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
    println("== contrasts = $contrasts ==")

    # ---- dense (Architecture A) solve ----
    aug_dense = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = contrasts, include_truncated_moment = false)
    ctx_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    r_dense = evaluate_fullA(x_free0, ctx_dense; use_cache = false, warm = false)
    check("dense solve feasible", r_dense.inner_status in (0, -100, -101, -102, -103, -400, -401, -402))

    n = aug_dense.obj_cm.outer_constr_index
    hbuf_dense = zeros(div(n * (n + 1), 2))
    aug_dense.obj_cm(aug_dense.obj_cm.x, h = hbuf_dense)

    # ---- structured (Architecture C) solve, independent KNITRO run. Use inner_loop_internal_archgeneric
    # directly rather than cm_base_state_v2 (trusted/unmodified production code, hardcoded to the
    # narrow (0,-100,-101,-103) set) -- at this SMALL diagnostic L=10/D=4 scale the extra L level
    # moments can push KNITRO past the default maxit before full convergence while still reaching a
    # genuinely feasible point (-400 = KN_RC_ITER_LIMIT_FEAS, is_feasible_result=true per
    # knitro_status.jl) -- this is a maxit-budget artifact of the small test, not evidence either way
    # about Hessian correctness, which the direct block-by-block comparison below settles independently. ----
    cfg_struct = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :structured,
                           contrasts = contrasts, marginal_restriction = :common_frechet, cm_moment_families = 1)
    pcx_struct = build_cm_production_context_v2(ctx, CS, cfg_struct; L = L)
    θ_full0_s = CS.reconstruct_full(x_free0, pcx_struct.ctx_cm.m)
    H_save_s, x_s, nStatus_s, n_fg_s, n_hess_s = inner_loop_internal_archgeneric(
        pcx_struct.ctx_cm.obj, θ_full0_s; hess_cb_builder = pcx_struct.hess_cb_builder)
    check("structured solve feasible", nStatus_s in (0, -100, -101, -102, -103, -400, -401, -402))
    @printf("  dense kappa proxy: obj_cm.H_save=%.10f   structured H_save=%.10f  (nStatus_s=%d, n_hess=%d)\n",
            aug_dense.obj_cm.H_save, H_save_s, nStatus_s, n_hess_s)
    check("dense and structured solves agree (H_save diff < 1e-6)", abs(aug_dense.obj_cm.H_save - H_save_s) < 1e-6)

    # ---- CROSS-EVALUATION: evaluate the structured Hessian formula at the DENSE solve's own
    # (theta, x) point (both objs built from identical ctx/L/contrasts -- Part II already validated
    # their moments! outputs agree to ~1e-15, so this is a genuine independent structured-vs-dense
    # Hessian check, not "structured matches itself"). ----
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
    rel_scale = max(maximum(abs.(hbuf_dense)), 1.0)
    check("structured Hessian == dense Hessian at shared (theta,x) point (max abs diff < 1e-8)", max_diff < 1e-8)
    @printf("  max|H_structured - H_dense| = %.3e   (dense max|H|=%.3e)\n", max_diff, maximum(abs.(hbuf_dense)))

    # ---- sanity: unpack and check specific blocks separately for a clearer failure signal ----
    NCORE = aug_dense.ncore
    ncm_cm = aug_dense.ncm_cm
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
    level_off = NCORE + ncm_cm
    max_HEE = maximum(abs.(Hfull_struct[1:NCORE,1:NCORE] .- Hfull_dense[1:NCORE,1:NCORE]))
    max_HEC = maximum(abs.(Hfull_struct[1:NCORE, NCORE+1:level_off] .- Hfull_dense[1:NCORE, NCORE+1:level_off]))
    max_HCC = maximum(abs.(Hfull_struct[NCORE+1:level_off, NCORE+1:level_off] .- Hfull_dense[NCORE+1:level_off, NCORE+1:level_off]))
    max_Elevel = maximum(abs.(Hfull_struct[1:NCORE, level_off+1:end] .- Hfull_dense[1:NCORE, level_off+1:end]))
    max_CMlevel = maximum(abs.(Hfull_struct[NCORE+1:level_off, level_off+1:end] .- Hfull_dense[NCORE+1:level_off, level_off+1:end]))
    max_levellevel = maximum(abs.(Hfull_struct[level_off+1:end, level_off+1:end] .- Hfull_dense[level_off+1:end, level_off+1:end]))
    @printf("  block diffs: HEE=%.3e HEC=%.3e HCC=%.3e H_E,level=%.3e H_CM,level=%.3e H_level,level=%.3e\n",
            max_HEE, max_HEC, max_HCC, max_Elevel, max_CMlevel, max_levellevel)
    check("H_EE block matches", max_HEE < 1e-8)
    check("H_EC block matches", max_HEC < 1e-8)
    check("H_CC block matches", max_HCC < 1e-8)
    check("H_E,level block matches (NEW)", max_Elevel < 1e-8)
    check("H_CM,level block matches (NEW)", max_CMlevel < 1e-8)
    check("H_level,level block matches (NEW)", max_levellevel < 1e-8)
    println()
end

println("==================================================")
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
