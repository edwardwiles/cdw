# Part VI: D=20 real-production-scale gates for fixed-Frechet-as-CM-plus-anchor.
#   1. Basis-equivalence (task Sec 16): direct country-by-country q_l vs CM-plus-level
#      construction, at real D=20/W=80,000 draws.
#   2. Flexible-CM-vs-common-Frechet nesting diagnostic (task Sec 17): moment dims,
#      Hessian-callback structure, allocations, at the SAME calibration point.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf
println("=== includes OK ==="); flush(stdout)

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

W = 80000
L = 10
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx.D
x_free0 = ctx.θ0_up[ctx.free_idx]
probs = cm_equal_grid_probs(L)

println("="^90); println("PART 1: BASIS EQUIVALENCE at real D=$D, W=$W"); println("="^90)
for contrasts in (:anchored, :orthonormal)
    println("-- contrasts=$contrasts --")
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, ctx.γ.refIndex1, L; contrasts = contrasts, probs = probs)
    level_targets = frechet_level_targets(D, L; probs = probs)
    LEVEL = precalc_frechet_level_dense(ctx.U, z, D, level_targets)
    level_probs = frechet_level_probs(L; probs = probs)

    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    C = zeros(D, D - 1)
    oi = 0
    for o in 1:D
        o == ctx.γ.refIndex1 && continue
        oi += 1
        C[o, oi] = 1.0
        C[ctx.γ.refIndex1, oi] = -1.0
    end
    Mmat = R === nothing ? hcat(C, ones(D) / sqrt(D)) : hcat(C * R, ones(D) / sqrt(D))
    check("M = [C(R) u] is $(D)x$(D)", size(Mmat) == (D, D))
    check("M is full rank ($D)", rank(Mmat) == D)
    condM = cond(Mmat)
    @printf("  cond(M) = %.4f  (theory: %s)\n", condM, contrasts == :orthonormal ? "1.0" : "sqrt(D)=$(sqrt(D))")
    check("cond(M) matches theory", contrasts == :orthonormal ? isapprox(condM, 1.0; atol=1e-6) : isapprox(condM, sqrt(D); atol=1e-4))

    Minv = inv(Mmat)
    max_cm_err = 0.0; max_level_err = 0.0; max_roundtrip_err = 0.0
    nl_check = min(L, 5)   # spot-check a subset of thresholds at this W (full loop is O(W*D*L), still cheap but keep runtime bounded)
    for l in 1:nl_check
        fl = Float64.(ctx.U .<= z[l])
        ql = fl .- level_probs[l]
        cm_cols = (l - 1) * (D - 1) + 1 : l * (D - 1)
        cm_direct = ql * (R === nothing ? C : C * R)
        max_cm_err = max(max_cm_err, maximum(abs.(cm_direct .- CM[:, cm_cols])))
        level_direct = ql * (ones(D) / sqrt(D))
        max_level_err = max(max_level_err, maximum(abs.(level_direct .- LEVEL[:, l])))
        stacked = hcat(cm_direct, level_direct)
        reconstructed = stacked * Minv
        max_roundtrip_err = max(max_roundtrip_err, maximum(abs.(reconstructed .- ql)))
    end
    @printf("  max_cm_err=%.3e max_level_err=%.3e max_roundtrip_err=%.3e (spot-checked %d/%d thresholds)\n",
            max_cm_err, max_level_err, max_roundtrip_err, nl_check, L)
    check("basis equivalence holds at real D=20 draws (< 1e-8)", max(max_cm_err, max_level_err, max_roundtrip_err) < 1e-8)
end

println()
println("="^90); println("PART 2: FLEXIBLE-CM vs COMMON-FRECHET NESTING DIAGNOSTIC"); println("="^90)
cfg_flex = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :structured,
                     contrasts = :anchored, marginal_restriction = :common_flexible)
cfg_frechet = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :structured,
                        contrasts = :anchored, marginal_restriction = :common_frechet)

t0 = time()
# build_cm_production_context_v2's plain-CM branch doesn't forward bins/cctx (unlike the
# frechet branch, which I added those to) -- call build_cm_production_context directly here to
# get a cctx-bearing pcx, matching how run_cm_upper_checkpointed itself does it.
pcx_flex = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs, threaded_bins = true)
t_ctx_flex = time() - t0
t0 = time()
pcx_frechet = build_cm_production_context_v2(ctx, CS, cfg_frechet; L = L)
t_ctx_frechet = time() - t0

println("moment dims: flexible ncm=", pcx_flex.aug.ncm, "  frechet ncm=", pcx_frechet.aug.ncm,
        "  diff=", pcx_frechet.aug.ncm - pcx_flex.aug.ncm, " (expect L=$L)")
check("frechet has exactly L more moments than flexible", pcx_frechet.aug.ncm - pcx_flex.aug.ncm == L)
@printf("context-construction wall: flexible=%.3fs  frechet=%.3fs  ratio=%.2fx\n", t_ctx_flex, t_ctx_frechet, t_ctx_frechet/t_ctx_flex)

θ_full0 = CS.reconstruct_full(x_free0, pcx_flex.ctx_cm.m)
hess_cb_builder_flex = _obj -> archC_hess_cb_builder(pcx_flex.cctx)   # build_cm_production_context (direct call) doesn't return hess_cb_builder itself, unlike the _v2 wrapper

# complete inner-solve timing (real KNITRO), same point, both families
t0 = time()
K_flex, x_flex, ns_flex, nfg_flex, nh_flex = inner_loop_internal_archgeneric(pcx_flex.ctx_cm.obj, θ_full0; hess_cb_builder = hess_cb_builder_flex)
t_solve_flex = time() - t0
check("flexible CM inner solve feasible", ns_flex in (0, -100, -101, -102, -103, -400, -401, -402))

t0 = time()
K_frechet, x_frechet, ns_frechet, nfg_frechet, nh_frechet = inner_loop_internal_archgeneric(pcx_frechet.ctx_cm.obj, θ_full0; hess_cb_builder = pcx_frechet.hess_cb_builder)
t_solve_frechet = time() - t0
check("common Frechet inner solve feasible", ns_frechet in (0, -100, -101, -102, -103, -400, -401, -402))

@printf("complete inner-solve wall: flexible=%.3fs (n_fg=%d n_hess=%d)  frechet=%.3fs (n_fg=%d n_hess=%d)  ratio=%.2fx\n",
        t_solve_flex, nfg_flex, nh_flex, t_solve_frechet, nfg_frechet, nh_frechet, t_solve_frechet/t_solve_flex)

# per-Hessian-callback allocation (single call, isolated)
_archC_prep_for_hessian!(pcx_flex.ctx_cm.obj, x_flex)
n_flex = pcx_flex.cctx.NCORE + pcx_flex.cctx.ncm
hbuf_flex = zeros(div(n_flex * (n_flex + 1), 2))
alloc_flex = @allocated hessian_cm_structured!(hbuf_flex, pcx_flex.ctx_cm.obj, pcx_flex.cctx)

_archC_prep_for_hessian!(pcx_frechet.ctx_cm.obj, x_frechet)
n_frechet = pcx_frechet.cctx.NCORE + pcx_frechet.cctx.ncm
hbuf_frechet = zeros(div(n_frechet * (n_frechet + 1), 2))
alloc_frechet = @allocated hessian_cm_frechet_structured!(hbuf_frechet, pcx_frechet.ctx_cm.obj, pcx_frechet.cctx, pcx_frechet.aug.level_targets)

@printf("Hessian-callback allocation: flexible=%d bytes  frechet=%d bytes  diff=%d bytes (%.1f%% of flexible)\n",
        alloc_flex, alloc_frechet, alloc_frechet - alloc_flex, 100*(alloc_frechet-alloc_flex)/alloc_flex)
check("frechet Hessian callback allocation not wildly larger (< 3x flexible)", alloc_frechet < 3 * alloc_flex)

println()
println("="^90)
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
