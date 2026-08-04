# fix/profiled-functional-readiness-closeout-2026-08-03, continuation of task §11/§9's scale
# progression: D20/real W=100,000 outer-gradient gate for flexible_CM and common_frechet -- the
# two REDUCED families whose native (A/gp-block) outer gradient had only ever been checked at D4
# (test_flexcm_outer_gradient_zerodense_d4_2026-08-02.jl / test_frechet_outer_gradient_zerodense_
# d4_2026-08-02.jl), unlike origin_ZC/CM_plus_ZC which already have real D20/W20000/W100000
# free-eta gates (test_zc_free_eta_evaluator_d20_2026-08-04.jl).
#
# Deliberately uses the SAME fixed-dual FD methodology as the D4 flexCM/frechet gates
# (delta_dual_fixed_dual_noeta: hold (zeta*,lambda*) fixed at the real converged solve, rebuild G
# fresh via obj.moments!, no re-optimization) -- NOT the full-resolve FD used by
# test_profiled_outer_gradient_gate_D20_W80000_2026-08-01.jl (the unrestricted-family D20 gate
# that found the still-open gp-coordinate discrepancy). That choice is deliberate: fixed-dual FD is
# the direct, cheap (no extra KNITRO solves) way to check an envelope-theorem analytic gradient
# formula against its own defining derivative, and is what already gave clean ~1e-10 results for
# origin_ZC/CM_plus_ZC's eta-gradient gates this session. It does NOT by itself investigate the
# unrestricted gp anomaly (that is being investigated separately, in parallel, via
# investigate_gp_fd_bandwidth_2026-08-04.jl's h-sweep against the full-resolve FD).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "reduced_restricted_family_verification_2026-08-03.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

t0 = time()
println("PID=", getpid()); flush(stdout)

"Same fixed-dual FD helper as the D4 flexCM/frechet gates (no re-optimization; rebuild G via
obj.moments! at fixed (zeta*,lambda*))."
function delta_dual_fixed_dual_noeta(obj, ζstar::Float64, λstar::Vector{Float64}, w_profiled::Vector{Float64},
        ctx, pe, W::Int)
    decoded = decode_outer_profiled(w_profiled, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_full, ctx.U, obj)
    r = fill(-ζstar, W) .- G * λstar
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζstar
    return -f
end

function gate_family(label::String, w_profiled_calib::Vector{Float64}, ctx, pe, fctx, ev, obj_dense::Any, W::Int; h_gp = 1e-4)
    println("\n" * "="^90); println("D20/W$(W): $label outer-gradient gate (fixed-dual FD, ALL coordinates)"); println("="^90); flush(stdout)
    n_total = length(w_profiled_calib)
    t1 = time()
    g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev)
    @printf("analytic gradient computed in %.2fs, gp_component=%.6e, norm(A-block)=%.6e\n", time() - t1, g_Agp[1], norm(g_Agp[2:end])); flush(stdout)

    t2 = time()
    max_rel_err = 0.0; worst = (coord = 0, rel_err = 0.0)
    rows = NamedTuple[]
    for coord in 1:n_total
        h_c = coord == 1 ? h_gp : meta.h_used[coord]
        wp = copy(w_profiled_calib); wp[coord] += h_c
        Kp = delta_dual_fixed_dual_noeta(obj_dense, ev.result.zeta, ev.result.beta, wp, ctx, pe, W)
        wm = copy(w_profiled_calib); wm[coord] -= h_c
        Km = delta_dual_fixed_dual_noeta(obj_dense, ev.result.zeta, ev.result.beta, wm, ctx, pe, W)
        fd = (Kp - Km) / (2h_c)
        err = abs(fd - g_Agp[coord])
        relerr = err / max(1.0, abs(fd))
        if relerr > max_rel_err
            max_rel_err = relerr; worst = (coord = coord, rel_err = relerr)
        end
        push!(rows, (coord = coord, analytic = g_Agp[coord], fd = fd, h = h_c, abs_err = err, rel_err = relerr))
    end
    @printf("fixed-dual FD over all %d coordinates computed in %.2fs\n", n_total, time() - t2); flush(stdout)
    @printf("max_rel_err = %.4e  at coord=%d (%s)\n", max_rel_err, worst.coord, worst.coord == 1 ? "gp" : "A_free[$(worst.coord-1)]")
    gp_row = rows[1]
    @printf("gp coordinate specifically: analytic=%+.6e  FD(h=%.2e)=%+.6e  rel_err=%.4e\n", gp_row.analytic, gp_row.h, gp_row.fd, gp_row.rel_err)
    n_bad = count(r -> r.rel_err > 5e-3, rows)
    println("coordinates with rel_err > 5e-3: $n_bad / $n_total")
    flush(stdout)
    return (label = label, n_total = n_total, max_rel_err = max_rel_err, worst = worst, gp_row = gp_row, n_bad = n_bad, t_fd = time() - t2)
end

W_VAL = 100_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
w_profiled_calib = reduce_to_w_profiled(θ0[3+D], z_calib, pe)
decoded_calib = decode_outer_profiled(w_profiled_calib, ctx, pe)
n_total = outer_dim_profiled(pe)
@printf("outer_dim_profiled = %d (1 gp + %d free-A)\n", n_total, n_total - 1); flush(stdout)

results = NamedTuple[]

println("\n" * "#"^90); println("FAMILY: flexible_CM"); println("#"^90); flush(stdout)
aug_reduced_cm = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
obj_dense_cm = aug_reduced_cm.obj_cm
cctx_cm = build_cm_bin_ctx(ctx, aug_reduced_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_cm)
t_solve = @elapsed ev_cm = evaluate_profiled_flexcm_point(w_profiled_calib, fctx_cm)
@printf("flexible_CM cold solve: %.2fs  nStatus=%d\n", t_solve, ev_cm.result.inner_status); flush(stdout)
push!(results, gate_family("flexible_CM", w_profiled_calib, ctx, pe, fctx_cm, ev_cm, obj_dense_cm, W_VAL))

println("\n" * "#"^90); println("FAMILY: common_frechet"); println("#"^90); flush(stdout)
aug_reduced_fr = build_cm_frechet_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_fr.level_targets
obj_dense_fr = aug_reduced_fr.obj_cm
cctx_fr = build_cm_bin_ctx(ctx, aug_reduced_fr; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx_fr = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_fr, level_targets)
t_solve_fr = @elapsed ev_fr = evaluate_profiled_frechet_point(w_profiled_calib, fctx_fr)
@printf("common_frechet cold solve: %.2fs  nStatus=%d\n", t_solve_fr, ev_fr.result.inner_status); flush(stdout)
push!(results, gate_family("common_frechet", w_profiled_calib, ctx, pe, fctx_fr, ev_fr, obj_dense_fr, W_VAL))

println("\n" * "="^90); println("SUMMARY"); println("="^90)
for r in results
    @printf("%-16s max_rel_err=%.4e (coord=%d)  gp_rel_err=%.4e  n_bad(>5e-3)=%d/%d\n",
        r.label, r.max_rel_err, r.worst.coord, r.gp_row.rel_err, r.n_bad, r.n_total)
end
all_pass = all(r.max_rel_err < 5e-3 for r in results)
println("\nD20/W$(W_VAL) FLEXCM+FRECHET OUTER-GRADIENT GATE: ", all_pass ? "PASS" : "NEEDS REVIEW")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
