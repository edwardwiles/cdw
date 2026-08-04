# fix/profiled-functional-readiness-closeout-2026-08-03, continuation of task §11/§9's scale
# progression: D20/real W=100,000 outer-gradient gate for flexible_CM and common_frechet -- the
# two REDUCED families whose native (A/gp-block) outer gradient had only ever been checked at D4.
#
# CORRECTED 2026-08-04 (first version killed after ~60 CPU-min): the first version of this file
# copied the D4 gates' `delta_dual_fixed_dual_noeta` FD-ground-truth helper verbatim, which calls
# `obj.moments!(K, G, theta_full, ctx.U, obj)` to materialize a full dense `G::Matrix{Float64}(W,
# obj.d-1)`. At D4 that is a tiny, harmless verification-only construct. At real D20/W=100,000 it
# is a ~100,000x381 (~300MB) dense matrix, rebuilt TWICE per coordinate across 361 coordinates x 2
# families -- exactly the pattern [[feedback-no-dense-reduced-ever-anywhere]] exists to prevent,
# even though it was "only for FD verification." Projected wall-clock was 1-2+ hours; killed after
# the user pointed this out directly.
#
# Fix: evaluate the fixed-dual objective through the SAME zero-dense operator machinery the real
# KNITRO inner solve already uses, instead of a hand-rolled dense alternative. `CMLookupState` (the
# object returned as `st` by `build_reduced_cm_operator_bundle`/`build_reduced_frechet_operator_
# bundle`, see profiled_reduced_lookup_kernels_2026-08-02.jl / profiled_reduced_frechet_lookup_
# kernels_2026-08-02.jl) is directly callable as `st(x, g)` -- this IS the exact FG functor KNITRO
# invokes on every inner-solve iteration (cm_lookup_kernels.jl's `(st::CMLookupState)(x, g)`),
# computing `f = sum(Psi!(dual_index!(st,x)))/M + zeta` via `economic_forward_into_arg0!`/
# `cm_forward_contribution!` -- genuinely zero dense G, matching the real production per-iteration
# cost. To evaluate the fixed-dual K(w) = -f at a perturbed outer point with (zeta*,lambda*) held
# fixed: rebuild the bundle/state at the perturbed theta (`build_..._operator_bundle` +
# `prime_operator!`, the exact same two calls `reduced_cm_base_state`/`reduced_frechet_base_state`
# make before their own KNITRO solve) then call `st(x_fixed, Float64[])` ONCE -- no dense G, no
# KNITRO re-optimization. NO_DENSE_G_COUNTERS is checked before/after the sweep to PROVE this, not
# just assert it (same discipline as test_flexcm_outer_gradient_zerodense_d4_2026-08-02.jl).
#
# SCOPE, second correction (same session): even zero-dense, the FIRST corrected version still
# looped over ALL 361 coordinates (722 rebuild-and-evaluate probes per family), estimated ~18min
# total -- disproportionate given this exact codebase's own established precedent
# (test_profiled_outer_gradient_gate_D20_W80000_2026-08-01.jl deliberately checks only an
# 11-coordinate REPRESENTATIVE SUBSET, with its own comment stating exhaustive per-coordinate
# checking "is not attempted here given real wall-clock cost"). The actual production quantity
# (the analytic gradient itself, via shared_family_outer_gradient) is computed ONCE per family in
# ~8.5s regardless -- the coordinate loop below is verification-only, and a systematic
# analytic-formula error would show up broadly across a spread subset, not hide in one specific
# untested coordinate. This version mirrors that gate's own representative_subset (gp + 10 spread
# A_free coords, fixed seed) instead of the exhaustive sweep.
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

"Zero-dense fixed-dual K(w) for flexible_CM: rebuild the operator bundle/state at the perturbed
theta (the SAME two calls reduced_cm_base_state makes before its own KNITRO solve), then evaluate
the real FG functor ONCE at the fixed (zeta*,lambda*) -- no dense G, no re-optimization."
function delta_dual_fixed_dual_zerodense_cm(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, x_fixed::Vector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_cm_operator_bundle(ctx, θ_full0, layout, cctx; method = :suffix)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    f = st(x_fixed, Float64[])
    return -f
end

"Same for common_frechet (level_targets threaded through, matching reduced_frechet_base_state)."
function delta_dual_fixed_dual_zerodense_frechet(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, level_targets::Vector{Float64}, x_fixed::Vector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_frechet_operator_bundle(ctx, θ_full0, layout, cctx, level_targets)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    f = st(x_fixed, Float64[])
    return -f
end

"Representative coordinate subset, SAME rationale/construction as
test_profiled_outer_gradient_gate_D20_W80000_2026-08-01.jl's own representative_subset: gp(1) +
10 spread A_free coordinates via a fixed seed -- that gate's own comment states exhaustive
per-coordinate checking is deliberately not attempted given real wall-clock cost, and a systematic
analytic-formula error would show up broadly across a spread subset, not hide in one specific
untested coordinate. Reusing that established, sanctioned scope here rather than the exhaustive
361-coordinate sweep the first version of this file wrongly defaulted to."
function representative_subset(n_total::Int; n_random::Int = 10, seed::Int = 7)
    Random.seed!(seed)
    subset = Set{Int}([1])
    for k in 1:n_random
        push!(subset, rand(2:n_total))
    end
    return sort(collect(subset))
end

function gate_family(label::String, w_profiled_calib::Vector{Float64}, ctx, pe, fctx, ev, fd_eval::Function; h_gp = 1e-4)
    println("\n" * "="^90); println("D20/W100000: $label outer-gradient gate (zero-dense fixed-dual FD, representative subset)"); println("="^90); flush(stdout)
    n_total = length(w_profiled_calib)
    t1 = time()
    g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev; threaded = false)
    @printf("analytic gradient computed in %.2fs, gp_component=%.6e, norm(A-block)=%.6e\n", time() - t1, g_Agp[1], norm(g_Agp[2:end])); flush(stdout)

    coords = representative_subset(n_total)
    println("representative subset (", length(coords), " coords): ", coords); flush(stdout)
    t2 = time()
    max_rel_err = 0.0; worst = (coord = 0, rel_err = 0.0)
    rows = NamedTuple[]
    for coord in coords
        h_c = coord == 1 ? h_gp : meta.h_used[coord]
        tprobe = time()
        wp = copy(w_profiled_calib); wp[coord] += h_c
        decoded_p = decode_outer_profiled(wp, ctx, pe)
        Kp = fd_eval(decoded_p.xf)
        wm = copy(w_profiled_calib); wm[coord] -= h_c
        decoded_m = decode_outer_profiled(wm, ctx, pe)
        Km = fd_eval(decoded_m.xf)
        fd = (Kp - Km) / (2h_c)
        err = abs(fd - g_Agp[coord])
        relerr = err / max(1.0, abs(fd))
        if relerr > max_rel_err
            max_rel_err = relerr; worst = (coord = coord, rel_err = relerr)
        end
        push!(rows, (coord = coord, analytic = g_Agp[coord], fd = fd, h = h_c, abs_err = err, rel_err = relerr))
        @printf("  coord=%-4d analytic=%+.4e  fd=%+.4e  rel_err=%.3e  (%.2fs/probe)\n",
            coord, g_Agp[coord], fd, relerr, time() - tprobe)
        flush(stdout)
    end
    @printf("fixed-dual FD over %d/%d representative coordinates computed in %.2fs\n", length(coords), n_total, time() - t2); flush(stdout)
    @printf("max_rel_err = %.4e  at coord=%d (%s)\n", max_rel_err, worst.coord, worst.coord == 1 ? "gp" : "A_free[$(worst.coord-1)]")
    gp_row = rows[1]
    @printf("gp coordinate specifically: analytic=%+.6e  FD(h=%.2e)=%+.6e  rel_err=%.4e\n", gp_row.analytic, gp_row.h, gp_row.fd, gp_row.rel_err)
    n_bad = count(r -> r.rel_err > 5e-3, rows)
    println("coordinates with rel_err > 5e-3: $n_bad / $(length(coords))")
    flush(stdout)
    return (label = label, n_total = n_total, n_done = length(coords), max_rel_err = max_rel_err, worst = worst, gp_row = gp_row, n_bad = n_bad, t_fd = time() - t2)
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
cctx_cm = build_cm_bin_ctx(ctx, aug_reduced_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_cm)
t_solve = @elapsed ev_cm = evaluate_profiled_flexcm_point(w_profiled_calib, fctx_cm)
@printf("flexible_CM cold solve: %.2fs  nStatus=%d\n", t_solve, ev_cm.result.inner_status); flush(stdout)
x_fixed_cm = vcat(ev_cm.result.zeta, ev_cm.result.beta)

before_cm = deepcopy(NO_DENSE_G_COUNTERS[])
res_cm = gate_family("flexible_CM", w_profiled_calib, ctx, pe, fctx_cm, ev_cm,
    xf -> delta_dual_fixed_dual_zerodense_cm(xf, ctx, layout, cctx_cm, x_fixed_cm))
after_cm = NO_DENSE_G_COUNTERS[]
d_econ_cm = after_cm.dense_economic_G_materializations - before_cm.dense_economic_G_materializations
d_cm_cm = after_cm.dense_CM_G_materializations - before_cm.dense_CM_G_materializations
@printf("flexible_CM: dense G materializations across whole FD sweep: economic=%d  CM-grid=%d  (expect 0, 0)\n", d_econ_cm, d_cm_cm)
push!(results, res_cm)

println("\n" * "#"^90); println("FAMILY: common_frechet"); println("#"^90); flush(stdout)
aug_reduced_fr = build_cm_frechet_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_fr.level_targets
cctx_fr = build_cm_bin_ctx(ctx, aug_reduced_fr; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx_fr = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_fr, level_targets)
t_solve_fr = @elapsed ev_fr = evaluate_profiled_frechet_point(w_profiled_calib, fctx_fr)
@printf("common_frechet cold solve: %.2fs  nStatus=%d\n", t_solve_fr, ev_fr.result.inner_status); flush(stdout)
x_fixed_fr = vcat(ev_fr.result.zeta, ev_fr.result.beta)

before_fr = deepcopy(NO_DENSE_G_COUNTERS[])
res_fr = gate_family("common_frechet", w_profiled_calib, ctx, pe, fctx_fr, ev_fr,
    xf -> delta_dual_fixed_dual_zerodense_frechet(xf, ctx, layout, cctx_fr, level_targets, x_fixed_fr))
after_fr = NO_DENSE_G_COUNTERS[]
d_econ_fr = after_fr.dense_economic_G_materializations - before_fr.dense_economic_G_materializations
d_cm_fr = after_fr.dense_CM_G_materializations - before_fr.dense_CM_G_materializations
@printf("common_frechet: dense G materializations across whole FD sweep: economic=%d  CM-grid=%d  (expect 0, 0)\n", d_econ_fr, d_cm_fr)
push!(results, res_fr)

println("\n" * "="^90); println("SUMMARY"); println("="^90)
for r in results
    @printf("%-16s max_rel_err=%.4e (coord=%d)  gp_rel_err=%.4e  n_bad(>5e-3)=%d/%d  n_done=%d/%d\n",
        r.label, r.max_rel_err, r.worst.coord, r.gp_row.rel_err, r.n_bad, r.n_done, r.n_done, r.n_total)
end
all_pass = all(r.max_rel_err < 5e-3 for r in results)
println("\nD20/W$(W_VAL) FLEXCM+FRECHET OUTER-GRADIENT GATE: ", all_pass ? "PASS" : "NEEDS REVIEW")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
