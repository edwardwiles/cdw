# fix/profiled-functional-readiness-closeout-2026-08-03, task §8/§9/§10: D4 native-coordinate
# outer-gradient gate for the NEW free-eta_nu evaluator surface itself
# (profiled_zc_free_eta_2026-08-04.jl) -- distinct from the pre-existing
# test_zc_lane_{originzc,cmzc}_outer_gradient_d4_2026-08-02.jl gates, which validate the underlying
# math (shared_family_outer_gradient + d_delta_dual_d_eta_origin_vec) against a hand-built `ev`
# NamedTuple, not the production evaluate_profiled_{originzc,cmzc}_point(w_econ, eta_nu, fctx, pes)
# call surface a real free-nu outer runner would actually call. This test exercises THAT surface
# end to end: real KNITRO inner solve via the eta-explicit evaluator, then
# reduced_{originzc,cmzc}_outer_gradient_with_eta, gated against complete fixed-dual finite
# differences, for gp / A_free / eta_nu coordinates, at calibration AND a perturbed point, for BOTH
# ZC families -- covering task §10's required coordinate-block breakdown at D4.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
          "cm_exact_cache_production.jl", "cm_checkpoint.jl",
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
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl",
          "profiled_zc_free_eta_2026-08-04.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function build_profiled_ab_spec_pe(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    return spec, gauge, pe
end
function reduce_calibration_to_w_profiled(ctx, pe::PivotGravityElimOnRetained)
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gp0 = θ0[3+D]
    return reduce_to_w_profiled(gp0, z_calib, pe)
end

"Fixed-dual Delta_dual at a perturbed (w_econ, eta_nu), zeta*/lambda* held FIXED. Same convention
as the 2026-08-02 ZC-lane gates' own helper of the same name."
function delta_dual_fixed_dual(obj, ζstar::Float64, λstar::Vector{Float64}, w_econ::Vector{Float64},
        eta_nu::Vector{Float64}, ctx, pe, W::Int)
    decoded = decode_outer_profiled(w_econ, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    θ_ext = vcat(θ_full, exp.(eta_nu))
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_ext, ctx.U, obj)
    r = fill(-ζstar, W) .- G * λstar
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζstar
    return -f
end

"Gate ONE family's evaluator (already solved `ev`) against fixed-dual FD, at ALL econ coords + ALL
eta coords, using ONLY the real evaluate/gradient-with-eta call surface. `h_used` (from
`shared_family_outer_gradient`'s own `meta.h_used`, PER-COORDINATE adaptive bandwidth for A_free
coords, chosen to land cleanly around winner switches) is REQUIRED for coord>1 -- comparing
against a different, fixed h reproduces this codebase's own documented false-mismatch pattern
(feedback-fd-bandwidth-mismatch-looks-like-a-bug). gp (coord 1) has no bandwidth search (a
genuinely analytic closed-form derivative), so a small fixed h is fine there, matching every other
gate in this codebase."
function gate_family(label::String, obj, w_econ::Vector{Float64}, eta_nu::Vector{Float64}, ev,
        g_ext::Vector{Float64}, ctx, pe, W::Int, h_used::Vector{Float64}; h_gp = 1e-4, h_eta = 1e-4, tol = 5e-3)
    n_econ = length(w_econ); n_eta_d = length(eta_nu)
    length(g_ext) == n_econ + n_eta_d || error("gate_family: g_ext length mismatch")
    ζstar = ev.result.zeta; λstar = ev.result.beta
    max_rel = 0.0
    for coord in 1:n_econ
        h_econ = coord == 1 ? h_gp : h_used[coord]
        wp = copy(w_econ); wp[coord] += h_econ
        wm = copy(w_econ); wm[coord] -= h_econ
        Kp = delta_dual_fixed_dual(obj, ζstar, λstar, wp, eta_nu, ctx, pe, W)
        Km = delta_dual_fixed_dual(obj, ζstar, λstar, wm, eta_nu, ctx, pe, W)
        fd = (Kp - Km) / (2h_econ)
        err = abs(fd - g_ext[coord]); relerr = err / max(1.0, abs(fd))
        max_rel = max(max_rel, relerr)
        @printf("  [%s] econ[%d]  analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", label, coord, g_ext[coord], fd, relerr)
    end
    for k in 1:n_eta_d
        ep = copy(eta_nu); ep[k] += h_eta
        em = copy(eta_nu); em[k] -= h_eta
        Kp = delta_dual_fixed_dual(obj, ζstar, λstar, w_econ, ep, ctx, pe, W)
        Km = delta_dual_fixed_dual(obj, ζstar, λstar, w_econ, em, ctx, pe, W)
        fd = (Kp - Km) / (2h_eta)
        err = abs(fd - g_ext[n_econ+k]); relerr = err / max(1.0, abs(fd))
        max_rel = max(max_rel, relerr)
        @printf("  [%s] eta[%d]   analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", label, k, g_ext[n_econ+k], fd, relerr)
    end
    check("$label: combined [econ;eta] gradient matches FD (max_rel_err < $tol)", max_rel < tol)
    return max_rel
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; W = size(ctx.U, 1)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
Random.seed!(20260804)

println("="^90); println("FAMILY origin-ZC: free-eta evaluator D4 gate"); println("="^90)
layout_o = OriginByPowerLayout(D, 1, 0)
# `moment_representation` defaults to :dense_reference (build_originzc_augmented_obj's own
# default) -- this `aug_reduced_oz.obj_cm` is a PsiObjectiveBundleImplicit, used ONLY as the FD
# ground truth below (mirrors test_zc_lane_originzc_outer_gradient_zerodense_d4_2026-08-02.jl's
# own comment: "used ONLY as the FD ground truth"). The REAL solve below goes through
# evaluate_profiled_originzc_point -> reduced_originzc_base_state -> its own internal
# build_reduced_originzc_operator_bundle, which returns a SEPARATE, genuinely zero-dense
# OperatorPsiBundle object (confirmed live: OperatorPsiBundle has no `.d` field, only `.M` --
# these are two different bundle instances built from the same aug_reduced_oz, not one shared obj).
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_fd_oz = aug_reduced_oz.obj_cm
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, fill(1.0, D))   # nu_full field unused by the eta-explicit method

eta0_oz = zeros(D)   # nu=1 at calibration, matches every fixed-nu ZC-lane gate's own convention
ev_oz = evaluate_profiled_originzc_point(w_profiled_calib, eta0_oz, fctx_oz, pes_oz)
check("origin-ZC free-eta: inner solve converges at calibration", ev_oz.result.inner_status == 0)
check("origin-ZC free-eta: eta_nu round-trips in ev", ev_oz.eta_nu == eta0_oz)
check("origin-ZC free-eta: nu_full == exp.(eta_nu)", ev_oz.nu_full ≈ exp.(eta0_oz))
g_ext_oz, meta_oz = reduced_originzc_outer_gradient_with_eta(w_profiled_calib, eta0_oz, ctx, fctx_oz, ev_oz)
@printf("origin-ZC: n_econ=%d n_eta=%d length(g_ext)=%d\n", length(w_profiled_calib), D, length(g_ext_oz))
gate_family("origin-ZC@calib", obj_fd_oz, w_profiled_calib, eta0_oz, ev_oz, g_ext_oz, ctx, pe, W, meta_oz.h_used)

# perturbed point (winner-changing/coupled coordinates), fresh octx (mirrors the 2026-08-02 gate's
# own "fresh state per solve" pattern to avoid any stale-workspace cross-talk)
w_pert = w_profiled_calib .+ 0.02 .* randn(length(w_profiled_calib))
eta_pert = 0.05 .* randn(D)
octx_reduced_oz2 = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
pes_oz2 = OriginZCPointEvalState(octx_reduced_oz2, fill(1.0, D))
ev_oz2 = evaluate_profiled_originzc_point(w_pert, eta_pert, fctx_oz, pes_oz2)
check("origin-ZC free-eta: inner solve converges at perturbed point", ev_oz2.result.inner_status == 0)
g_ext_oz2, meta_oz2 = reduced_originzc_outer_gradient_with_eta(w_pert, eta_pert, ctx, fctx_oz, ev_oz2)
gate_family("origin-ZC@pert", obj_fd_oz, w_pert, eta_pert, ev_oz2, g_ext_oz2, ctx, pe, W, meta_oz2.h_used)

println("\n" * "="^90); println("FAMILY CM+ZC: free-eta evaluator D4 gate"); println("="^90)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
bins_u32 = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
obj_fd_cz = aug_reduced_cz.obj_cm   # dense reference, FD ground truth only -- ev_cz.obj (from reduced_meanzc_base_state) is a separate zero-dense OperatorPsiBundle
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, fill(1.0, K_MEAN))

eta0_cz = zeros(K_MEAN)
ev_cz = evaluate_profiled_cmzc_point(w_profiled_calib, eta0_cz, fctx_cz, pes_cz)
check("CM+ZC free-eta: inner solve feasible/optimal at calibration", ev_cz.result.inner_status in (0, -100, -101, -103))
g_ext_cz, meta_cz = reduced_cmzc_outer_gradient_with_eta(w_profiled_calib, eta0_cz, ctx, fctx_cz, ev_cz)
@printf("CM+ZC: n_econ=%d n_eta=%d length(g_ext)=%d\n", length(w_profiled_calib), K_MEAN, length(g_ext_cz))
gate_family("CM+ZC@calib", obj_fd_cz, w_profiled_calib, eta0_cz, ev_cz, g_ext_cz, ctx, pe, W, meta_cz.h_used)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
