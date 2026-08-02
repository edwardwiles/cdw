# ZC lane task (2026-08-02), Phase I outer gradient: combined A/gp + eta/nu outer gradient for
# origin-ZC's real reduced context, gated against COMPLETE fixed-dual finite differences.
#
# "Fixed-dual" FD: hold (zeta*, lambda*) -- the REAL solved reduced dual point -- FIXED, perturb
# ONE outer coordinate (gp, a free relative-A coordinate, or an eta/nu target), rebuild G fresh at
# the perturbed theta_ext via the SAME obj.moments! every real solve uses, and recompute
# Delta_dual = -(sum(Psi(-zeta*-G*lambda*))/W + zeta*) -- exactly what a KNITRO objective evaluation
# at that theta would report, holding the dual fixed. This needs no q_decomposition sophistication:
# obj.moments! already captures every coordinate's true effect on G (economic block depends on
# theta/A; Z mean/pair columns depend on nu via mean_columns_direct!/pair_columns! -- confirmed by
# direct read of wrap_moments_with_originzc's own closure).
#
# Analytic side: shared_family_outer_gradient (the SAME one-method A/gp engine the unrestricted
# family uses, task's own "do not rederive" requirement) for [gp; A_free], PLUS
# d_delta_dual_d_eta_origin_vec (existing, UNCHANGED production sensitivity formula,
# cm_originzc_moments.jl) for eta/nu -- never re-derived.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl"]
    include(joinpath(D4X, f))
end

"""
    build_profiled_ab_spec_pe(ctx; global_overrides=Dict{Int,Int}()) -> (spec, gauge, pe)

Inlined COPY of profiled_outer_evaluator_2026-08-01.jl's own function of the same name (that file
is deliberately NOT included in this gate -- see the note below) -- same body, byte-for-byte.
"""
function build_profiled_ab_spec_pe(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    return spec, gauge, pe
end

"reduce_calibration_to_w_profiled(ctx, pe) -- inlined copy, same rationale as build_profiled_ab_spec_pe above."
function reduce_calibration_to_w_profiled(ctx, pe::PivotGravityElimOnRetained)
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gp0 = θ0[3+D]
    return reduce_to_w_profiled(gp0, z_calib, pe)
end

# NOTE (2026-08-02, found live via bisection): profiled_operator_bundle_2026-08-01.jl (the
# UNRESTRICTED family's own :profiled_destination_scales KNITRO driver -- a second, independent
# inner_loop_KNITRO_profiled that calls its own KN_new()) causes cross-talk with origin-ZC's own
# real KNITRO solve when BOTH are loaded in the same process: origin-ZC's solve silently degenerates
# to the untouched all-zeros initial point (zeta*=0.0 exactly) with a misleadingly-reported
# nStatus=0, and a KNITRO.jl-side exception ("...exception in puts callback: FieldError(...,:obj)")
# gets silently swallowed. Root cause not fully chased (a KNITRO/Julia global-callback-dispatch
# interaction, not a bug in this branch's own numerical code -- confirmed by bisection: the SAME
# origin-ZC solve is bit-identical/correct with every OTHER file loaded, only regresses once this
# ONE file is added). This gate does not need the unrestricted family's own driver at all, so the
# clean fix is simply not to load it -- `evaluate_profiled_point` is stubbed below (never called)
# purely to satisfy profiled_lfix_incremental_2026-08-01.jl's own unrelated include-guard.
function evaluate_profiled_point end
for f in [
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Fixed-dual Delta_dual at a perturbed (w_profiled, nu_full), zeta*/lambda* held FIXED at the real
solved point. Rebuilds G fresh via obj.moments! (the SAME production closure every real solve
uses), never a hand-derived formula."
function delta_dual_fixed_dual(obj, ζstar::Float64, λstar::Vector{Float64}, w_profiled::Vector{Float64},
        νfull::Vector{Float64}, ctx, pe, W::Int)
    decoded = decode_outer_profiled(w_profiled, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    θ_ext = vcat(θ_full, νfull)
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_ext, ctx.U, obj)
    r = fill(-ζstar, W) .- G * λstar
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζstar
    return -f
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)
D = ctx.D
W = size(ctx.U, 1)

spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("outer_dim_profiled = $n_total (1 gp + $(n_total-1) free-A)")

layout_o = OriginByPowerLayout(D, 1, 0)   # K_mean=1, K_pair=0 -- the one safe config
νvec0 = fill(1.0, D)   # n_eta = K_mean*D = D

aug_reduced = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
octx_reduced = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
ctx_cm_reduced = (obj = obj_reduced, m = ctx.m, octx = octx_reduced)

println("="^78); println("STEP 1: REDUCED origin-ZC inner solve at calibration"); println("="^78)
base_reduced = archOZ_base_state(x_free_calib, νvec0, ctx_cm_reduced)
check("REDUCED inner solve converges", base_reduced.inner_status == 0)
n_econ = layout.total_reduced_economic_moments
n_eta_total = n_eta(layout_o)
β_full = base_reduced.λstar
@printf("nStatus=%d  zeta*=%.8f  n_econ=%d  n_eta=%d  length(beta)=%d\n",
    base_reduced.inner_status, base_reduced.ζstar, n_econ, n_eta_total, length(β_full))

println("\n" * "="^78); println("STEP 2: analytic combined gradient at calibration"); println("="^78)
fctx = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced)
cf_reduced = octx_reduced.core_cf_ref[]
cf_reduced isa CompressedFactual || error("core_cf_ref[] is not a CompressedFactual after solve")
K_v = Vector{Float64}(undef, W); G_v = Matrix{Float64}(undef, W, obj_reduced.d - 1)
obj_reduced.moments!(K_v, G_v, vcat(collect(θ_full_calib), νvec0), ctx.U, obj_reduced)
r_v = fill(-base_reduced.ζstar, W) .- G_v * β_full
m_weights = similar(r_v); obj_reduced.dPsi!(m_weights, r_v)
mean_m = sum(m_weights) / W
ev = (result = (beta = β_full, zeta = base_reduced.ζstar), st = (cf = cf_reduced, layout = layout),
      theta_full = collect(θ_full_calib), obj = (M = W,), m_weights = m_weights, nu_full = νvec0)

cache_diag = build_shared_profiled_lfix_cache(w_profiled_calib, fctx, ctx, ev)
q_true = r_v   # = -zeta* .- G*beta, already computed above (r_v), the REAL solved q at calibration
println("q0 vs q_true diagnostic: max|cache.q0 - q_true|=", maximum(abs.(cache_diag.q0 .- q_true)),
    "  (should be ~0 -- q0 is the cache's own baseline reconstruction of the SAME quantity)")
println("  q0[1:3]=", cache_diag.q0[1:3], "  q_true[1:3]=", q_true[1:3])
println("  const_part=", cache_diag.const_part, "  sum(contrib0[1,:])=", sum(cache_diag.contrib0[1, :]))

g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev)
g_eta = d_delta_dual_d_eta_origin_vec(β_full, aug_reduced, νvec0; mean_m = mean_m)
println("g_Agp: length=$(length(g_Agp))  g_Agp[1](dK/dgp)=$(g_Agp[1])")
println("g_eta: length=$(length(g_eta))  g_eta[1:3]=$(g_eta[1:min(3,end)])")

println("\n" * "="^78); println("STEP 3: gate vs COMPLETE fixed-dual finite differences (calibration point)"); println("="^78)
# IMPORTANT (this repo's own documented pitfall, memory feedback-fd-bandwidth-mismatch-looks-like-
# a-bug: "adaptive-h vs fixed-h FD comparison gives false ~1e-3 gap; match h before hunting bugs"):
# the ANALYTIC gradient's own A-block is ITSELF a finite difference internally
# (profiled_lfix_incremental_at at w_profiled[coord]+/-h), using profiled_select_bandwidth's own
# PER-COORDINATE ADAPTIVE h (meta.h_used[coord], chosen to land cleanly around winner switches) --
# NOT a fixed h. Comparing against a DIFFERENT, fixed h here would reproduce exactly that documented
# false-mismatch pattern. Match h_used exactly (gp, coord 1, has no bandwidth -- it's a genuinely
# analytic closed-form derivative, so a small fixed h is fine and used only for gp).
h_A = 1e-4; h_eta = 1e-4
max_rel_err_A = 0.0; max_abs_err_A = 0.0
for coord in 1:n_total
    h_c = coord == 1 ? h_A : meta.h_used[coord]
    wp = copy(w_profiled_calib); wp[coord] += h_c
    Kp = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, wp, νvec0, ctx, pe, W)
    wm = copy(w_profiled_calib); wm[coord] -= h_c
    Km = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, wm, νvec0, ctx, pe, W)
    fd = (Kp - Km) / (2h_c)
    err = abs(fd - g_Agp[coord])
    relerr = err / max(1.0, abs(fd))
    global max_rel_err_A = max(max_rel_err_A, relerr)
    global max_abs_err_A = max(max_abs_err_A, err)
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  abs_err=%.3e  rel_err=%.3e\n", label, g_Agp[coord], h_c, fd, err, relerr)
end
check("A/gp gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_A < 5e-3)

max_rel_err_eta = 0.0
n_eta_test = min(n_eta_total, D)   # test every eta/nu coordinate at D4 (small)
for k in 1:n_eta_test
    νp = copy(νvec0); νp[k] += h_eta
    Kp = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, w_profiled_calib, νp, ctx, pe, W)
    νm = copy(νvec0); νm[k] -= h_eta
    Km = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, w_profiled_calib, νm, ctx, pe, W)
    fd_nu = (Kp - Km) / (2h_eta)
    # g_eta is d(Delta_dual)/d(eta_k) = nu_k * d(Delta_dual)/d(nu_k) (chain rule, nu=exp(eta));
    # FD above is w.r.t. nu directly, so compare fd_nu*nu_k against g_eta[k].
    fd_eta = fd_nu * νvec0[k]
    err = abs(fd_eta - g_eta[k])
    relerr = err / max(1.0, abs(fd_eta))
    global max_rel_err_eta = max(max_rel_err_eta, relerr)
    @printf("  eta[%d]         analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", k, g_eta[k], fd_eta, relerr)
end
check("eta/nu gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_eta < 5e-3)

println("\n" * "="^78); println("STEP 4: gate at a PERTURBED (non-calibration) point -- winner-changing/coupled coordinates"); println("="^78)
w_profiled_pert = w_profiled_calib .+ 0.02 .* randn(n_total)
νvec_pert = νvec0 .* exp.(0.05 .* randn(D))
decoded_pert = decode_outer_profiled(w_profiled_pert, ctx, pe)
θ_full_pert = CS.reconstruct_full(decoded_pert.xf, ctx.m)
aug_reduced2 = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced2 = aug_reduced2.obj_cm
octx_reduced2 = build_originzc_core_hess_ctx(aug_reduced2, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
ctx_cm_reduced2 = (obj = obj_reduced2, m = ctx.m, octx = octx_reduced2)
x_free_pert = θ_full_pert[ctx.free_idx]
base_pert = archOZ_base_state(x_free_pert, νvec_pert, ctx_cm_reduced2)
check("REDUCED inner solve converges at perturbed point", base_pert.inner_status == 0)
β_pert = base_pert.λstar
fctx2 = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced2)
cf_pert = octx_reduced2.core_cf_ref[]
K_v2 = Vector{Float64}(undef, W); G_v2 = Matrix{Float64}(undef, W, obj_reduced2.d - 1)
obj_reduced2.moments!(K_v2, G_v2, vcat(θ_full_pert, νvec_pert), ctx.U, obj_reduced2)
r_v2 = fill(-base_pert.ζstar, W) .- G_v2 * β_pert
mw2 = similar(r_v2); obj_reduced2.dPsi!(mw2, r_v2)
ev2 = (result = (beta = β_pert, zeta = base_pert.ζstar), st = (cf = cf_pert, layout = layout),
       theta_full = θ_full_pert, obj = (M = W,), m_weights = mw2, nu_full = νvec_pert)
g_Agp2, meta2 = shared_family_outer_gradient(w_profiled_pert, ctx, fctx2, ev2)

max_rel_err_A2 = 0.0
test_coords = unique(vcat(1, 2, pe.pivot_pos == 2 ? 3 : 2, rand(2:n_total, min(4, n_total - 1))))
for coord in test_coords
    h_c = coord == 1 ? h_A : meta2.h_used[coord]
    wp = copy(w_profiled_pert); wp[coord] += h_c
    Kp = delta_dual_fixed_dual(obj_reduced2, base_pert.ζstar, β_pert, wp, νvec_pert, ctx, pe, W)
    wm = copy(w_profiled_pert); wm[coord] -= h_c
    Km = delta_dual_fixed_dual(obj_reduced2, base_pert.ζstar, β_pert, wm, νvec_pert, ctx, pe, W)
    fd = (Kp - Km) / (2h_c)
    err = abs(fd - g_Agp2[coord])
    relerr = err / max(1.0, abs(fd))
    global max_rel_err_A2 = max(max_rel_err_A2, relerr)
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  rel_err=%.3e\n", label, g_Agp2[coord], h_c, fd, relerr)
end
check("A/gp gradient matches FD at perturbed point (max_rel_err < 5e-3)", max_rel_err_A2 < 5e-3)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
