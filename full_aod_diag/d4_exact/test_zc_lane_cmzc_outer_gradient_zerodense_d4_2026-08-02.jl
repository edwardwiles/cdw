# Phase2b task (2026-08-02): TRUE zero-dense sibling of test_zc_lane_cmzc_outer_gradient_d4_2026-08-02.jl.
#
# The existing dense-FG gate solves CM+ZC's inner problem via `archC_meanzc_base_state` (the
# DENSE-width :dense_reference FG evaluator), only afterward flipping `cctx.inner_fg_backend` to
# `:cm_lookup` as documentation of intent (a mutable-field flag that the DENSE driver never actually
# reads/dispatches on). This file instead solves through `reduced_meanzc_base_state` --
# `profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl`'s own genuinely matrix-free
# `ReducedCMMeanZCOperatorState`/`inner_loop_KNITRO_reduced_meanzc` driver, the SAME incantation
# `test_profiled_reduced_meanzc_operator_and_autodiff_2026-08-02.jl` already uses and gates via
# NO_DENSE_G_COUNTERS. `CMZCFamilyCtx` is built from the identical `(aug_reduced, cctx_reduced)`
# pair regardless of which driver solved the inner problem -- the adapter only ever reads
# POST-SOLVE outputs (`ev.result.beta`/`ev.nu_full`/`ev.st.cf`), never per-draw kernel internals, so
# no adapter code changes are needed, only which function performs the solve+refresh.
#
# `m_weights`/`mean_m` (needed by `d_delta_dual_d_eta_nu_vec`) are recovered WITHOUT any dense G: a
# single extra FG evaluation of the solved point through the SAME zero-dense `st` functor
# (`ReducedCMMeanZCOperatorState`'s own callable) refreshes `st.arg1 = dPsi(q*)` in place -- exactly
# the quantity `economic_transpose_into_g1_and_gE_reduced!` computes internally on every gradient
# call, reused here rather than re-derived.
#
# Gate: NO_DENSE_G_COUNTERS deltas (economic/CM/ZC) are asserted EXACTLY ZERO across inner solve +
# outer gradient evaluation (both calibration and the extra dPsi refresh) -- not just the inner
# solve. The FD ground truth (`delta_dual_fixed_dual`, reused verbatim from the existing dense gate)
# necessarily rebuilds a dense reference G via `obj.moments!` -- by design, exactly as every other
# gate in this codebase does for its FD reference -- and is therefore measured OUTSIDE that bracket.
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
          "cm_meanzc_lookup_kernels.jl", "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
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
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end

"Inlined copy of profiled_outer_evaluator_2026-08-01.jl's own function (NOT loaded here -- same
cross-talk risk with a real family KNITRO solve documented in the ZC-lane sibling gates)."
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

function evaluate_profiled_point end
for f in ["profiled_lfix_incremental_2026-08-01.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Fixed-dual Delta_dual at a perturbed (w_profiled, nu_full), zeta*/lambda* held FIXED at the real
solved point -- rebuilds G fresh via obj.moments! (the SAME production closure every real solve
uses), never a hand-derived formula. IDENTICAL to the existing dense gate's own helper -- the FD
GROUND TRUTH is legitimately allowed to use a dense reference; only the SOLVE and the GRADIENT under
test must be zero-dense."
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

const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0 = fill(1.0, K_MEAN)

aug_reduced = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm   # dense-reference closure, used ONLY as the FD ground truth below
cctx_reduced = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)

println("="^78); println("STEP 1: TRUE ZERO-DENSE reduced CM+ZC inner solve at calibration"); println("="^78)
before_counters = deepcopy(NO_DENSE_G_COUNTERS[])
base_reduced = reduced_meanzc_base_state(x_free_calib, νvec0, ctx, layout, cctx_reduced)
check("REDUCED (operator) inner solve feasible/optimal-within-tolerance", base_reduced.inner_status in (0, -100, -101, -103))
n_econ = layout.total_reduced_economic_moments
β_full = base_reduced.λstar
@printf("nStatus=%d  zeta*=%.8f  n_econ=%d  length(beta)=%d  n_fg=%d\n",
    base_reduced.inner_status, base_reduced.ζstar, n_econ, length(β_full), base_reduced.n_fg)

println("\n" * "="^78); println("STEP 2: analytic combined gradient at calibration (zero-dense throughout)"); println("="^78)
bins_u32 = cctx_reduced.Bidx isa Matrix{UInt32} ? cctx_reduced.Bidx : Matrix{UInt32}(cctx_reduced.Bidx)
fctx = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced, cctx_reduced, bins_u32)

# Zero-dense m_weights/mean_m recovery: re-evaluate the solved point through the SAME operator
# functor (no dense G) to refresh st.arg1 = dPsi(q*) in place.
st_reduced = base_reduced.st
x_full = vcat(base_reduced.ζstar, β_full)
g_buf = zeros(length(x_full))
f_check = st_reduced(x_full, g_buf)
m_weights = copy(st_reduced.arg1)
mean_m = sum(m_weights) / W
cf_reduced = cctx_reduced.core_cf_ref[]
cf_reduced isa CompressedFactual || error("core_cf_ref[] is not a CompressedFactual after solve")
ev = (result = (beta = β_full, zeta = base_reduced.ζstar), st = (cf = cf_reduced, layout = layout),
      theta_full = collect(θ_full_calib), obj = (M = W,), m_weights = m_weights, nu_full = νvec0)

g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev)
g_eta = d_delta_dual_d_eta_nu_vec(β_full, aug_reduced, νvec0; mean_m = mean_m)
after_counters = NO_DENSE_G_COUNTERS[]
println("g_Agp: length=$(length(g_Agp))  g_Agp[1](dK/dgp)=$(g_Agp[1])")
println("g_eta: length=$(length(g_eta))  g_eta=$(g_eta)")

println("\n" * "="^78); println("STEP 2b: NO_DENSE_G_COUNTERS delta across inner solve + outer gradient evaluation"); println("="^78)
d_econ = after_counters.dense_economic_G_materializations - before_counters.dense_economic_G_materializations
d_cm = after_counters.dense_CM_G_materializations - before_counters.dense_CM_G_materializations
d_zc = after_counters.dense_ZC_G_materializations - before_counters.dense_ZC_G_materializations
@printf("  delta dense_economic_G_materializations=%d  delta dense_CM_G_materializations=%d  delta dense_ZC_G_materializations=%d\n",
    d_econ, d_cm, d_zc)
check("ZERO dense economic G materializations across solve+gradient", d_econ == 0)
check("ZERO dense CM-grid G materializations across solve+gradient", d_cm == 0)
check("ZERO dense ZC G materializations across solve+gradient", d_zc == 0)

println("\n" * "="^78); println("STEP 3: gate vs COMPLETE fixed-dual finite differences (calibration point)"); println("="^78)
h_A = 1e-4; h_eta = 1e-4
max_rel_err_A = 0.0
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
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  rel_err=%.3e\n", label, g_Agp[coord], h_c, fd, relerr)
end
check("A/gp gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_A < 5e-3)

max_rel_err_eta = 0.0
for k in 1:K_MEAN
    νp = copy(νvec0); νp[k] += h_eta
    Kp = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, w_profiled_calib, νp, ctx, pe, W)
    νm = copy(νvec0); νm[k] -= h_eta
    Km = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, w_profiled_calib, νm, ctx, pe, W)
    fd_nu = (Kp - Km) / (2h_eta)
    fd_eta = fd_nu * νvec0[k]
    err = abs(fd_eta - g_eta[k])
    relerr = err / max(1.0, abs(fd_eta))
    global max_rel_err_eta = max(max_rel_err_eta, relerr)
    @printf("  eta[%d]         analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", k, g_eta[k], fd_eta, relerr)
end
check("eta/nu gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_eta < 5e-3)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
