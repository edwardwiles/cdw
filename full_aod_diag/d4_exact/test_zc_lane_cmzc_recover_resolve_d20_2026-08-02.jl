# ZC lane task (2026-08-02), Phase II: real D20 omit-ROW recover-then-resolve equivalence for
# CM+ZC's widened-core reduced formulation. D20 analogue of
# test_zc_lane_cmzc_recover_resolve_2026-08-02.jl (decisive PASS at D4). W is read from
# ENV["ZC_LANE_D20_W"] (defaults to 20000).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "cm_meanzc_lookup_kernels.jl", "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl", "reduced_recovery_from_lfd_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random
flush(stdout)

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end

function brute_force_verify(obj, ζ::Float64, λ::AbstractVector{Float64}, θ_ext, U, W)
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_ext, U, obj)
    r = fill(-ζ, W) .- G * λ
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζ
    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g = -(G' * dPsi_r) ./ W
    return (r = r, f = f, g = g, m_weights = dPsi_r, Delta_dual = -f, Delta_primal = primal_divergence(dPsi_r),
        kkt_resid = maximum(abs, g))
end

const W_VAL = parse(Int, get(ENV, "ZC_LANE_D20_W", "20000"))
println("="^90); println("Building real D=20 :exclude_row context at W=$W_VAL ..."); flush(stdout)
ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx.D; Ddest = ctx.D_dest
@assert D == 20 && Ddest == 19 "expected live D=20, Ddest=19 -- got D=$D, Ddest=$Ddest"
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)
W = size(ctx.U, 1)

korea_idx = 14; brazil_idx = 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
println("LIVE DIMENSIONS: retained_factual=$(length(layout.retained_full_factual_j))  france=$has_france  total_reduced=$(layout.total_reduced_economic_moments)")
flush(stdout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

const K_MEAN, K_PAIR, L_GRID = 1, 0, 2   # L=2 -> ncm=(D-1)*L=38 at D=20, keeps the D20 solve tractable
νvec0 = fill(1.0, K_MEAN)

t_full = @elapsed begin
    aug_full = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
        moment_representation = :dense_reference)
    cctx_full = build_cm_meanzc_bin_ctx(ctx, aug_full; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference)
end
ctx_cm_full = (obj = aug_full.obj_cm, m = ctx.m)
println("built FULL cctx in $(t_full)s"); flush(stdout)

t_reduced_build = @elapsed begin
    aug_reduced = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
        base_obj = reduced_obj0, profiled_layout = layout)
    obj_reduced = aug_reduced.obj_cm
    cctx_reduced = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
        profiled_layout = layout)
end
ctx_cm_reduced = (obj = obj_reduced, m = ctx.m)
println("built REDUCED cctx in $(t_reduced_build)s"); flush(stdout)
println("cctx_reduced: ncore_core=$(cctx_reduced.ncore_core) NCORE=$(cctx_reduced.NCORE) ncm=$(cctx_reduced.ncm)")
println("ZC_GRAM_BACKEND_DEFAULT[]=", ZC_GRAM_BACKEND_DEFAULT[]); flush(stdout)

println("\n" * "="^90); println("STEP 1: REDUCED CM+ZC inner solve at calibration (D20, W=$W_VAL)"); println("="^90); flush(stdout)
reset_no_dense_g_counters!()
t_reduced = @elapsed base_reduced = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_reduced, cctx_reduced)
check("REDUCED inner solve feasible/optimal-within-tolerance", base_reduced.inner_status in (0, -100, -101, -103))
@printf("REDUCED: time=%.2fs  nStatus=%d  zeta*=%.10f\n", t_reduced, base_reduced.inner_status, base_reduced.ζstar)
flush(stdout)
c_disp = NO_DENSE_G_COUNTERS[]
check("dispatch proof: winner-based cross-Hessian path fired, zero dense fallback",
    c_disp.winner_cross_hessian_calls > 0 && c_disp.dense_cross_hessian_calls == 0)
@printf("winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d\n", c_disp.winner_cross_hessian_calls, c_disp.dense_cross_hessian_calls)
flush(stdout)

θ_ext_calib = vcat(collect(θ_full_calib), νvec0)
ov_r = brute_force_verify(obj_reduced, base_reduced.ζstar, base_reduced.λstar, θ_ext_calib, ctx.U, W)
println("verify_r: Delta_dual=", ov_r.Delta_dual, "  Delta_primal=", ov_r.Delta_primal, "  kkt_resid=", ov_r.kkt_resid)
flush(stdout)
cf_reduced = cctx_reduced.core_cf_ref[]

println("\n" * "="^90); println("STEP 2: RECOVER full-A using the REDUCED solve's own verified LFD"); println("="^90); flush(stdout)
z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(θ_full_calib, ctx, cf_reduced, ov_r.m_weights)
θ_full_recovered = copy(collect(θ_full_calib))
θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest] .= vec(exp.(z_recovered))
recovery_change_log = maximum(abs.(z_recovered .- log.(reshape(θ_full_calib[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))))
@printf("max|log A_recovered - log A_calib| = %.4g\n", recovery_change_log)
x_free_recovered = θ_full_recovered[ctx.free_idx]
flush(stdout)

println("\n" * "="^90); println("STEP 3: re-solve the LEGACY FULL CM+ZC problem AT THE RECOVERED A"); println("="^90); flush(stdout)
t_full2 = @elapsed base_full2 = archC_meanzc_base_state(x_free_recovered, νvec0, ctx_cm_full, cctx_full)
check("FULL@recovered inner solve feasible/optimal-within-tolerance", base_full2.inner_status in (0, -100, -101, -103))
@printf("FULL@recovered: time=%.2fs  nStatus=%d\n", t_full2, base_full2.inner_status)
flush(stdout)
θ_ext_recovered = vcat(θ_full_recovered, νvec0)
ov_full2 = brute_force_verify(aug_full.obj_cm, base_full2.ζstar, base_full2.λstar, θ_ext_recovered, ctx.U, W)
println("verify_full2: Delta_dual=", ov_full2.Delta_dual, "  Delta_primal=", ov_full2.Delta_primal, "  kkt_resid=", ov_full2.kkt_resid)
flush(stdout)

println("\n" * "="^90); println("STEP 4: THE DECISIVE COMPARISON (D=20, W=$W_VAL)"); println("="^90)
mw_diff = maximum(abs.(ov_full2.m_weights .- ov_r.m_weights))
delta_primal_diff = abs(ov_full2.Delta_primal - ov_r.Delta_primal)
@printf("%-32s %18.10g %18.10g %14.4g\n", "Delta_primal", ov_full2.Delta_primal, ov_r.Delta_primal, delta_primal_diff)
@printf("%-32s %18.10g %18.10g %14.4g\n", "Delta_dual", ov_full2.Delta_dual, ov_r.Delta_dual, abs(ov_full2.Delta_dual - ov_r.Delta_dual))
@printf("%-32s %18.4g\n", "max|m_weights diff|", mw_diff)
flush(stdout)

tol_LFD = 1e-2; tol_div = 1e-2
lfd_ok = mw_diff < tol_LFD * max(1.0, maximum(abs.(ov_r.m_weights)))
div_ok = delta_primal_diff < tol_div * max(1.0, abs(ov_r.Delta_primal))
check("recover-then-resolve: LFD match (tol $tol_LFD rel)", lfd_ok)
check("recover-then-resolve: Delta_primal match (tol $tol_div rel)", div_ok)

println()
println("W=$W_VAL  ", ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
