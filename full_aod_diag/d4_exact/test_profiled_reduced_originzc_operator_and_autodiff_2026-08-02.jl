# Integration continuation (2026-08-02): verification gate for ZC-only's genuine, matrix-free
# reduced+Z(mean/pair) operator FG (profiled_reduced_originzc_lookup_kernels_2026-08-02.jl).
# Combines: real KNITRO solve + zero-dense-G counters + machine-precision ForwardDiff gradient/
# Hessian cross-check (per explicit user instruction, same method as flexible CM/common Fréchet's
# own autodiff gates).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "zc_restriction_operator.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random, ForwardDiff

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end
function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

layout_o = OriginByPowerLayout(ctx.D, 1, 0)
νfull0 = fill(1.0, ctx.D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)

# =====================================================================================
# Check 1+2: real KNITRO solve via the new operator path + zero dense-G materialization
# =====================================================================================
println("="^90); println("Check 1: real KNITRO solve via the new (gravity-free) operator path"); println("="^90)
octx_reduced = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                             zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
before = deepcopy(NO_DENSE_G_COUNTERS[])
base_new = reduced_originzc_base_state(x_free_calib, ctx, layout, octx_reduced, νfull0)
after = NO_DENSE_G_COUNTERS[]
@printf("  NEW (operator, zero-dense-G, no gravity)  inner_status=%d  zeta*=%.12f\n", base_new.inner_status, base_new.ζstar)
check("Check1: NEW operator-path reduced solve reaches optimality (nStatus==0)", base_new.inner_status == 0)

println("="^90); println("Check 2: zero dense-G materialization (NO_DENSE_G_COUNTERS delta across the new-path solve)"); println("="^90)
d_econ = after.dense_economic_G_materializations - before.dense_economic_G_materializations
@printf("  delta dense_economic_G_materializations=%d  (n_fg_calls this solve=%d)\n", d_econ, base_new.n_fg)
check("Check2: zero dense economic G materializations during the new-path solve", d_econ == 0)

# =====================================================================================
# Check 3: ForwardDiff gradient+Hessian cross-check (machine precision)
# =====================================================================================
println("="^90); println("Check 3: ForwardDiff gradient+Hessian cross-check (machine precision)"); println("="^90)
octx_probe = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                           zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
obj_probe, st_probe = build_reduced_originzc_operator_bundle(ctx, θ_full_calib, layout, octx_probe)
refresh_zc_targets!(st_probe.zc_ws, st_probe.op, st_probe.zc_layout, νfull0)
octx_probe.nu_ref[] = collect(νfull0)   # Hessian callback's own H_ZZ primitive reads this SEPARATE box
prime_operator!(obj_probe, θ_full_calib, ctx, octx_probe.core_cf_ref; restriction_state = octx_probe)
octx_probe.profiled_theta_ref[] = copy(θ_full_calib)
cf = octx_probe.core_cf_ref[]::CompressedFactual

op = st_probe.op
n_econ = layout.total_reduced_economic_moments
nm = n_mean(op); npr = n_pair(op)
n = 1 + n_econ + nm + npr
W = cf.W

println("Materializing D4-scale reference G_econ/G_Z (test-only)")
G_econ = Matrix{Float64}(undef, W, n_econ)
e_econ = zeros(n_econ)
for j in 1:n_econ
    e_econ[j] = 1.0
    G_econ[:, j] .= reduced_homogeneous_dual_contraction(e_econ, cf, ctx, θ_full_calib, layout)
    e_econ[j] = 0.0
end
G_Z = Matrix{Float64}(undef, W, nm + npr)
e_mean = zeros(nm); e_pair = zeros(npr)
for j in 1:(nm + npr)
    dest = zeros(W)
    if j <= nm
        e_mean[j] = 1.0
        restriction_forward!(dest, e_mean, e_pair, op, st_probe.zc_ws)
        e_mean[j] = 0.0
    else
        e_pair[j - nm] = 1.0
        restriction_forward!(dest, e_mean, e_pair, op, st_probe.zc_ws)
        e_pair[j - nm] = 0.0
    end
    G_Z[:, j] .= dest
end
check("reference matrices finite", all(isfinite, G_econ) && all(isfinite, G_Z))
@printf("  G_econ: %dx%d  G_Z: %dx%d (n_mean=%d n_pair=%d)\n", size(G_econ)..., size(G_Z)..., nm, npr)

psi_scalar(q) = q <= 1.0 ? exp(q) - 1.0 : 0.5 * exp(1) * (q^2 + 1.0) - 1.0
M = obj_probe.M
function f_ref(x)
    ζ = x[1]
    β = @view x[2:1+n_econ]
    λZ = @view x[2+n_econ:1+n_econ+nm+npr]
    q = (-ζ) .- G_econ * β .+ G_Z * λZ
    return ζ + sum(psi_scalar, q) / M
end

Random.seed!(2027)
x0 = 0.01 .* randn(n)

g_ad = ForwardDiff.gradient(f_ref, x0)
g_analytic = zeros(n)
f0_analytic = st_probe(x0, g_analytic)
f0_ref = f_ref(x0)
@printf("  f0: analytic=%.15g  ref=%.15g  |Δ|=%.3e\n", f0_analytic, f0_ref, abs(f0_analytic - f0_ref))
check("objective value matches ForwardDiff reference (<1e-10)", abs(f0_analytic - f0_ref) < 1e-10)
err_g = maximum(abs.(g_analytic .- g_ad))
relerr_g = err_g / maximum(abs.(g_ad))
@printf("  gradient: max_abs_err=%.3e  max_rel_err=%.3e\n", err_g, relerr_g)
check("analytic gradient matches ForwardDiff (machine precision, <1e-10)", err_g < 1e-10)

H_ad = ForwardDiff.hessian(f_ref, x0)
dual_index!(st_probe, x0)
obj_probe.arg0 .= st_probe.arg0
octx_probe.fg_lookup_st = st_probe   # so _prep_dual_index_for_archA! routes through operator_prep_for_hessian!, not the dense fallback
octx_probe.fg_backend = :operator
h_packed = Vector{Float64}(undef, n * (n + 1) ÷ 2)
hess_cb = archA_partitioned_hess_cb_builder(octx_probe)
evalReq = (x = x0,)
evalRes = (hess = h_packed,)
hess_cb(nothing, nothing, evalReq, evalRes, obj_probe)
H_prod = unpack_packed(h_packed, n)

check("production Hessian finite", all(isfinite, H_prod))
check("production Hessian symmetric", maximum(abs.(H_prod .- H_prod')) < 1e-10)
err_h = maximum(abs.(H_ad .- H_prod))
scale_h = max(1.0, maximum(abs.(H_ad)))
relerr_h = err_h / scale_h
@printf("  Hessian: max_abs_err=%.3e  max_rel_err=%.3e  (n=%d)\n", err_h, relerr_h, n)
check("production Hessian matches ForwardDiff (machine precision, <1e-7 abs)", err_h < 1e-7)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
