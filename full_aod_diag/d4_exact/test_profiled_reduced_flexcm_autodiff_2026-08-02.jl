# Integration continuation (2026-08-02): machine-precision ForwardDiff cross-check of both the
# gradient AND the Hessian for flexible CM's reduced+CM-grid operator FG (same method as common
# Fréchet's own test_profiled_reduced_frechet_autodiff_2026-08-02.jl).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl"]
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
L = 10; contrasts = :anchored

aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
cctx_probe = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
obj_probe, st_probe = build_reduced_cm_operator_bundle(ctx, θ_full_calib, layout, cctx_probe)
prime_operator!(obj_probe, θ_full_calib, ctx, cctx_probe.core_cf_ref; restriction_state = cctx_probe)
cctx_probe.profiled_theta_ref[] = copy(θ_full_calib)
cf = cctx_probe.core_cf_ref[]::CompressedFactual

n_econ = layout.total_reduced_economic_moments
ncm = cctx_probe.ncm
n = 1 + n_econ + ncm
D = cctx_probe.D; W = cf.W

println("="^90); println("Materializing D4-scale reference G_econ/G_cm (test-only)"); println("="^90)
G_econ = Matrix{Float64}(undef, W, n_econ)
e_econ = zeros(n_econ)
for j in 1:n_econ
    e_econ[j] = 1.0
    G_econ[:, j] .= reduced_homogeneous_dual_contraction(e_econ, cf, ctx, θ_full_calib, layout)
    e_econ[j] = 0.0
end
G_cm = Matrix{Float64}(undef, W, ncm)
bins_int = Int.(cctx_probe.Bidx)
fill_cm_columns_from_bins!(G_cm, bins_int, cctx_probe.origins, cctx_probe.refIndex1, cctx_probe.L, cctx_probe.R)
check("reference matrices finite", all(isfinite, G_econ) && all(isfinite, G_cm))
@printf("  G_econ: %dx%d  G_cm: %dx%d\n", size(G_econ)..., size(G_cm)...)

psi_scalar(q) = q <= 1.0 ? exp(q) - 1.0 : 0.5 * exp(1) * (q^2 + 1.0) - 1.0
M = obj_probe.M
function f_ref(x)
    ζ = x[1]
    β = @view x[2:1+n_econ]
    λcm = @view x[2+n_econ:1+n_econ+ncm]
    q = (-ζ) .- G_econ * β .- G_cm * λcm
    return ζ + sum(psi_scalar, q) / M
end

Random.seed!(2027)
x0 = 0.01 .* randn(n)

println("="^90); println("ForwardDiff gradient vs analytic FG (machine precision)"); println("="^90)
g_ad = ForwardDiff.gradient(f_ref, x0)
g_analytic = zeros(n)
f0_analytic = st_probe(x0, g_analytic)
f0_ref = f_ref(x0)
@printf("  f0: analytic=%.15g  ref=%.15g  |Δ|=%.3e\n", f0_analytic, f0_ref, abs(f0_analytic - f0_ref))
check("objective value matches ForwardDiff reference (<1e-10)", abs(f0_analytic - f0_ref) < 1e-10)
err_g = maximum(abs.(g_analytic .- g_ad))
relerr_g = err_g / maximum(abs.(g_ad))
@printf("  max_abs_err=%.3e  max_rel_err=%.3e\n", err_g, relerr_g)
check("analytic gradient matches ForwardDiff (machine precision, <1e-10)", err_g < 1e-10)

println("="^90); println("ForwardDiff Hessian vs production hessian_cm_structured! (machine precision)"); println("="^90)
H_ad = ForwardDiff.hessian(f_ref, x0)
dual_index!(st_probe, x0)
obj_probe.arg0 .= st_probe.arg0
h_packed = Vector{Float64}(undef, n * (n + 1) ÷ 2)
hessian_cm_structured!(h_packed, obj_probe, cctx_probe)
H_prod = unpack_packed(h_packed, n)

check("production Hessian finite", all(isfinite, H_prod))
check("production Hessian symmetric", maximum(abs.(H_prod .- H_prod')) < 1e-10)
err_h = maximum(abs.(H_ad .- H_prod))
scale_h = max(1.0, maximum(abs.(H_ad)))
relerr_h = err_h / scale_h
@printf("  max_abs_err=%.3e  max_rel_err=%.3e  (n=%d)\n", err_h, relerr_h, n)
check("production Hessian matches ForwardDiff (machine precision, <1e-7 abs)", err_h < 1e-7)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
