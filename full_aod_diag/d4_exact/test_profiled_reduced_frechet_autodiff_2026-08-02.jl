# Integration continuation (2026-08-02): machine-precision ForwardDiff (not finite-difference)
# cross-check of both the gradient AND the Hessian for common Fréchet's reduced+CM-grid+level
# operator FG (per explicit user instruction).
#
# METHOD: materialize small, D4-scale, TEST-ONLY dense reference matrices G_econ/G_cm/G_level via
# unit-vector calls to the SAME already-validated forward kernels this file's own production code
# uses (reduced_homogeneous_dual_contraction for economic -- exactly materialize_homogeneous_dense_
# G_reduced!'s own technique; fill_cm_columns_from_bins! for CM -- the SAME cumulative-basis fill
# wrap_moments_with_cm_frechet_archB itself uses; frechet_level_forward_sum! for level, by the same
# linearity argument). Each of the three blocks is exactly LINEAR in its own dual sub-vector (no
# additive x-independent constant beyond what unit-vector materialization already captures), so
# `q(x) = -zeta*1 - G_econ*beta_econ - G_cm*lambda_cm - G_level*lambda_level` is an EXACT (not
# approximate) reference formula, and is trivially ForwardDiff-generic (plain dense matrix-vector
# products against fixed Float64 matrices). This is NOT a new architecture or a production change --
# it is a one-off, test-only reference construction for cross-validation, exactly mirroring this
# whole codebase's own "safe by linearity" verification discipline (see Part 1 of every D4 solve
# gate: "reduced G bilateral columns match gathered-from-full HOMOGENEOUS G").
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "compressed_factual_buffer_reuse.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl"]
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

aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_f.level_targets

cctx_probe = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
                               threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
obj_probe, st_probe = build_reduced_frechet_operator_bundle(ctx, θ_full_calib, layout, cctx_probe, level_targets)
prime_operator!(obj_probe, θ_full_calib, ctx, cctx_probe.core_cf_ref; restriction_state = cctx_probe)
cctx_probe.profiled_theta_ref[] = copy(θ_full_calib)
cf = cctx_probe.core_cf_ref[]::CompressedFactual

n_econ = layout.total_reduced_economic_moments
ncm_level = cctx_probe.L
ncm_cm = cctx_probe.ncm - ncm_level
n = 1 + n_econ + ncm_cm + ncm_level
D = cctx_probe.D; W = cf.W

println("="^90); println("Materializing D4-scale reference G_econ/G_cm/G_level (test-only, via already-validated forward kernels)"); println("="^90)

# ---- G_econ: unit-vector loop through reduced_homogeneous_dual_contraction (SAME technique
# materialize_homogeneous_dense_G_reduced! itself uses) ----
G_econ = Matrix{Float64}(undef, W, n_econ)
e_econ = zeros(n_econ)
for j in 1:n_econ
    e_econ[j] = 1.0
    G_econ[:, j] .= reduced_homogeneous_dual_contraction(e_econ, cf, ctx, θ_full_calib, layout)
    e_econ[j] = 0.0
end

# ---- G_cm: fill_cm_columns_from_bins! (the SAME cumulative-basis fill wrap_moments_with_cm_
# frechet_archB itself uses for this family's CM block) ----
G_cm = Matrix{Float64}(undef, W, ncm_cm)
bins_int = Int.(cctx_probe.Bidx)
fill_cm_columns_from_bins!(G_cm, bins_int, cctx_probe.origins, cctx_probe.refIndex1, cctx_probe.L, cctx_probe.R)

# ---- G_level: unit-vector loop through frechet_level_suffix_sums!/frechet_level_forward_sum!
# (the SAME kernels the production level-block FG itself uses) ----
G_level = Matrix{Float64}(undef, W, ncm_level)
P_level = zeros(ncm_level + 1)
e_level = zeros(ncm_level)
level_contrib = zeros(W)
bins_u = cctx_probe.Bidx isa Matrix{UInt32} ? cctx_probe.Bidx : Matrix{UInt32}(cctx_probe.Bidx)
for l in 1:ncm_level
    e_level[l] = 1.0
    frechet_level_suffix_sums!(P_level, e_level)
    frechet_level_forward_sum!(level_contrib, bins_u, D, P_level)
    # forward contribution for a UNIT lambda_level[l] is invsqrtD*level_contrib .- level_targets[l]
    # (const_term = e_level'*level_targets = level_targets[l] for this unit vector)
    G_level[:, l] .= (1.0 / sqrt(D)) .* level_contrib .- level_targets[l]
    e_level[l] = 0.0
end

check("reference matrices finite", all(isfinite, G_econ) && all(isfinite, G_cm) && all(isfinite, G_level))
@printf("  G_econ: %dx%d  G_cm: %dx%d  G_level: %dx%d\n", size(G_econ)..., size(G_cm)..., size(G_level)...)

# ---- Autodiff-generic reference objective (pure function of x, no mutation, no persistent state) ----
psi_scalar(q) = q <= 1.0 ? exp(q) - 1.0 : 0.5 * exp(1) * (q^2 + 1.0) - 1.0
M = obj_probe.M
function f_ref(x)
    ζ = x[1]
    β = @view x[2:1+n_econ]
    λcm = @view x[2+n_econ:1+n_econ+ncm_cm]
    λlvl = @view x[2+n_econ+ncm_cm:1+n_econ+ncm_cm+ncm_level]
    q = (-ζ) .- G_econ * β .- G_cm * λcm .- G_level * λlvl
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

# Prime the production Hessian machinery at x0 EXACTLY as _prep_dual_index_for_archC!/
# operator_prep_for_hessian! would (dual_index! + obj.arg0 sync), then call hessian_cm_structured!
# DIRECTLY (same technique test_profiled_flexcm_d4_hessian_gate_2026-08-01.jl's own "New code under
# test" section already uses) with the CORRECT common-Fréchet extension resolved via
# archC_frechet_hess_cb_builder's own _resolve_frechet_ext!/archC_frechet_hess_cb_builder, so the
# level block is genuinely exercised (not silently skipped via a default extension=nothing).
dual_index!(st_probe, x0)
obj_probe.arg0 .= st_probe.arg0
frechet_ext = _resolve_frechet_ext!(cctx_probe, level_targets)
h_packed = Vector{Float64}(undef, n * (n + 1) ÷ 2)
hessian_cm_structured!(h_packed, obj_probe, cctx_probe, frechet_ext)
H_prod = unpack_packed(h_packed, n)

check("production Hessian finite", all(isfinite, H_prod))
check("production Hessian symmetric", maximum(abs.(H_prod .- H_prod')) < 1e-10)
err_h = maximum(abs.(H_ad .- H_prod))
scale_h = max(1.0, maximum(abs.(H_ad)))
relerr_h = err_h / scale_h
@printf("  max_abs_err=%.3e  max_rel_err=%.3e  (n=%d)\n", err_h, relerr_h, n)
check("production Hessian matches ForwardDiff (machine precision, <1e-7 abs, generous for D4 conditioning)", err_h < 1e-7)

if err_h >= 1e-7
    println("  Block-by-block breakdown (to localize the discrepancy):")
    HEE_ad = H_ad[1:1+n_econ, 1:1+n_econ]; HEE_prod = H_prod[1:1+n_econ, 1:1+n_econ]
    HEC_ad = H_ad[1:1+n_econ, 2+n_econ:1+n_econ+ncm_cm]; HEC_prod = H_prod[1:1+n_econ, 2+n_econ:1+n_econ+ncm_cm]
    HEF_ad = H_ad[1:1+n_econ, 2+n_econ+ncm_cm:end]; HEF_prod = H_prod[1:1+n_econ, 2+n_econ+ncm_cm:end]
    HCC_ad = H_ad[2+n_econ:1+n_econ+ncm_cm, 2+n_econ:1+n_econ+ncm_cm]; HCC_prod = H_prod[2+n_econ:1+n_econ+ncm_cm, 2+n_econ:1+n_econ+ncm_cm]
    HCF_ad = H_ad[2+n_econ:1+n_econ+ncm_cm, 2+n_econ+ncm_cm:end]; HCF_prod = H_prod[2+n_econ:1+n_econ+ncm_cm, 2+n_econ+ncm_cm:end]
    HFF_ad = H_ad[2+n_econ+ncm_cm:end, 2+n_econ+ncm_cm:end]; HFF_prod = H_prod[2+n_econ+ncm_cm:end, 2+n_econ+ncm_cm:end]
    @printf("    H_EE (zeta+econ):   max|Δ|=%.3e\n", maximum(abs.(HEE_ad .- HEE_prod)))
    @printf("    H_EC (econ x CM):   max|Δ|=%.3e\n", maximum(abs.(HEC_ad .- HEC_prod)))
    @printf("    H_EF (econ x level): max|Δ|=%.3e\n", maximum(abs.(HEF_ad .- HEF_prod)))
    @printf("    H_CC (CM x CM):     max|Δ|=%.3e\n", maximum(abs.(HCC_ad .- HCC_prod)))
    @printf("    H_CF (CM x level):  max|Δ|=%.3e\n", maximum(abs.(HCF_ad .- HCF_prod)))
    @printf("    H_FF (level x level): max|Δ|=%.3e\n", maximum(abs.(HFF_ad .- HFF_prod)))
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
