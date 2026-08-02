# Phase 9 gate (integration/profiled-all-five-production-closeout, 2026-08-02): common Fréchet's
# own dense-truth diagnostic Hessian audit, requested explicitly by the task ("complete Hessian vs
# diagnostic dense truth G'*Diagonal(S)*G ... build the equivalent for flexible_CM/common_Frechet if
# missing, reusing the SAME dense-truth construction pattern"). Mirrors
# test_cmzc_hessian_symmetry_audit_2026-08-02.jl's `dense_truth_gram` recipe exactly (scale G's
# columns by sqrt(curvature weights), single BLAS gemm), applied to common Fréchet's reduced
# [zeta | economic | CM-grid | level] Hessian. Reuses test_profiled_reduced_frechet_autodiff_
# 2026-08-02.jl's own G_econ/G_cm/G_level construction (already-validated forward kernels),
# ADDITIVE ONLY -- does not modify that file.
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

function dense_truth_gram(Gfull::AbstractMatrix{Float64}, S::AbstractVector{Float64}, M)
    Gs = Gfull .* sqrt.(S)
    n = size(Gfull, 2)
    Hd = Matrix{Float64}(undef, n, n)
    BLAS.gemm!('T', 'N', 1.0 / M, Gs, Gs, 0.0, Hd)
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
M = obj_probe.M
@printf("  n_econ=%d  ncm_cm=%d  ncm_level=%d  n_total=%d  W=%d\n", n_econ, ncm_cm, ncm_level, n, W)

println("="^90); println("Materializing D4-scale reference G_econ/G_cm/G_level (test-only, validated-kernel method)"); println("="^90)
G_econ = Matrix{Float64}(undef, W, n_econ)
e_econ = zeros(n_econ)
for j in 1:n_econ
    e_econ[j] = 1.0
    G_econ[:, j] .= reduced_homogeneous_dual_contraction(e_econ, cf, ctx, θ_full_calib, layout)
    e_econ[j] = 0.0
end
G_cm = Matrix{Float64}(undef, W, ncm_cm)
bins_int = Int.(cctx_probe.Bidx)
fill_cm_columns_from_bins!(G_cm, bins_int, cctx_probe.origins, cctx_probe.refIndex1, cctx_probe.L, cctx_probe.R)

G_level = Matrix{Float64}(undef, W, ncm_level)
P_level = zeros(ncm_level + 1)
e_level = zeros(ncm_level)
level_contrib = zeros(W)
bins_u = cctx_probe.Bidx isa Matrix{UInt32} ? cctx_probe.Bidx : Matrix{UInt32}(cctx_probe.Bidx)
for l in 1:ncm_level
    e_level[l] = 1.0
    frechet_level_suffix_sums!(P_level, e_level)
    frechet_level_forward_sum!(level_contrib, bins_u, D, P_level)
    G_level[:, l] .= (1.0 / sqrt(D)) .* level_contrib .- level_targets[l]
    e_level[l] = 0.0
end
check("reference matrices finite", all(isfinite, G_econ) && all(isfinite, G_cm) && all(isfinite, G_level))
Gfull = hcat(-ones(W), -G_econ, -G_cm, -G_level)

psi_scalar(q) = q <= 1.0 ? exp(q) - 1.0 : 0.5 * exp(1) * (q^2 + 1.0) - 1.0
function f_ref(x)
    ζ = x[1]
    β = @view x[2:1+n_econ]
    λcm = @view x[2+n_econ:1+n_econ+ncm_cm]
    λlvl = @view x[2+n_econ+ncm_cm:1+n_econ+ncm_cm+ncm_level]
    q = (-ζ) .- G_econ * β .- G_cm * λcm .- G_level * λlvl
    return ζ + sum(psi_scalar, q) / M
end

rE = 1:1+n_econ; rC = 2+n_econ:1+n_econ+ncm_cm; rF = 2+n_econ+ncm_cm:n
ranges = Dict("EE" => (rE, rE), "EC" => (rE, rC), "EF" => (rE, rF), "CC" => (rC, rC), "CF" => (rC, rF), "FF" => (rF, rF))
function block_report(label, Diff)
    for (name, r) in ranges
        rr, rc = r
        d = @view Diff[rr, rc]
        @printf("    %-8s (%s): max|Δ|=%.3e\n", name, label, isempty(d) ? 0.0 : maximum(d))
    end
end

Random.seed!(2027)
NPTS = 4
for pt in 1:NPTS
    x0 = pt == 1 ? 0.01 .* randn(n) : 0.05 .* randn(n)
    println("-"^90)
    @printf("Point %d/%d\n", pt, NPTS)

    dual_index!(st_probe, x0)
    obj_probe.arg0 .= st_probe.arg0
    frechet_ext = _resolve_frechet_ext!(cctx_probe, level_targets)
    h_packed = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured!(h_packed, obj_probe, cctx_probe, frechet_ext)
    H_prod = unpack_packed(h_packed, n)

    check("pt $pt: production Hessian finite", all(isfinite, H_prod))
    check("pt $pt: production Hessian symmetric (<1e-10)", maximum(abs.(H_prod .- H_prod')) < 1e-10)

    H_ad = ForwardDiff.hessian(f_ref, x0)
    S = copy(obj_probe.arg2)
    H_truth = dense_truth_gram(Gfull, S, M)

    Diff_ad = abs.(H_ad .- H_prod)
    Diff_truth = abs.(H_truth .- H_prod)
    scale_h = max(1.0, maximum(abs.(H_ad)))
    @printf("  vs ForwardDiff:  max|Δ|=%.3e  max_rel=%.3e\n", maximum(Diff_ad), maximum(Diff_ad) / scale_h)
    @printf("  vs dense-truth:  max|Δ|=%.3e  max_rel=%.3e\n", maximum(Diff_truth), maximum(Diff_truth) / scale_h)
    println("    block max|Δ| vs ForwardDiff:"); block_report("FD", Diff_ad)
    println("    block max|Δ| vs dense-truth:"); block_report("truth", Diff_truth)
    check("pt $pt: production Hessian matches ForwardDiff (<1e-7 abs)", maximum(Diff_ad) < 1e-7)
    check("pt $pt: production Hessian matches dense-truth G'Diag(S)G (<1e-7 abs)", maximum(Diff_truth) < 1e-7)
    check("pt $pt: ForwardDiff matches dense-truth directly (<1e-7 abs, sanity)", maximum(abs.(H_ad .- H_truth)) < 1e-7)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
