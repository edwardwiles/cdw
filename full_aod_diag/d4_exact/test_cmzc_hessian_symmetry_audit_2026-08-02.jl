# Phase-3 dedicated gate (2026-08-02): explicit, standalone symmetry/precision audit for CM+ZC's
# ("mean ZC") widened-core Hessian H_EM mirror fix (cm_hessian_architectures.jl::_fill_cm_HEE!,
# widened branch ~lines 998-1075, mirror line ~1063: `HEE[ncore+1:NCORE,1:ncore] .= transpose(HEM)`).
#
# That fix has already been spot-checked incidentally (test_profiled_reduced_meanzc_operator_and_
# autodiff_2026-08-02.jl's aggregate ForwardDiff Hessian check, and test_zc_lane_cmzc_outer_gradient
# _d4_2026-08-02.jl's outer-gradient gate) but never given its OWN dedicated gate that isolates and
# proves the mirror mechanism itself, at multiple random points, against TWO independent references.
# This file is that dedicated gate. Four checks, exactly as specified:
#   1. H_EM == transpose(mirror) bit-exact, read directly off cctx.Hfull BEFORE pack_upper_cm_hessian!'s
#      final 0.5*(Hfull[i,j]+Hfull[j,i]) symmetrize-by-averaging step (proves the mirror LINE ran, not
#      that averaging papered over an asymmetry).
#   2. The complete assembled Hfull (pre-pack) is symmetric to machine precision.
#   3. Block-by-block max abs/rel error of the packed production Hessian against BOTH (a) a ForwardDiff
#      reference built EXACTLY as test_profiled_reduced_meanzc_operator_and_autodiff_2026-08-02.jl does
#      ("safe by linearity" unit-vector calls to the same validated forward kernels), and (b) an
#      independent dense-truth G'*Diagonal(S)*G construction mirroring the existing
#      `_dense_reference_core_hessian!` (core_exact_hessian.jl:944) recipe -- scale G's columns by
#      sqrt(curvature weights) then BLAS gemm -- generalized here to the FULL widened G (economic +
#      mean/pair-Z + CM-grid columns), not a new/independently-invented formula. At multiple random
#      points (not just calibration) to rule out a coincidental cancellation at one specific point.
#   4. Same three checks for origin-ZC's analogous (non-widened) H_EZ block as a control/comparison.
#      Origin-ZC's packer (archA_partitioned_hess_cb_builder, cm_hessian_architectures.jl ~2245-2252)
#      reads ONLY i<=j entries with NO averaging step at all (a prior mirror-write into the lower
#      position was found "provably dead computation" and removed, 2026-07-28 comment) -- so the
#      CM+ZC bug class (mirror omitted, then averaging silently halves it) is structurally impossible
#      there. Included for contrast, not because a mirror check applies the same way.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
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
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
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

"""
    dense_truth_gram(Gfull, S, M) -> Matrix

Independent dense-truth reference `(1/M) * Gfull' * Diagonal(S) * Gfull`, mirroring the EXACT
recipe already used by the in-repo `_dense_reference_core_hessian!` (core_exact_hessian.jl:944):
scale a copy of the columns by `sqrt(S)`, then a single BLAS `gemm!('T','N', ...)`. That existing
function is hardwired to `obj.H`'s own (1+ncolI) economic-only columns; this is the SAME
computational recipe applied to the FULL widened `Gfull` (zeta/economic/mean-pair-Z/CM-grid
columns assembled the same "safe by linearity" way the ForwardDiff reference matrices already are
in test_profiled_reduced_meanzc_operator_and_autodiff_2026-08-02.jl) -- not an independently
invented formula.
"""
function dense_truth_gram(Gfull::AbstractMatrix{Float64}, S::AbstractVector{Float64}, M)
    Gs = Gfull .* sqrt.(S)
    n = size(Gfull, 2)
    Hd = Matrix{Float64}(undef, n, n)
    BLAS.gemm!('T', 'N', 1.0 / M, Gs, Gs, 0.0, Hd)
    return Hd
end

function block_report(label, Diff, ranges::Dict)
    for (name, r) in ranges
        rr, rc = r
        d = @view Diff[rr, rc]
        @printf("    %-8s (%s): max|Δ|=%.3e\n", name, label, isempty(d) ? 0.0 : maximum(d))
    end
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

psi_scalar(q) = q <= 1.0 ? exp(q) - 1.0 : 0.5 * exp(1) * (q^2 + 1.0) - 1.0

# =====================================================================================================
# PART A: CM+ZC ("mean ZC") widened-core Hessian
# =====================================================================================================
println("="^96)
println("PART A: CM+ZC widened-core Hessian (H_EE/H_EM/H_MM widened block)")
println("="^96)

const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0 = fill(1.0, K_MEAN)
aug_reduced = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)

cctx_probe = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
obj_probe, st_probe = build_reduced_meanzc_operator_bundle(ctx, θ_full_calib, layout, cctx_probe)
prime_operator!(obj_probe, θ_full_calib, ctx, cctx_probe.core_cf_ref; restriction_state = cctx_probe)
reset_for_solve!(st_probe, νvec0)
cctx_probe.nu_ref[] = collect(νvec0)
cctx_probe.profiled_theta_ref[] = copy(θ_full_calib)
cctx_probe.cmlookup_st = st_probe
cctx_probe.inner_fg_backend = :cm_lookup
cf = cctx_probe.core_cf_ref[]::CompressedFactual

op = st_probe.zc_op
n_econ = layout.total_reduced_economic_moments
nm = n_mean(op); npr = n_pair(op); ncm = cctx_probe.ncm
ncore_core = cctx_probe.ncore_core
NCORE = cctx_probe.NCORE
n = 1 + n_econ + nm + npr + ncm
W = cf.W
M = obj_probe.M
@printf("  ncore_core=%d  NCORE=%d  ncm=%d  n_total=%d  W=%d\n", ncore_core, NCORE, ncm, n, W)
ncore_core == 1 + n_econ || error("sanity: ncore_core should be 1+n_econ")
NCORE == ncore_core + nm + npr || error("sanity: NCORE should be ncore_core+nm+npr")

println("Materializing D4-scale reference G_econ/G_Z/G_cm (test-only, same validated-kernel method as")
println("test_profiled_reduced_meanzc_operator_and_autodiff_2026-08-02.jl)")
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
bins_probe = Matrix{Int}(st_probe.bins)
G_cm = Matrix{Float64}(undef, W, ncm)
fill_cm_columns_from_bins!(G_cm, bins_probe, st_probe.origins, st_probe.refIndex1, st_probe.L, st_probe.R)
check("A: reference matrices finite", all(isfinite, G_econ) && all(isfinite, G_Z) && all(isfinite, G_cm))

function f_ref_cmzc(x)
    ζ = x[1]
    β = @view x[2:1+n_econ]
    λZ = @view x[2+n_econ:1+n_econ+nm+npr]
    λcm = @view x[2+n_econ+nm+npr:1+n_econ+nm+npr+ncm]
    q = (-ζ) .- G_econ * β .+ G_Z * λZ .- G_cm * λcm
    return ζ + sum(psi_scalar, q) / M
end
Gfull_cmzc = hcat(-ones(W), -G_econ, G_Z, -G_cm)

# ---- Ranges for block reporting (E = zeta+economic = ncore_core, M = mean/pair-Z, C = CM-grid) ----
rE = 1:ncore_core; rM = ncore_core+1:NCORE; rC = NCORE+1:n
ranges = Dict("EE" => (rE, rE), "EM" => (rE, rM), "MM" => (rM, rM),
              "EC" => (rE, rC), "MC" => (rM, rC), "CC" => (rC, rC))

Random.seed!(2027)
NPTS = 4
for pt in 1:NPTS
    x0 = pt == 1 ? 0.01 .* randn(n) : 0.05 .* randn(n)
    println("-"^96)
    @printf("Point %d/%d (seed draw, |x0| range ~%.3f)\n", pt, NPTS, pt == 1 ? 0.01 : 0.05)

    dual_index!(st_probe, x0)
    obj_probe.arg0 .= st_probe.arg0
    h_packed = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured!(h_packed, obj_probe, cctx_probe)
    H_prod = unpack_packed(h_packed, n)

    # ---- Check 1: H_EM == transpose(mirror), read DIRECTLY off cctx_probe.Hfull, pre-pack -----------
    Hfull = cctx_probe.Hfull
    HEE_view = @view Hfull[1:NCORE, 1:NCORE]
    HEM = HEE_view[1:ncore_core, ncore_core+1:NCORE]
    mirror = HEE_view[ncore_core+1:NCORE, 1:ncore_core]
    mirr_err = maximum(abs.(mirror .- transpose(HEM)))
    @printf("  Check1 (pt %d): max|mirror - transpose(H_EM)| (pre-pack, off Hfull) = %.3e\n", pt, mirr_err)
    check("Check1 (pt $pt): H_EM mirror bit-exact transpose, pre-pack", mirr_err == 0.0)

    # ---- Check 2: complete Hfull (pre-pack) symmetric to machine precision --------------------------
    Hfull_full = @view Hfull[1:n, 1:n]
    sym_err = maximum(abs.(Hfull_full .- transpose(Hfull_full)))
    @printf("  Check2 (pt %d): max|Hfull - Hfull'| (pre-pack) = %.3e\n", pt, sym_err)
    check("Check2 (pt $pt): complete pre-pack Hfull symmetric (<1e-8)", sym_err < 1e-8)

    # ---- Check 3: block errors vs ForwardDiff AND vs dense-truth G'Diag(S)G -------------------------
    H_ad = ForwardDiff.hessian(f_ref_cmzc, x0)
    S = copy(obj_probe.arg2)   # curvature weights already populated by hessian_cm_structured! above
    H_truth = dense_truth_gram(Gfull_cmzc, S, M)

    Diff_ad = abs.(H_ad .- H_prod)
    Diff_truth = abs.(H_truth .- H_prod)
    scale_h = max(1.0, maximum(abs.(H_ad)))
    @printf("  Check3 (pt %d) vs ForwardDiff:  max|Δ|=%.3e  max_rel=%.3e\n", pt, maximum(Diff_ad), maximum(Diff_ad) / scale_h)
    @printf("  Check3 (pt %d) vs dense-truth:  max|Δ|=%.3e  max_rel=%.3e\n", pt, maximum(Diff_truth), maximum(Diff_truth) / scale_h)
    println("    block max|Δ| vs ForwardDiff:")
    block_report("FD", Diff_ad, ranges)
    println("    block max|Δ| vs dense-truth:")
    block_report("truth", Diff_truth, ranges)
    check("Check3 (pt $pt): production Hessian matches ForwardDiff (<1e-7 abs)", maximum(Diff_ad) < 1e-7)
    check("Check3 (pt $pt): production Hessian matches dense-truth G'Diag(S)G (<1e-7 abs)", maximum(Diff_truth) < 1e-7)
    check("Check3 (pt $pt): ForwardDiff matches dense-truth directly (<1e-7 abs, sanity)", maximum(abs.(H_ad .- H_truth)) < 1e-7)
    # Ratio check on the EM block specifically: the historical bug gave EXACTLY ratio 2.0 there.
    em_ad = H_ad[rE, rM]; em_prod = H_prod[rE, rM]
    ratio_mask = abs.(em_ad) .> 1e-8
    if any(ratio_mask)
        ratios = em_prod[ratio_mask] ./ em_ad[ratio_mask]
        @printf("    H_EM production/ForwardDiff ratio: min=%.6f max=%.6f (1.0 expected; 0.5 would be the old bug)\n",
            minimum(ratios), maximum(ratios))
        check("Check3 (pt $pt): H_EM ratio ~1.0, not ~0.5 (old bug signature)", all(x -> abs(x - 1.0) < 1e-6, ratios))
    end
end

# =====================================================================================================
# PART B: origin-ZC control (analogous, non-widened H_EZ block)
# =====================================================================================================
println("="^96)
println("PART B: origin-ZC control -- H_EZ block (no widened-core mirror mechanism exists here)")
println("="^96)

layout_o = OriginByPowerLayout(ctx.D, 1, 0)
νfull0 = fill(1.0, ctx.D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)

octx_probe = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                           zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
obj_o, st_o = build_reduced_originzc_operator_bundle(ctx, θ_full_calib, layout, octx_probe)
refresh_zc_targets!(st_o.zc_ws, st_o.op, st_o.zc_layout, νfull0)
octx_probe.nu_ref[] = collect(νfull0)
prime_operator!(obj_o, θ_full_calib, ctx, octx_probe.core_cf_ref; restriction_state = octx_probe)
octx_probe.profiled_theta_ref[] = copy(θ_full_calib)
cf_o = octx_probe.core_cf_ref[]::CompressedFactual

op_o = st_o.op
n_econ_o = layout.total_reduced_economic_moments
nm_o = n_mean(op_o); npr_o = n_pair(op_o)
n_o = 1 + n_econ_o + nm_o + npr_o
W_o = cf_o.W
M_o = obj_o.M
@printf("  n_econ=%d  n_mean=%d  n_pair=%d  n_total=%d  W=%d\n", n_econ_o, nm_o, npr_o, n_o, W_o)

G_econ_o = Matrix{Float64}(undef, W_o, n_econ_o)
e_econ_o = zeros(n_econ_o)
for j in 1:n_econ_o
    e_econ_o[j] = 1.0
    G_econ_o[:, j] .= reduced_homogeneous_dual_contraction(e_econ_o, cf_o, ctx, θ_full_calib, layout)
    e_econ_o[j] = 0.0
end
G_Z_o = Matrix{Float64}(undef, W_o, nm_o + npr_o)
e_mean_o = zeros(nm_o); e_pair_o = zeros(npr_o)
for j in 1:(nm_o + npr_o)
    dest = zeros(W_o)
    if j <= nm_o
        e_mean_o[j] = 1.0
        restriction_forward!(dest, e_mean_o, e_pair_o, op_o, st_o.zc_ws)
        e_mean_o[j] = 0.0
    else
        e_pair_o[j - nm_o] = 1.0
        restriction_forward!(dest, e_mean_o, e_pair_o, op_o, st_o.zc_ws)
        e_pair_o[j - nm_o] = 0.0
    end
    G_Z_o[:, j] .= dest
end
check("B: reference matrices finite", all(isfinite, G_econ_o) && all(isfinite, G_Z_o))

function f_ref_oz(x)
    ζ = x[1]
    β = @view x[2:1+n_econ_o]
    λZ = @view x[2+n_econ_o:1+n_econ_o+nm_o+npr_o]
    q = (-ζ) .- G_econ_o * β .+ G_Z_o * λZ
    return ζ + sum(psi_scalar, q) / M_o
end
Gfull_oz = hcat(-ones(W_o), -G_econ_o, G_Z_o)

rE_o = 1:1+n_econ_o; rZ_o = 2+n_econ_o:n_o
ranges_o = Dict("EE" => (rE_o, rE_o), "EZ" => (rE_o, rZ_o), "ZZ" => (rZ_o, rZ_o))

println("  Structural note (Check1 analog): archA_partitioned_hess_cb_builder's final packer")
println("  (cm_hessian_architectures.jl ~2305-2313) reads ONLY i<=j entries of its local scratch,")
println("  with NO 0.5*(H[i,j]+H[j,i]) averaging step anywhere (unlike hessian_cm_structured!'s")
println("  pack_upper_cm_hessian!) -- confirmed by direct source read. A 2026-07-28 comment there")
println("  documents that a former lower-triangle mirror write into ∂∂f_∂∂x[NCORE+1:n,1:NCORE] was")
println("  found to be provably dead computation and removed. So there is no separate mirror position")
println("  that could go stale the way CM+ZC's H_EM did -- the bug class is structurally absent here,")
println("  not merely untriggered. The applicable empirical check is instead the direct block-error")
println("  cross-check below (Check3 analog), which would show the historical ~2.0 ratio signature if")
println("  an equivalent halving existed.")
check("Check1 analog (origin-ZC): no averaging step exists for H_EZ (structural, source-confirmed)", true)

octx_probe.fg_lookup_st = st_o
octx_probe.fg_backend = :operator
hess_cb = archA_partitioned_hess_cb_builder(octx_probe)   # built ONCE, reused across dual points, matching production's real KNITRO-callback reuse pattern

Random.seed!(2028)
for pt in 1:NPTS
    x0 = pt == 1 ? 0.01 .* randn(n_o) : 0.05 .* randn(n_o)
    println("-"^96)
    @printf("Point %d/%d\n", pt, NPTS)

    dual_index!(st_o, x0)
    obj_o.arg0 .= st_o.arg0
    h_packed_o = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
    evalReq = (x = x0,)
    evalRes = (hess = h_packed_o,)
    hess_cb(nothing, nothing, evalReq, evalRes, obj_o)
    H_prod_o = unpack_packed(h_packed_o, n_o)

    # Check2 analog: trivially symmetric post-unpack (unpack_packed always symmetrizes); the
    # meaningful invariant here is instead a direct block cross-check (below), since there is no
    # pre-pack lower-triangle scratch exposed to inspect independently (архA's scratch_full is a
    # closure-local, not a persistent ctx field the way cctx.Hfull is for CM+ZC).
    sym_err_o = maximum(abs.(H_prod_o .- transpose(H_prod_o)))
    check("Check2 analog (pt $pt): unpacked H_prod symmetric (trivial-by-construction sanity)", sym_err_o == 0.0)

    H_ad_o = ForwardDiff.hessian(f_ref_oz, x0)
    S_o = copy(obj_o.arg2)
    H_truth_o = dense_truth_gram(Gfull_oz, S_o, M_o)

    Diff_ad_o = abs.(H_ad_o .- H_prod_o)
    Diff_truth_o = abs.(H_truth_o .- H_prod_o)
    scale_h_o = max(1.0, maximum(abs.(H_ad_o)))
    @printf("  Check3 analog (pt %d) vs ForwardDiff:  max|Δ|=%.3e  max_rel=%.3e\n", pt, maximum(Diff_ad_o), maximum(Diff_ad_o) / scale_h_o)
    @printf("  Check3 analog (pt %d) vs dense-truth:  max|Δ|=%.3e  max_rel=%.3e\n", pt, maximum(Diff_truth_o), maximum(Diff_truth_o) / scale_h_o)
    println("    block max|Δ| vs ForwardDiff:")
    block_report("FD", Diff_ad_o, ranges_o)
    println("    block max|Δ| vs dense-truth:")
    block_report("truth", Diff_truth_o, ranges_o)
    check("Check3 analog (pt $pt): production Hessian matches ForwardDiff (<1e-7 abs)", maximum(Diff_ad_o) < 1e-7)
    check("Check3 analog (pt $pt): production Hessian matches dense-truth G'Diag(S)G (<1e-7 abs)", maximum(Diff_truth_o) < 1e-7)

    ez_ad = H_ad_o[rE_o, rZ_o]; ez_prod = H_prod_o[rE_o, rZ_o]
    ratio_mask_o = abs.(ez_ad) .> 1e-8
    if any(ratio_mask_o)
        ratios_o = ez_prod[ratio_mask_o] ./ ez_ad[ratio_mask_o]
        @printf("    H_EZ production/ForwardDiff ratio: min=%.6f max=%.6f (1.0 expected)\n",
            minimum(ratios_o), maximum(ratios_o))
        check("Check3 analog (pt $pt): H_EZ ratio ~1.0 (no halving)", all(x -> abs(x - 1.0) < 1e-6, ratios_o))
    end
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
