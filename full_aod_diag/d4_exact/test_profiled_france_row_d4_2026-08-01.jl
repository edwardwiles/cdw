# Profiled economic block port (2026-08-01), follow-up: D4 gate for the France/cf row's
# use_profiled_correction=true path, now enabled via build_winner_pair_ctx's new bi_slot keyword
# (core_exact_hessian.jl). Previously an explicit, documented scope boundary in every amended
# cross-block function (the cf row always fell back to the OLD destination-independent correction).
#
# Covers H_EC (winner_pair_cross_hessian_cm_block!) and H_EF (colsum!/esum!) via the flexible-CM and
# common-Frechet D4 fixtures already proven in test_profiled_hec_correction_d4_2026-08-01.jl /
# test_profiled_hef_correction_d4_2026-08-01.jl. H_EZ's serial cf row is covered directly here too;
# its threaded twin is covered for free by re-running test_profiled_hez_threaded_d4_2026-08-01.jl
# with bi_slot now supplied (that test already compares the FULL matrix, row jcf+1 included).
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
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
bi_slot = dest_slot(ctx, ctx.bi)
println("ctx.bi=", ctx.bi, "  bi_slot=", bi_slot)
Random.seed!(2026)

# ============================================================================
# H_EC (flexible CM) -- France row via winner_pair_cross_hessian_cm_block!
# ============================================================================
println("="^80); println("H_EC France row"); println("="^80)
pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, use_compressed_core = true,
                                   moment_representation = :dense_reference)
cctx = pcx.cctx
obj = pcx.ctx_cm.obj
base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx)
check("H_EC: inner solve feasible", base.inner_status in (0, -100, -101, -103))
x = vcat(base.ζstar, base.λstar) .+ vcat(0.01, 0.02 .* randn(length(base.λstar)))
_archC_prep_for_hessian!(obj, x)
cf = cctx.core_cf_ref[]
check("H_EC: cf is a real CompressedFactual, has_cf", cf isa CompressedFactual && cf.cf_col > 0)
wctx = build_winner_pair_ctx(cf; bi_slot = bi_slot)
check("H_EC: wctx.target_slot[cf.cf_col] == bi_slot", wctx.target_slot[cf.cf_col] == bi_slot)
ws_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
ws = ensure_winner_bin_cross_scratch!(ws_ref, wctx.ncolI, cctx.D, cctx.L, wctx.Ddest)
winner_pair_cross_hessian_fill!(wctx, ws, obj, cctx.Bidx)
S = obj.arg2
M = obj.M
jcf = cf.cf_col
maxdiff_hec = 0.0
for l in (1, cctx.L)
    Hraw_EC = zeros(wctx.ncolI + 1, cctx.nO)
    winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, ws, l, cctx.origins, cctx.refIndex1, M; use_profiled_correction = true)
    for (oi, o_col) in enumerate(cctx.origins)
        # brute force: keep term is cf.cf_raw_scaled-weighted, unconditional on winner (mirrors
        # QCfCScum's own accumulation, NOT winner-conditioned like the bilateral columns)
        qcf_o = 0.0; qcf_ref = 0.0
        Tbi_o = 0.0; Tbi_ref = 0.0
        for w in 1:cf.W
            snu = S[w] * cf.SW[w]
            in_o = cctx.Bidx[w, o_col] <= l
            in_ref = cctx.Bidx[w, cctx.refIndex1] <= l
            v = snu * wctx.kappa0[jcf] * cf.cf_raw[w]
            in_o && (qcf_o += v)
            in_ref && (qcf_ref += v)
            m = snu * cf.wval[w, bi_slot]
            in_o && (Tbi_o += m)
            in_ref && (Tbi_ref += m)
        end
        qcf_diff = qcf_o - qcf_ref
        Tbi_diff = Tbi_o - Tbi_ref
        expected = (qcf_diff - wctx.pi_vec[jcf] * Tbi_diff) * (1.0 / M)
        got = Hraw_EC[jcf + 1, oi]
        global maxdiff_hec = max(maxdiff_hec, abs(got - expected))
    end
end
check("H_EC: France row use_profiled_correction=true matches independent brute-force T^R_bi (max|Δ|=$(maxdiff_hec))",
      maxdiff_hec < 1e-6)
@printf("  H_EC France row max|Δ|=%.3e\n", maxdiff_hec)

# ============================================================================
# H_EZ (CM+ZC) -- France row via winner_pair_cross_hessian_zc_block!
# ============================================================================
println("="^80); println("H_EZ France row"); println("="^80)
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = 1, K_pair = 1, meanzc_basis = :direct)
objA = aug.obj_cm
n = objA.outer_constr_index
θ_ext_calib = vcat(θ_full_calib, nu0vec(1))
Kv = zeros(size(ctx.U, 1))
objA.moments!(Kv, CS.select_G_from_H(objA, objA.H), θ_ext_calib, objA.U, objA)
objA.H[:, 1] .= Kv
objA.H[:, 2] .= 1.0
cctx_z = build_cm_meanzc_bin_ctx(ctx, aug)
ncore = cctx_z.ncore_core; NCORE_ext = cctx_z.NCORE
n_restr = NCORE_ext - ncore
Random.seed!(9101)
xz = 0.01 .* randn(n)
_archC_prep_for_hessian!(objA, xz)
cfz = cctx_z.core_cf_ref[]
check("H_EZ: cf is a real CompressedFactual, has_cf", cfz isa CompressedFactual && cfz.cf_col > 0)
Hz = objA.H; Mz = objA.M
ddPsi! = objA.ddPsi!; ddPsi!(objA.arg2, objA.arg0); Sz = objA.arg2
Ez = @view Hz[:, 2:1+NCORE_ext]
Zz = @view Ez[:, ncore+1:NCORE_ext]
wctxz = build_winner_pair_ctx(cfz; bi_slot = bi_slot)
check("H_EZ: wctx.target_slot[cf.cf_col] == bi_slot", wctxz.target_slot[cfz.cf_col] == bi_slot)
wsz_ref = Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing)
wsz = ensure_winner_zc_cross_scratch!(wsz_ref, wctxz.W, n_restr, wctxz.Ddest)
winner_pair_cross_hessian_zc_prep!(wsz, wctxz, Sz)
H_new = zeros(ncore, n_restr)
winner_pair_cross_hessian_zc_block!(H_new, wctxz, wsz, Sz, Zz, Mz; use_profiled_correction = true)
jcfz = cfz.cf_col
maxdiff_hez = 0.0
for x in 1:n_restr
    row_bf = 0.0
    Tbi_bf = 0.0
    for w in 1:cfz.W
        snu = Sz[w] * cfz.SW[w]
        row_bf += snu * wctxz.kappa0[jcfz] * cfz.cf_raw[w] * Zz[w, x]
        Tbi_bf += snu * cfz.wval[w, bi_slot] * Zz[w, x]
    end
    expected = (row_bf - wctxz.pi_vec[jcfz] * Tbi_bf) / Mz
    got = H_new[jcfz + 1, x]
    global maxdiff_hez = max(maxdiff_hez, abs(got - expected))
end
check("H_EZ: France row use_profiled_correction=true matches independent brute-force T^Z_bi (max|Δ|=$(maxdiff_hez))",
      maxdiff_hez < 1e-6)
@printf("  H_EZ France row max|Δ|=%.3e\n", maxdiff_hez)

# regression: use_profiled_correction=false still matches the dense reference exactly (unaffected by bi_slot)
H_old = zeros(ncore, n_restr)
winner_pair_cross_hessian_zc_block!(H_old, wctxz, wsz, Sz, Zz, Mz; use_profiled_correction = false)
H_dense = ((@view(Ez[:, 1:ncore])) .* Sz)' * Zz ./ Mz
maxdiff_hez_old = maximum(abs.(H_dense .- H_old))
check("H_EZ: use_profiled_correction=false still regression-safe with bi_slot supplied (max|Δ|=$(maxdiff_hez_old))",
      maxdiff_hez_old < 1e-8)

# ============================================================================
# H_EF (common Frechet) -- France row via colsum!/esum!
# ============================================================================
println("="^80); println("H_EF France row"); println("="^80)
pcxf = build_cm_frechet_production_context(ctx, CS; L = 10, contrasts = :anchored, use_compressed_core = true,
    cm_hessian_backend = :structured, cm_cross_hessian_backend = :winner_bin, moment_representation = :dense_reference)
cctxf = pcxf.cctx
objf = pcxf.ctx_cm.obj
basef = archC_frechet_base_state(x_free_calib, pcxf.ctx_cm, cctxf, pcxf.aug.level_targets)
check("H_EF: inner solve feasible", basef.inner_status in (0, -100, -101, -103))
xf = vcat(basef.ζstar, basef.λstar) .+ vcat(0.01, 0.02 .* randn(length(basef.λstar)))
_archC_prep_for_hessian!(objf, xf)
cff = cctxf.core_cf_ref[]
check("H_EF: cf is a real CompressedFactual, has_cf", cff isa CompressedFactual && cff.cf_col > 0)
wctxf = build_winner_pair_ctx(cff; bi_slot = bi_slot)
check("H_EF: wctx.target_slot[cf.cf_col] == bi_slot", wctxf.target_slot[cff.cf_col] == bi_slot)
wsf_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
wsf = ensure_winner_bin_cross_scratch!(wsf_ref, wctxf.ncolI, cctxf.D, cctxf.L, wctxf.Ddest)
winner_pair_cross_hessian_fill!(wctxf, wsf, objf, cctxf.Bidx)
Sf = objf.arg2
Mf = objf.M
jcff = cff.cf_col
maxdiff_colsum = 0.0
for l in (1, cctxf.L)
    colsum_new = zeros(wctxf.ncolI + 1)
    winner_pair_cross_hessian_colsum!(colsum_new, wctxf, wsf, l; use_profiled_correction = true)
    sumQCf_bf = 0.0
    Tbi_bf = 0.0
    for x_ in 1:cctxf.D
        for w in 1:cff.W
            cctxf.Bidx[w, x_] <= l || continue
            snu = Sf[w] * cff.SW[w]
            sumQCf_bf += snu * wctxf.kappa0[jcff] * cff.cf_raw[w]
            Tbi_bf += snu * cff.wval[w, bi_slot]
        end
    end
    expected = sumQCf_bf - wctxf.pi_vec[jcff] * Tbi_bf
    global maxdiff_colsum = max(maxdiff_colsum, abs(colsum_new[jcff + 1] - expected))
end
check("H_EF: France row colsum! use_profiled_correction=true matches brute-force (max|Δ|=$(maxdiff_colsum))",
      maxdiff_colsum < 1e-6)

Wtot = sum(Sf)
Esum_new = zeros(wctxf.ncolI + 1)
winner_pair_cross_hessian_esum!(Esum_new, wctxf, wsf, Sf, Wtot; use_profiled_correction = true)
ecf_bf = 0.0
T0bi_bf = 0.0
for w in 1:cff.W
    snu = Sf[w] * cff.SW[w]
    global ecf_bf += snu * wctxf.kappa0[jcff] * cff.cf_raw[w]
    global T0bi_bf += snu * cff.wval[w, bi_slot]
end
expected_esum = ecf_bf - wctxf.pi_vec[jcff] * T0bi_bf
maxdiff_esum = abs(Esum_new[jcff + 1] - expected_esum)
check("H_EF: France row esum! use_profiled_correction=true matches brute-force (max|Δ|=$(maxdiff_esum))",
      maxdiff_esum < 1e-6)
@printf("  H_EF France row: colsum max|Δ|=%.3e  esum max|Δ|=%.3e\n", maxdiff_colsum, maxdiff_esum)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
