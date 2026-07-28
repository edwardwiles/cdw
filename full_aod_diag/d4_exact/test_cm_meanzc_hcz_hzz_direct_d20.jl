# CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): real D=20/W=80,000/L=50 gate for
# the NEW direct H_CZ (CM-grid x mean/pair-ZC-restriction cross) and shared direct H_ZZ (mean/pair
# self-Gram) Hessian primitives, `cm_cross_hessian_backend=:winner_bin` TOGETHER with
# `zc_cross_hessian_backend=:winner_bin` (the combination that relaxes `_cm_cross_hessian_wants_
# winner_bin`'s `ncore_core==NCORE` guard and fills the widened Z-rows via the NEW
# `bin_zc_cross_hessian_block!`/`zc_restriction_gram!` primitives). Companion to
# test_cm_meanzc_hcz_hzz_direct_d4.jl -- exercises the WIRED backend end-to-end at real production
# scale: complete packed Hessian (serial + threaded_v2), complete real KNITRO inner solve (status +
# dual point), at calibration and a near-delta=1 perturbed point, both contrasts, K_mean=1/K_pair=1
# (Point-A-sized, matching d20_meanzc_release_gates.jl's own Point A config -- SAME config
# test_cm_meanzc_winner_bin_hez_wiring_d20.jl already gates for H_EM alone). Also runs a bare
# K_pair=0 (mean-only) smoke at the end: builds a fresh direct-backend context and confirms a
# complete inner solve + Hessian call succeeds with no crash.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

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

println("Building real D=20 context (W=80000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2027)

reset_no_dense_g_counters!()

const K_mean, K_pair = 1, 1
ν0 = nu0vec(K_mean)

for contrasts in (:anchored, :orthonormal)
    println("=== contrasts=$contrasts L=50 K_mean=$K_mean K_pair=$K_pair ==="); flush(stdout)
    aug_dense = build_cm_meanzc_augmented_obj(ctx, CS; L = 50, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    aug_direct = build_cm_meanzc_augmented_obj(ctx, CS; L = 50, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    ctx_cm_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    ctx_cm_direct = merge(ctx, (obj = aug_direct.obj_cm,))
    cctx_dense = build_cm_meanzc_bin_ctx(ctx, aug_dense; cm_cross_hessian_backend = :dense_reference, zc_cross_hessian_backend = :dense_reference)
    cctx_direct = build_cm_meanzc_bin_ctx(ctx, aug_direct; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    check("contrasts=$contrasts: direct cctx backends set", cctx_direct.cm_cross_hessian_backend === :winner_bin && cctx_direct.zc_cross_hessian_backend === :winner_bin)
    check("contrasts=$contrasts: hzz_zc_op built", cctx_direct.hzz_zc_op !== nothing)

    println("  solving dense-reference base state..."); flush(stdout)
    @time base_dense = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_dense, cctx_dense)
    println("  solving direct base state..."); flush(stdout)
    @time base_direct = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_direct, cctx_direct)
    check("contrasts=$contrasts: dense inner solve feasible (status=$(base_dense.inner_status))", base_dense.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts: direct inner solve feasible (status=$(base_direct.inner_status))", base_direct.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts: complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_direct.inner_status))",
        base_dense.inner_status == base_direct.inner_status)
    dual_diff = maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_direct.ζstar, base_direct.λstar)))
    check("contrasts=$contrasts: complete inner solve dual point matches (max|Δ|=$dual_diff)", dual_diff < 1e-6)

    NCORE = cctx_dense.NCORE; ncm = cctx_dense.ncm; ncore_core = cctx_dense.ncore_core
    n = NCORE + ncm

    for (label, x) in (("calib", vcat(base_dense.ζstar, base_dense.λstar)),
                        ("near_delta1_perturbed", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.005, 0.01 .* randn(length(base_dense.λstar)))))
        for (arch_label, hess_fn) in (("serial", (h, obj, cctx) -> hessian_cm_structured!(h, obj, cctx)),
                                       ("threaded_v2", (h, obj, cctx) -> hessian_cm_structured_v2!(h, obj, cctx; threaded_bins = cctx.use_threaded_bins, tls = cctx.tls, use_syrk = true)))
            obj_d = ctx_cm_dense.obj; obj_r = ctx_cm_direct.obj
            _archC_prep_for_hessian!(obj_d, x); _archC_prep_for_hessian!(obj_r, x)
            hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hr = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            print("    $label $arch_label dense:  "); @time hess_fn(hd, obj_d, cctx_dense)
            print("    $label $arch_label direct: "); @time hess_fn(hr, obj_r, cctx_direct)
            Hd = unpack_packed(hd, n); Hr = unpack_packed(hr, n)
            maxdiff = maximum(abs.(Hd .- Hr))
            relscale = max(1.0, maximum(abs.(Hd)))
            ok = maxdiff < 1e-6 * relscale
            check("contrasts=$contrasts $label $arch_label: complete packed Hessian dense vs direct(H_CZ+H_ZZ) matches (max|Δ|=$(maxdiff))", ok)
            @printf("      max|ΔH|=%.3e  scale=%.3e\n", maxdiff, relscale)

            Zrows = ncore_core+1:NCORE
            zdiff = maximum(abs.(Hd[Zrows, :] .- Hr[Zrows, :]))
            zscale = max(1.0, maximum(abs.(Hd[Zrows, :])))
            zok = zdiff < 1e-6 * zscale
            check("contrasts=$contrasts $label $arch_label: Z-rows (H_EZ|H_CZ|H_ZZ) sub-block matches (max|Δ|=$(zdiff))", zok)
        end
    end

    # persistent-workspace allocation check (threaded v2, the production default path)
    x0 = vcat(base_direct.ζstar, base_direct.λstar)
    _archC_prep_for_hessian!(ctx_cm_direct.obj, x0)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured_v2!(h, ctx_cm_direct.obj, cctx_direct; threaded_bins = cctx_direct.use_threaded_bins, tls = cctx_direct.tls, use_syrk = true)  # warmup
    id_hzz_before = objectid(cctx_direct.hzz_centered)
    id_binzc_before = objectid(cctx_direct.bin_zc_cross)
    bytes = @allocated hessian_cm_structured_v2!(h, ctx_cm_direct.obj, cctx_direct; threaded_bins = cctx_direct.use_threaded_bins, tls = cctx_direct.tls, use_syrk = true)
    id_hzz_after = objectid(cctx_direct.hzz_centered)
    id_binzc_after = objectid(cctx_direct.bin_zc_cross)
    check("contrasts=$contrasts: hzz_centered object identity stable across calls (no resize)", id_hzz_before == id_hzz_after)
    check("contrasts=$contrasts: bin_zc_cross object identity stable across calls (no resize)", id_binzc_before == id_binzc_after)
    @printf("  contrasts=%s: warm hessian_cm_structured_v2! (direct H_CZ/H_ZZ) @allocated=%.3f MB\n", contrasts, bytes / 1e6)
    flush(stdout)
end

println()
println("Winner cross-Hessian counters (before K_pair=0 smoke): ", no_dense_g_report())

# ---- K_pair=0 (mean-only) smoke: status/no-crash check only, both families' widths differ so run
# each family's own small smoke separately below (this file: CM+ZC only; origin-ZC's own smoke is in
# the companion origin-ZC D20 script). ----
println("=== K_pair=0 mean-only smoke (CM+ZC, contrasts=:orthonormal) ==="); flush(stdout)
let K_mean0 = 1, K_pair0 = 0
    ν00 = nu0vec(K_mean0)
    aug0 = build_cm_meanzc_augmented_obj(ctx, CS; L = 50, K_mean = K_mean0, K_pair = K_pair0, contrasts = :orthonormal, meanzc_basis = :direct)
    ctx_cm0 = merge(ctx, (obj = aug0.obj_cm,))
    cctx0 = build_cm_meanzc_bin_ctx(ctx, aug0; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    check("K_pair=0 smoke: hzz_zc_op built", cctx0.hzz_zc_op !== nothing)
    @time base0 = archC_meanzc_base_state(x_free_calib, ν00, ctx_cm0, cctx0)
    check("K_pair=0 smoke: inner solve feasible (status=$(base0.inner_status))", base0.inner_status in (0, -100, -101, -103))
    n0 = cctx0.NCORE + cctx0.ncm
    h0 = Vector{Float64}(undef, n0 * (n0 + 1) ÷ 2)
    x0v = vcat(base0.ζstar, base0.λstar)
    _archC_prep_for_hessian!(ctx_cm0.obj, x0v)
    @time hessian_cm_structured_v2!(h0, ctx_cm0.obj, cctx0; threaded_bins = cctx0.use_threaded_bins, tls = cctx0.tls, use_syrk = true)
    check("K_pair=0 smoke: Hessian call completed with no NaN/Inf", all(isfinite, h0))
end

println()
println("Winner cross-Hessian counters (final): ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
