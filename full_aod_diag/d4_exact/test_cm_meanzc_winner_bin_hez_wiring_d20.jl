# Winner-aware H_ER phase (2026-07-27), Section 4: real D=20/W=80,000/L=50 wiring-level gate for
# CM+ZC's H_EM (core x mean/pair cross block). Companion to
# test_cm_meanzc_winner_bin_hez_wiring_d4.jl -- exercises the WIRED backend end-to-end at real
# production scale: complete packed Hessian (serial + threaded_v2), complete real KNITRO inner
# solve (status + dual point), and persistent-workspace allocation, at calibration and a
# near-delta=1 perturbed point, both contrasts, K_mean=1/K_pair=1 (Point-A-sized, matching
# d20_meanzc_release_gates.jl's own Point A config).
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
Random.seed!(2026)

reset_no_dense_g_counters!()

const K_mean, K_pair = 1, 1
ν0 = nu0vec(K_mean)

for contrasts in (:anchored, :orthonormal)
    println("=== contrasts=$contrasts L=50 K_mean=$K_mean K_pair=$K_pair ==="); flush(stdout)
    aug_dense = build_cm_meanzc_augmented_obj(ctx, CS; L = 50, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    aug_wbin = build_cm_meanzc_augmented_obj(ctx, CS; L = 50, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    ctx_cm_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    ctx_cm_wbin = merge(ctx, (obj = aug_wbin.obj_cm,))
    cctx_dense = build_cm_meanzc_bin_ctx(ctx, aug_dense; zc_cross_hessian_backend = :dense_reference)
    cctx_wbin = build_cm_meanzc_bin_ctx(ctx, aug_wbin; zc_cross_hessian_backend = :winner_bin)
    check("contrasts=$contrasts: winner_bin cctx backend set", cctx_wbin.zc_cross_hessian_backend === :winner_bin)

    println("  solving dense-reference base state..."); flush(stdout)
    @time base_dense = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_dense, cctx_dense)
    println("  solving winner_bin base state..."); flush(stdout)
    @time base_wbin = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_wbin, cctx_wbin)
    check("contrasts=$contrasts: dense inner solve feasible (status=$(base_dense.inner_status))", base_dense.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts: winner_bin inner solve feasible (status=$(base_wbin.inner_status))", base_wbin.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts: complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_wbin.inner_status))",
        base_dense.inner_status == base_wbin.inner_status)
    dual_diff = maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_wbin.ζstar, base_wbin.λstar)))
    check("contrasts=$contrasts: complete inner solve dual point matches (max|Δ|=$dual_diff)", dual_diff < 1e-6)

    NCORE = cctx_dense.NCORE; ncm = cctx_dense.ncm
    n = NCORE + ncm

    for (label, x) in (("calib", vcat(base_dense.ζstar, base_dense.λstar)),
                        ("near_delta1_perturbed", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.005, 0.01 .* randn(length(base_dense.λstar)))))
        for (arch_label, hess_fn) in (("serial", (h, obj, cctx) -> hessian_cm_structured!(h, obj, cctx)),
                                       ("threaded_v2", (h, obj, cctx) -> hessian_cm_structured_v2!(h, obj, cctx; threaded_bins = cctx.use_threaded_bins, tls = cctx.tls, use_syrk = true)))
            obj_d = ctx_cm_dense.obj; obj_w = ctx_cm_wbin.obj
            _archC_prep_for_hessian!(obj_d, x); _archC_prep_for_hessian!(obj_w, x)
            hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hw = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            print("    $label $arch_label dense:      "); @time hess_fn(hd, obj_d, cctx_dense)
            print("    $label $arch_label winner_bin: "); @time hess_fn(hw, obj_w, cctx_wbin)
            Hd = unpack_packed(hd, n); Hw = unpack_packed(hw, n)
            maxdiff = maximum(abs.(Hd .- Hw))
            relscale = max(1.0, maximum(abs.(Hd)))
            ok = maxdiff < 1e-6 * relscale
            check("contrasts=$contrasts $label $arch_label: complete packed Hessian dense vs winner_bin matches (max|Δ|=$(maxdiff))", ok)
            @printf("      max|ΔH|=%.3e  scale=%.3e\n", maxdiff, relscale)
        end
    end

    # persistent-workspace allocation check (threaded v2, the production default path)
    x0 = vcat(base_wbin.ζstar, base_wbin.λstar)
    _archC_prep_for_hessian!(ctx_cm_wbin.obj, x0)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured_v2!(h, ctx_cm_wbin.obj, cctx_wbin; threaded_bins = cctx_wbin.use_threaded_bins, tls = cctx_wbin.tls, use_syrk = true)  # warmup
    scratch_id_before = objectid(cctx_wbin.zc_cross_scratch)
    bytes = @allocated hessian_cm_structured_v2!(h, ctx_cm_wbin.obj, cctx_wbin; threaded_bins = cctx_wbin.use_threaded_bins, tls = cctx_wbin.tls, use_syrk = true)
    scratch_id_after = objectid(cctx_wbin.zc_cross_scratch)
    check("contrasts=$contrasts: zc_cross_scratch object identity stable across calls (no resize)", scratch_id_before == scratch_id_after)
    @printf("  contrasts=%s: warm hessian_cm_structured_v2! (winner_bin) @allocated=%.3f MB\n", contrasts, bytes / 1e6)
    flush(stdout)
end

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
