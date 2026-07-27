# Winner-aware H_ER phase (2026-07-27), Section 4: D=4 wiring-level gate for CM+ZC's H_EM
# (core x mean/pair cross block), :dense_reference vs :winner_bin, wired into
# _fill_cm_HEE!'s own ncore<NCORE branch (cm_hessian_architectures.jl). Exercises the WIRED
# backend end-to-end (complete packed Hessian, both serial hessian_cm_structured! and threaded
# hessian_cm_structured_v2!, plus a complete real KNITRO inner solve), mirroring
# test_flexible_cm_winner_bin_her_wiring_d4.jl's own methodology.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2026)

reset_no_dense_g_counters!()

# (K_mean, K_pair, label) -- K1 (regression-sized, matches test_cm_meanzc_d4_gates.jl's own
# original two named arms) and K2 (the larger generalized power-level config).
const CONFIGS = [(1, 1, "K1"), (2, 2, "K2")]

for (K_mean, K_pair, label) in CONFIGS, contrasts in (:anchored, :orthonormal)
    ν0 = nu0vec(K_mean)
    # Independent aug/obj_cm per backend (mirrors test_flexible_cm_winner_bin_her_wiring_d4.jl's
    # own pattern exactly) -- avoids any risk of cross-contaminating obj state between two
    # sequential complete KNITRO inner solves driven off the SAME mutable obj.
    aug_dense = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    aug_wbin = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    ctx_cm_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    ctx_cm_wbin = merge(ctx, (obj = aug_wbin.obj_cm,))
    cctx_dense = build_cm_meanzc_bin_ctx(ctx, aug_dense; zc_cross_hessian_backend = :dense_reference)
    cctx_wbin = build_cm_meanzc_bin_ctx(ctx, aug_wbin; zc_cross_hessian_backend = :winner_bin)
    check("$label contrasts=$contrasts: winner_bin cctx backend set", cctx_wbin.zc_cross_hessian_backend === :winner_bin)

    base_dense = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_dense, cctx_dense)
    base_wbin = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_wbin, cctx_wbin)
    check("$label contrasts=$contrasts: dense inner solve feasible", base_dense.inner_status in (0, -100, -101, -103))
    check("$label contrasts=$contrasts: winner_bin inner solve feasible", base_wbin.inner_status in (0, -100, -101, -103))
    check("$label contrasts=$contrasts: complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_wbin.inner_status))",
        base_dense.inner_status == base_wbin.inner_status)
    check("$label contrasts=$contrasts: complete inner solve dual point matches",
        maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_wbin.ζstar, base_wbin.λstar))) < 1e-8)

    NCORE = cctx_dense.NCORE; ncm = cctx_dense.ncm
    n = NCORE + ncm

    for (plabel, x) in (("calib", vcat(base_dense.ζstar, base_dense.λstar)),
                        ("perturbed_A", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.01, 0.02 .* randn(length(base_dense.λstar)))),
                        ("perturbed_B", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(-0.008, 0.015 .* randn(length(base_dense.λstar)))))
        for (arch_label, hess_fn) in (("serial", (h, obj, cctx) -> hessian_cm_structured!(h, obj, cctx)),
                                       ("threaded_v2", (h, obj, cctx) -> hessian_cm_structured_v2!(h, obj, cctx; threaded_bins = cctx.use_threaded_bins, tls = cctx.tls, use_syrk = true)))
            obj_d = ctx_cm_dense.obj; obj_w = ctx_cm_wbin.obj
            _archC_prep_for_hessian!(obj_d, x)
            hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hess_fn(hd, obj_d, cctx_dense)
            Hd = unpack_packed(hd, n)

            _archC_prep_for_hessian!(obj_w, x)
            hw = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hess_fn(hw, obj_w, cctx_wbin)
            Hw = unpack_packed(hw, n)

            maxdiff = maximum(abs.(Hd .- Hw))
            relscale = max(1.0, maximum(abs.(Hd)))
            ok = maxdiff < 1e-8 * relscale
            check("$label contrasts=$contrasts $plabel $arch_label: complete packed Hessian dense vs winner_bin matches (max|Δ|=$(maxdiff))", ok)
            @printf("  %s contrasts=%s %s %s: max|ΔH|=%.3e  scale=%.3e\n", label, contrasts, plabel, arch_label, maxdiff, relscale)
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
    check("$label contrasts=$contrasts: zc_cross_scratch object identity stable across calls (no resize)", scratch_id_before == scratch_id_after)
    @printf("  %s contrasts=%s: warm hessian_cm_structured_v2! (winner_bin) @allocated=%d bytes\n", label, contrasts, bytes)
end

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
