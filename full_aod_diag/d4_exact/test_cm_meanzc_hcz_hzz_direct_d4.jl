# CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): D=4 correctness gate for the NEW
# direct H_CZ (CM-grid x mean/pair-ZC-restriction cross) and shared direct H_ZZ (mean/pair-ZC
# self-Gram) Hessian primitives, exercised end-to-end via `cm_cross_hessian_backend=:winner_bin`
# TOGETHER with `zc_cross_hessian_backend=:winner_bin` (the combination that actually relaxes
# `_cm_cross_hessian_wants_winner_bin`'s `ncore_core==NCORE` guard and fills the widened rows via
# `bin_zc_cross_hessian_block!`/`zc_restriction_gram!` -- the pre-existing
# test_cm_meanzc_winner_bin_hez_wiring_d4.jl only ever sets `zc_cross_hessian_backend`, so it
# validates H_EM/H_ZZ but never exercises the NEW H_CZ path at all). Compares the COMPLETE packed
# Hessian against the untouched pre-refactor dense-reference construction (`obj.H` columns, both
# backends left at :dense_reference) at machine precision, for K_mean=1/K_pair=1 (production
# default width) and K_mean=2/K_pair=2 (a larger restriction width).
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
Random.seed!(2027)

reset_no_dense_g_counters!()

const CONFIGS = [(1, 1, "K1"), (2, 2, "K2")]

for (K_mean, K_pair, label) in CONFIGS, contrasts in (:anchored, :orthonormal)
    ν0 = nu0vec(K_mean)
    aug_dense = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    aug_direct = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    ctx_cm_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    ctx_cm_direct = merge(ctx, (obj = aug_direct.obj_cm,))
    cctx_dense = build_cm_meanzc_bin_ctx(ctx, aug_dense; cm_cross_hessian_backend = :dense_reference, zc_cross_hessian_backend = :dense_reference)
    cctx_direct = build_cm_meanzc_bin_ctx(ctx, aug_direct; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    check("$label contrasts=$contrasts: direct cctx backends set", cctx_direct.cm_cross_hessian_backend === :winner_bin && cctx_direct.zc_cross_hessian_backend === :winner_bin)
    check("$label contrasts=$contrasts: hzz_zc_op built", cctx_direct.hzz_zc_op !== nothing)

    base_dense = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_dense, cctx_dense)
    base_direct = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_direct, cctx_direct)
    check("$label contrasts=$contrasts: dense inner solve feasible", base_dense.inner_status in (0, -100, -101, -103))
    check("$label contrasts=$contrasts: direct inner solve feasible", base_direct.inner_status in (0, -100, -101, -103))
    check("$label contrasts=$contrasts: complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_direct.inner_status))",
        base_dense.inner_status == base_direct.inner_status)
    check("$label contrasts=$contrasts: complete inner solve dual point matches",
        maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_direct.ζstar, base_direct.λstar))) < 1e-8)

    NCORE = cctx_dense.NCORE; ncm = cctx_dense.ncm
    n = NCORE + ncm

    for (plabel, x) in (("calib", vcat(base_dense.ζstar, base_dense.λstar)),
                        ("perturbed_A", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.01, 0.02 .* randn(length(base_dense.λstar)))),
                        ("perturbed_B", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(-0.008, 0.015 .* randn(length(base_dense.λstar)))))
        for (arch_label, hess_fn) in (("serial", (h, obj, cctx) -> hessian_cm_structured!(h, obj, cctx)),
                                       ("threaded_v2", (h, obj, cctx) -> hessian_cm_structured_v2!(h, obj, cctx; threaded_bins = cctx.use_threaded_bins, tls = cctx.tls, use_syrk = true)))
            obj_d = ctx_cm_dense.obj; obj_r = ctx_cm_direct.obj
            _archC_prep_for_hessian!(obj_d, x)
            hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hess_fn(hd, obj_d, cctx_dense)
            Hd = unpack_packed(hd, n)

            _archC_prep_for_hessian!(obj_r, x)
            hr = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hess_fn(hr, obj_r, cctx_direct)
            Hr = unpack_packed(hr, n)

            maxdiff = maximum(abs.(Hd .- Hr))
            relscale = max(1.0, maximum(abs.(Hd)))
            ok = maxdiff < 1e-8 * relscale
            check("$label contrasts=$contrasts $plabel $arch_label: complete packed Hessian dense vs direct(H_CZ+H_ZZ) matches (max|Δ|=$(maxdiff))", ok)
            @printf("  %s contrasts=%s %s %s: max|ΔH|=%.3e  scale=%.3e\n", label, contrasts, plabel, arch_label, maxdiff, relscale)

            # Isolate the Z-rows sub-block (H_EZ/H_CZ/H_ZZ) specifically -- the part this session's
            # NEW code touches -- so a pass here can't be masked by the (already-validated) E/C
            # blocks dominating the max-abs-diff scale.
            ncore_core = cctx_dense.ncore_core
            Zrows = ncore_core+1:NCORE
            zdiff = maximum(abs.(Hd[Zrows, :] .- Hr[Zrows, :]))
            zscale = max(1.0, maximum(abs.(Hd[Zrows, :])))
            zok = zdiff < 1e-8 * zscale
            check("$label contrasts=$contrasts $plabel $arch_label: Z-rows (H_EZ|H_CZ|H_ZZ) sub-block matches (max|Δ|=$(zdiff))", zok)
        end
    end

    # persistent-workspace allocation / object-identity check for the NEW scratch structs
    x0 = vcat(base_direct.ζstar, base_direct.λstar)
    _archC_prep_for_hessian!(ctx_cm_direct.obj, x0)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured_v2!(h, ctx_cm_direct.obj, cctx_direct; threaded_bins = cctx_direct.use_threaded_bins, tls = cctx_direct.tls, use_syrk = true)  # warmup
    id_hzz_before = objectid(cctx_direct.hzz_centered)
    id_binzc_before = objectid(cctx_direct.bin_zc_cross)
    bytes = @allocated hessian_cm_structured_v2!(h, ctx_cm_direct.obj, cctx_direct; threaded_bins = cctx_direct.use_threaded_bins, tls = cctx_direct.tls, use_syrk = true)
    id_hzz_after = objectid(cctx_direct.hzz_centered)
    id_binzc_after = objectid(cctx_direct.bin_zc_cross)
    check("$label contrasts=$contrasts: hzz_centered object identity stable across calls (no resize)", id_hzz_before == id_hzz_after)
    check("$label contrasts=$contrasts: bin_zc_cross object identity stable across calls (no resize)", id_binzc_before == id_binzc_after)
    @printf("  %s contrasts=%s: warm hessian_cm_structured_v2! (direct H_CZ/H_ZZ) @allocated=%d bytes\n", label, contrasts, bytes)
end

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
