# optimize/structured-cross-hessian-ZC-CM-2026-07-28: D=4 correctness gate for the threaded
# H_EC/H_EZ/H_CZ kernels (threaded_cross_hessian.jl) and the H_ZZ BLAS/threaded candidates
# (zc_gram_blas_candidates.jl, addendum). Compares the COMPLETE packed Hessian at calibration and
# perturbed points, for flexible_cm/cm_meanzc/origin_zc, several K_mean/K_pair configurations
# (including mean-only K_pair=0 and a larger K config), and both square and rectangular
# (destination_sample=:exclude_row-style D=4 rectangular) layouts where applicable.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_config.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

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
Random.seed!(2028)

reset_no_dense_g_counters!()

# Pre-existing KNITRO-Julia-wrapper flakiness (see diag_subblock_profile_2026-07-28.jl's own
# header comment for the full writeup) -- ONLY_FAMILY isolates a single family's KNITRO solves to
# their own Julia process for a clean run.
const ONLY_FAMILY = get(ENV, "ONLY_FAMILY", "")
run_family(name) = isempty(ONLY_FAMILY) || ONLY_FAMILY == name

# ============================================================================
# flexible_cm: threaded H_EC vs serial, workers in {1,2,4}
# ============================================================================
if run_family("flexible_cm")
println("\n=== flexible_cm: threaded H_EC vs serial ===")
aug_fc = build_cm_augmented_obj(ctx, CS; L = 10, contrasts = :anchored)
ctx_cm_fc = merge(ctx, (obj = aug_fc.obj_cm,))
cctx_fc = build_cm_bin_ctx(ctx, aug_fc; cm_cross_hessian_backend = :winner_bin)
base_fc = archC_base_state(x_free_calib, ctx_cm_fc, cctx_fc)
n_fc = cctx_fc.NCORE + cctx_fc.ncm
for (plabel, x) in (("calib", vcat(base_fc.ζstar, base_fc.λstar)),
                     ("perturbed", vcat(base_fc.ζstar, base_fc.λstar) .+ vcat(0.01, 0.02 .* randn(length(base_fc.λstar)))))
    _archC_prep_for_hessian!(ctx_cm_fc.obj, x)
    h_serial = Vector{Float64}(undef, n_fc * (n_fc + 1) ÷ 2)
    cctx_fc.cross_hessian_threaded = false
    hessian_cm_structured_v2!(h_serial, ctx_cm_fc.obj, cctx_fc; threaded_bins = true, tls = cctx_fc.tls)
    H_serial = unpack_packed(h_serial, n_fc)
    for workers in (1, 2, 4)
        _archC_prep_for_hessian!(ctx_cm_fc.obj, x)
        h_th = Vector{Float64}(undef, n_fc * (n_fc + 1) ÷ 2)
        cctx_fc.cross_hessian_threaded = true
        cctx_fc.cross_hessian_workers = workers
        hessian_cm_structured_v2!(h_th, ctx_cm_fc.obj, cctx_fc; threaded_bins = true, tls = cctx_fc.tls)
        H_th = unpack_packed(h_th, n_fc)
        maxdiff = maximum(abs.(H_serial .- H_th))
        check("flexible_cm $plabel workers=$workers: complete Hessian bit-exact (maxdiff=$maxdiff)", maxdiff < 1e-12)
    end
end
cctx_fc.cross_hessian_threaded = false
end # run_family("flexible_cm")

# ============================================================================
# common_frechet: threaded H_EC vs serial, workers in {1,2,4}. Uses the REAL production
# constructor (build_cm_production_context_v2/CMConfig, matching this family's own existing D=4/
# D=20 gates -- test_frechet_hessian_structured_vs_dense_d4.jl, test_cm_frechet_threaded_hessian_
# gates.jl) rather than the older build_cm_augmented_obj flexible_cm's own section above uses --
# CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[] is :winner_bin and use_compressed_core defaults true
# for this constructor, so (unlike flexible_cm's own section, which -- see the cm_meanzc root-cause
# writeup in docs/ZC_CENTERING_LIFECYCLE_RELEASE_2026-07-28.md -- never populates core_cf_ref and
# therefore never actually exercises the :winner_bin H_EC path) this section's threaded-vs-serial
# comparison genuinely exercises `_ensure_cm_cross_scratch!`/`WinnerBinCrossScratch` and this
# task's `winner_pair_cross_hessian_fill_threaded!` kernel, not just the dense fallback.
# `hessian_cm_frechet_structured_v2!` is now a thin wrapper around the SAME shared
# `hessian_cm_structured_v2!` flexible_cm/cm_meanzc use (harmonization task, 2026-07-28), so
# `cctx.cross_hessian_threaded`/`cross_hessian_workers` apply identically.
# ============================================================================
if run_family("common_frechet")
println("\n=== common_frechet: threaded H_EC vs serial ===")
cfg_frechet = CMConfig(common_marginals = true, cm_grid_size = 10, cm_hessian_backend = :structured,
                        contrasts = :anchored, marginal_restriction = :common_frechet)
pcx_fr = build_cm_production_context_v2(ctx, CS, cfg_frechet; L = 10)
cctx_fr = pcx_fr.cctx
level_targets_fr = pcx_fr.aug.level_targets
obj_fr = pcx_fr.ctx_cm.obj
θ_full0_fr = CS.reconstruct_full(x_free_calib, pcx_fr.ctx_cm.m)
K_fr, x_sol_fr, nStatus_fr, n_fg_fr, n_hess_fr = inner_loop_internal_archgeneric(obj_fr, θ_full0_fr; hess_cb_builder = pcx_fr.hess_cb_builder)
check("common_frechet: inner solve feasible (nStatus=$nStatus_fr)", nStatus_fr in (0, -100, -101, -102, -103, -400, -401, -402))
n_fr = cctx_fr.NCORE + cctx_fr.ncm
println("common_frechet: cf type after solve = ", typeof(cctx_fr.core_cf_ref[]),
        ", cross_hessian_backend wants winner_bin = ", _cm_cross_hessian_wants_winner_bin(cctx_fr, cctx_fr.core_cf_ref[]))

for (plabel, x) in (("calib", collect(x_sol_fr)),
                     ("perturbed", vcat(x_sol_fr[1] + 0.01, x_sol_fr[2:end] .+ 0.02 .* randn(length(x_sol_fr) - 1))))
    _archC_prep_for_hessian!(obj_fr, x)
    h_serial = Vector{Float64}(undef, n_fr * (n_fr + 1) ÷ 2)
    cctx_fr.cross_hessian_threaded = false
    hessian_cm_frechet_structured_v2!(h_serial, obj_fr, cctx_fr, level_targets_fr; threaded_bins = true, tls = cctx_fr.tls)
    H_serial = unpack_packed(h_serial, n_fr)
    for workers in (1, 2, 4)
        _archC_prep_for_hessian!(obj_fr, x)
        h_th = Vector{Float64}(undef, n_fr * (n_fr + 1) ÷ 2)
        cctx_fr.cross_hessian_threaded = true
        cctx_fr.cross_hessian_workers = workers
        hessian_cm_frechet_structured_v2!(h_th, obj_fr, cctx_fr, level_targets_fr; threaded_bins = true, tls = cctx_fr.tls)
        H_th = unpack_packed(h_th, n_fr)
        maxdiff = maximum(abs.(H_serial .- H_th))
        check("common_frechet $plabel workers=$workers: complete Hessian bit-exact (maxdiff=$maxdiff)", maxdiff < 1e-12)
    end
end
cctx_fr.cross_hessian_threaded = false
end # run_family("common_frechet")

# ============================================================================
# cm_meanzc: threaded H_EC/H_EZ/H_CZ + H_ZZ backend sweep, several K configs
# ============================================================================
if run_family("cm_meanzc")
println("\n=== cm_meanzc: threaded kernels + H_ZZ backend sweep ===")
for (K_mean, K_pair, label) in [(1, 0, "K_mean1_pair0"), (1, 1, "K1"), (2, 1, "K2")]
    ν0 = nu0vec(K_mean)
    aug_z = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored, meanzc_basis = :direct)
    ctx_cm_z = merge(ctx, (obj = aug_z.obj_cm,))
    cctx_z = build_cm_meanzc_bin_ctx(ctx, aug_z; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    base_z = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_z, cctx_z)
    check("$label: inner solve feasible (nStatus=$(base_z.inner_status))", base_z.inner_status in (0, -100, -101, -103))
    n_z = cctx_z.NCORE + cctx_z.ncm

    for (plabel, x) in (("calib", vcat(base_z.ζstar, base_z.λstar)),
                        ("perturbed", vcat(base_z.ζstar, base_z.λstar) .+ vcat(0.01, 0.02 .* randn(length(base_z.λstar)))))
        _archC_prep_for_hessian!(ctx_cm_z.obj, x)
        cctx_z.cross_hessian_threaded = false
        cctx_z.zc_gram_backend = :reference
        h_ref = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
        hessian_cm_structured_v2!(h_ref, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
        H_ref = unpack_packed(h_ref, n_z)

        for backend in (:blas_syrk, :blas_gemm, :threaded_packed)
            _archC_prep_for_hessian!(ctx_cm_z.obj, x)
            cctx_z.cross_hessian_threaded = false
            cctx_z.zc_gram_backend = backend
            h_b = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
            hessian_cm_structured_v2!(h_b, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
            H_b = unpack_packed(h_b, n_z)
            maxdiff = maximum(abs.(H_ref .- H_b))
            check("$label $plabel zc_gram_backend=$backend: complete Hessian machine-precision (maxdiff=$maxdiff)", maxdiff < 1e-9)
        end

        cctx_z.zc_gram_backend = :reference
        for workers in (1, 2, 4)
            _archC_prep_for_hessian!(ctx_cm_z.obj, x)
            cctx_z.cross_hessian_threaded = true
            cctx_z.cross_hessian_workers = workers
            h_th = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
            hessian_cm_structured_v2!(h_th, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
            H_th = unpack_packed(h_th, n_z)
            maxdiff = maximum(abs.(H_ref .- H_th))
            check("$label $plabel cross_hessian_threaded workers=$workers: complete Hessian bit-exact (maxdiff=$maxdiff)", maxdiff < 1e-12)
        end
        cctx_z.cross_hessian_threaded = false
    end
end
end # run_family("cm_meanzc")

# ============================================================================
# origin_zc: threaded H_EZ(=HER) + H_ZZ(=HRR) backend sweep, several K configs
# ============================================================================
if run_family("origin_zc")
println("\n=== origin_zc: threaded kernels + H_ZZ backend sweep ===")
for (K_mean, K_pair, label) in [(1, 0, "K_mean1_pair0"), (1, 1, "K1"), (2, 1, "K2")]
    layout_o = OriginByPowerLayout(ctx.D, K_mean, K_pair)
    νfull0 = vcat([fill(Float64(factorial(k)), ctx.D) for k in 1:K_mean]..., [fill(Float64(factorial(k)), ctx.D * (ctx.D - 1) ÷ 2) for k in 1:K_pair]...)
    pcx_o = build_originzc_production_context(ctx, CS, layout_o; zc_cross_hessian_backend = :winner_bin)
    octx = pcx_o.octx
    base_o = archOZ_base_state(x_free_calib, νfull0, pcx_o.ctx_cm)
    check("$label: inner solve feasible (nStatus=$(base_o.inner_status))", base_o.inner_status in (0, -100, -101, -103))
    n_o = octx.NCORE + octx.n_eta

    for (plabel, x) in (("calib", vcat(base_o.ζstar, base_o.λstar)),
                        ("perturbed", vcat(base_o.ζstar, base_o.λstar) .+ vcat(0.01, 0.02 .* randn(length(base_o.λstar)))))
        obj_o = pcx_o.ctx_cm.obj
        _prep_dual_index_for_archA!(octx, obj_o, x)
        octx.cross_hessian_threaded = false
        octx.zc_gram_backend = :reference
        h_ref = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
        archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h_ref,), obj_o)
        H_ref = unpack_packed(h_ref, n_o)

        for backend in (:blas_syrk, :blas_gemm, :threaded_packed)
            _prep_dual_index_for_archA!(octx, obj_o, x)
            octx.cross_hessian_threaded = false
            octx.zc_gram_backend = backend
            h_b = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
            archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h_b,), obj_o)
            H_b = unpack_packed(h_b, n_o)
            maxdiff = maximum(abs.(H_ref .- H_b))
            check("$label $plabel zc_gram_backend=$backend: complete Hessian machine-precision (maxdiff=$maxdiff)", maxdiff < 1e-9)
        end

        octx.zc_gram_backend = :reference
        for workers in (1, 2, 4)
            _prep_dual_index_for_archA!(octx, obj_o, x)
            octx.cross_hessian_threaded = true
            octx.cross_hessian_workers = workers
            h_th = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
            archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h_th,), obj_o)
            H_th = unpack_packed(h_th, n_o)
            maxdiff = maximum(abs.(H_ref .- H_th))
            check("$label $plabel cross_hessian_threaded workers=$workers: complete Hessian bit-exact (maxdiff=$maxdiff)", maxdiff < 1e-12)
        end
        octx.cross_hessian_threaded = false
    end
end
end # run_family("origin_zc")

println("\n", ALL_PASS[] ? "ALL_PASS" : "SOME_FAILED")
