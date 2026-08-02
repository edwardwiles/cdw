# Winner-aware H_ER phase (2026-07-27), Section 3.2: common-Fréchet D=4 wiring-level gate.
#
# Modeled EXACTLY on test_flexible_cm_winner_bin_her_wiring_d4.jl (flexible-CM's own Section 2.2
# gate): exercises the WIRED backend end-to-end -- the complete packed Hessian from
# hessian_cm_frechet_structured! (serial) AND hessian_cm_frechet_structured_v2! (threaded), an
# ISOLATED H_E,level slice comparison (so a bug in the new level-anchor math localizes cleanly,
# separate from the CM-grid H_EC reuse), a complete real KNITRO inner solve through both backends,
# and persistent-workspace allocation stability.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "cm_checkpoint.jl"]
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
Random.seed!(2026)

reset_no_dense_g_counters!()

for contrasts in (:anchored, :orthonormal), L in (10, 20, 50)
    pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
        cm_hessian_backend = :structured, cm_cross_hessian_backend = :dense_reference)
    pcx_wbin = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
        cm_hessian_backend = :structured, cm_cross_hessian_backend = :winner_bin)
    check("contrasts=$contrasts L=$L: winner_bin cctx backend set", pcx_wbin.cctx.cm_cross_hessian_backend === :winner_bin)

    base_dense = archC_frechet_base_state(x_free_calib, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
    base_wbin = archC_frechet_base_state(x_free_calib, pcx_wbin.ctx_cm, pcx_wbin.cctx, pcx_wbin.aug.level_targets)
    check("contrasts=$contrasts L=$L: dense inner solve feasible", base_dense.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts L=$L: winner_bin inner solve feasible", base_wbin.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts L=$L: complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_wbin.inner_status))",
        base_dense.inner_status == base_wbin.inner_status)
    check("contrasts=$contrasts L=$L: complete inner solve dual point matches",
        maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_wbin.ζstar, base_wbin.λstar))) < 1e-8)

    NCORE = pcx_dense.cctx.NCORE; ncm = pcx_dense.cctx.ncm
    n = NCORE + ncm
    ncm_cm = pcx_dense.cctx.nO * L
    level_off = NCORE + ncm_cm

    level_targets_d = pcx_dense.aug.level_targets
    level_targets_w = pcx_wbin.aug.level_targets
    check("contrasts=$contrasts L=$L: level_targets agree dense vs winner_bin",
        maximum(abs.(level_targets_d .- level_targets_w)) < 1e-14)

    tls_dense = build_thread_local_scratch(pcx_dense.cctx)
    tls_wbin = build_thread_local_scratch(pcx_wbin.cctx)

    for (label, x) in (("calib", vcat(base_dense.ζstar, base_dense.λstar)),
                        ("perturbed_A", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.01, 0.02 .* randn(length(base_dense.λstar)))),
                        ("perturbed_B", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(-0.008, 0.015 .* randn(length(base_dense.λstar)))))
        for (arch_label, hess_fn) in (
                ("serial", (h, obj, cctx) -> hessian_cm_frechet_structured!(h, obj, cctx, cctx === pcx_dense.cctx ? level_targets_d : level_targets_w)),
                ("threaded_v2", (h, obj, cctx) -> hessian_cm_frechet_structured_v2!(h, obj, cctx, cctx === pcx_dense.cctx ? level_targets_d : level_targets_w; threaded_bins = true, tls = cctx === pcx_dense.cctx ? tls_dense : tls_wbin)))
            obj_d = pcx_dense.ctx_cm.obj; obj_w = pcx_wbin.ctx_cm.obj
            _archC_prep_for_hessian!(obj_d, x); _archC_prep_for_hessian!(obj_w, x)
            hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hw = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hess_fn(hd, obj_d, pcx_dense.cctx)
            # isolated H_E,level slice -- read straight off the persistent cctx.Hfull BEFORE the
            # next call overwrites it, so a level-block-only bug localizes cleanly against the
            # complete-Hessian comparison below.
            HElevel_dense = copy(pcx_dense.cctx.Hfull[1:NCORE, level_off+1:level_off+L])
            hess_fn(hw, obj_w, pcx_wbin.cctx)
            HElevel_wbin = copy(pcx_wbin.cctx.Hfull[1:NCORE, level_off+1:level_off+L])
            maxdiff_El = maximum(abs.(HElevel_dense .- HElevel_wbin))
            relscale_El = max(1.0, maximum(abs.(HElevel_dense)))
            check("contrasts=$contrasts L=$L $label $arch_label: ISOLATED H_E,level slice dense vs winner_bin matches (max|Δ|=$(maxdiff_El))",
                maxdiff_El < 1e-8 * relscale_El)

            Hd = unpack_packed(hd, n); Hw = unpack_packed(hw, n)
            maxdiff = maximum(abs.(Hd .- Hw))
            relscale = max(1.0, maximum(abs.(Hd)))
            ok = maxdiff < 1e-8 * relscale
            check("contrasts=$contrasts L=$L $label $arch_label: complete packed Hessian dense vs winner_bin matches (max|Δ|=$(maxdiff))", ok)
            @printf("  contrasts=%s L=%d %s %s: max|ΔH|=%.3e  max|ΔH_E,level|=%.3e  scale=%.3e\n", contrasts, L, label, arch_label, maxdiff, maxdiff_El, relscale)
        end
    end

    # persistent-workspace allocation check (threaded v2, winner_bin)
    obj_w = pcx_wbin.ctx_cm.obj; cctx_w = pcx_wbin.cctx
    x0 = vcat(base_wbin.ζstar, base_wbin.λstar)
    _archC_prep_for_hessian!(obj_w, x0)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_frechet_structured_v2!(h, obj_w, cctx_w, level_targets_w; threaded_bins = true, tls = tls_wbin)  # warmup
    scratch_id_before = objectid(cctx_w.cross_scratch)
    bytes = @allocated hessian_cm_frechet_structured_v2!(h, obj_w, cctx_w, level_targets_w; threaded_bins = true, tls = tls_wbin)
    scratch_id_after = objectid(cctx_w.cross_scratch)
    check("contrasts=$contrasts L=$L: cross_scratch object identity stable across calls (no resize)", scratch_id_before == scratch_id_after)
    @printf("  contrasts=%s L=%d: warm hessian_cm_frechet_structured_v2! (winner_bin) @allocated=%d bytes\n", contrasts, L, bytes)
end

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
