# Winner-aware H_ER phase (2026-07-27), Section 3.2: common-Fréchet real D=20/W=80,000/L=50
# wiring-level gate. Companion to test_frechet_winner_bin_her_wiring_d4.jl -- exercises the WIRED
# backend end-to-end at real production scale: isolated H_E,level slice, complete packed Hessian
# (serial + threaded_v2), complete inner solve (KNITRO status + dual point), and
# persistent-workspace allocation, at calibration, a near-delta=1 perturbed point, and a genuinely
# hard point (both contrasts).
#
# "Hard point": this family's own D20 gates (test_frechet_d20_gates_L50.jl,
# docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md) establish `x_free0 .* 1.01` (a uniform 1%
# perturbation of every free coordinate away from calibration -- NOT just a "near-delta=1" single
# small nudge) as the point that actually stresses this family's own Hessian/FG machinery hardest
# in this repo's own prior sessions (it is the exact point that exposed the skip_cm_fill_ref bug
# documented in that file). Reused here rather than inventing a new one.
const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_cplus.jl", "cm_checkpoint.jl"]
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
Random.seed!(2026)

reset_no_dense_g_counters!()
L = 50

for contrasts in (:anchored, :orthonormal)
    println("=== contrasts=$contrasts L=$L ==="); flush(stdout)
    pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
        cm_hessian_backend = :structured, cm_cross_hessian_backend = :dense_reference)
    pcx_wbin = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
        cm_hessian_backend = :structured, cm_cross_hessian_backend = :winner_bin)
    check("contrasts=$contrasts: winner_bin cctx backend set", pcx_wbin.cctx.cm_cross_hessian_backend === :winner_bin)

    println("  solving dense-reference base state..."); flush(stdout)
    @time base_dense = archC_frechet_base_state(x_free_calib, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
    println("  solving winner_bin base state..."); flush(stdout)
    @time base_wbin = archC_frechet_base_state(x_free_calib, pcx_wbin.ctx_cm, pcx_wbin.cctx, pcx_wbin.aug.level_targets)
    check("contrasts=$contrasts: dense inner solve feasible (status=$(base_dense.inner_status))", base_dense.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts: winner_bin inner solve feasible (status=$(base_wbin.inner_status))", base_wbin.inner_status in (0, -100, -101, -103))
    check("contrasts=$contrasts: complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_wbin.inner_status))",
        base_dense.inner_status == base_wbin.inner_status)
    dual_diff = maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_wbin.ζstar, base_wbin.λstar)))
    check("contrasts=$contrasts: complete inner solve dual point matches (max|Δ|=$dual_diff)", dual_diff < 1e-6)

    NCORE = pcx_dense.cctx.NCORE; ncm = pcx_dense.cctx.ncm
    n = NCORE + ncm
    ncm_cm = pcx_dense.cctx.nO * L
    level_off = NCORE + ncm_cm

    level_targets_d = pcx_dense.aug.level_targets
    level_targets_w = pcx_wbin.aug.level_targets

    tls_dense = build_thread_local_scratch(pcx_dense.cctx)
    tls_wbin = build_thread_local_scratch(pcx_wbin.cctx)

    # "hard point": x_free0 .* 1.01, the exact perturbation that exposed the skip_cm_fill_ref bug
    # in this family's own history (docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md). We solve
    # it at the DENSE backend first (known-good reference), then feed the SAME converged dual point
    # into both backends' Hessian callbacks for the actual comparison (matching how the calib/
    # near-delta1 points below are handled too -- solve once with dense, compare Hessians at that
    # shared point).
    x_free_hard = x_free_calib .* 1.01

    for (label, x_free_for_solve, x) in (
            ("calib", x_free_calib, vcat(base_dense.ζstar, base_dense.λstar)),
            ("near_delta1_perturbed", x_free_calib, vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.005, 0.01 .* randn(length(base_dense.λstar)))),
            ("hard_point_x1.01", x_free_hard, nothing))
        if label == "hard_point_x1.01"
            println("  solving hard point (x_free0 .* 1.01) at dense backend..."); flush(stdout)
            base_hard = archC_frechet_base_state(x_free_hard, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
            check("contrasts=$contrasts hard_point: dense inner solve feasible (status=$(base_hard.inner_status))",
                base_hard.inner_status in (0, -100, -101, -103))
            x = vcat(base_hard.ζstar, base_hard.λstar)
        end
        for (arch_label, hess_fn) in (
                ("serial", (h, obj, cctx) -> hessian_cm_frechet_structured!(h, obj, cctx, cctx === pcx_dense.cctx ? level_targets_d : level_targets_w)),
                ("threaded_v2", (h, obj, cctx) -> hessian_cm_frechet_structured_v2!(h, obj, cctx, cctx === pcx_dense.cctx ? level_targets_d : level_targets_w; threaded_bins = true, tls = cctx === pcx_dense.cctx ? tls_dense : tls_wbin)))
            obj_d = pcx_dense.ctx_cm.obj; obj_w = pcx_wbin.ctx_cm.obj
            _archC_prep_for_hessian!(obj_d, x); _archC_prep_for_hessian!(obj_w, x)
            hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            hw = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            print("    $label $arch_label dense:      "); @time hess_fn(hd, obj_d, pcx_dense.cctx)
            HElevel_dense = copy(pcx_dense.cctx.Hfull[1:NCORE, level_off+1:level_off+L])
            print("    $label $arch_label winner_bin: "); @time hess_fn(hw, obj_w, pcx_wbin.cctx)
            HElevel_wbin = copy(pcx_wbin.cctx.Hfull[1:NCORE, level_off+1:level_off+L])

            maxdiff_El = maximum(abs.(HElevel_dense .- HElevel_wbin))
            relscale_El = max(1.0, maximum(abs.(HElevel_dense)))
            check("contrasts=$contrasts $label $arch_label: ISOLATED H_E,level slice dense vs winner_bin matches (max|Δ|=$(maxdiff_El))",
                maxdiff_El < 1e-6 * relscale_El)

            Hd = unpack_packed(hd, n); Hw = unpack_packed(hw, n)
            maxdiff = maximum(abs.(Hd .- Hw))
            relscale = max(1.0, maximum(abs.(Hd)))
            ok = maxdiff < 1e-6 * relscale
            check("contrasts=$contrasts $label $arch_label: complete packed Hessian dense vs winner_bin matches (max|Δ|=$(maxdiff))", ok)
            @printf("      max|ΔH|=%.3e  max|ΔH_E,level|=%.3e  scale=%.3e\n", maxdiff, maxdiff_El, relscale)
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
    check("contrasts=$contrasts: cross_scratch object identity stable across calls (no resize)", scratch_id_before == scratch_id_after)
    @printf("  contrasts=%s: warm hessian_cm_frechet_structured_v2! (winner_bin) @allocated=%.3f MB\n", contrasts, bytes / 1e6)
    flush(stdout)
end

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
