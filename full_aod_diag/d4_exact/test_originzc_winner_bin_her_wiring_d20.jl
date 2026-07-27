# Winner-aware H_ER phase (2026-07-27), Section 5: real D=20/W=80,000 wiring-level gate for
# origin-ZC's H_ER (core x mean/pair cross block, this family's ONLY restriction block). Companion
# to test_originzc_winner_bin_her_wiring_d4.jl -- exercises the WIRED backend end-to-end at real
# production scale via the SAME KNITRO Hessian-callback closure production uses, plus a complete
# real KNITRO inner solve, at calibration and a near-delta=1 perturbed point, K_mean=1/K_pair=1
# (matching d20_meanzc_release_gates.jl's own Point A config).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_originzc_moments.jl",
          "cm_originzc_lookup_production.jl", "cm_originzc_production.jl"]
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

"Invoke archA_partitioned_hess_cb_builder(octx)'s closure directly with mock KNITRO
evalRequest/evalResult NamedTuples (only .x/.hess are ever read/written by that closure)."
function packed_hess_via_octx(octx, obj, x::AbstractVector{Float64}, n::Int)
    cb = archA_partitioned_hess_cb_builder(octx)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    evalRequest = (x = x,)
    evalResult = (hess = h,)
    cb(nothing, nothing, evalRequest, evalResult, obj)
    return h
end

println("Building real D=20 context (W=80000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2026)

reset_no_dense_g_counters!()

const K_mean, K_pair = 1, 1
ν0 = nu0vec(K_mean)
layout = SharedByPowerLayout(K_mean, K_pair)

println("Building augmented objects (K_mean=$K_mean, K_pair=$K_pair)..."); flush(stdout)
aug_dense = build_originzc_augmented_obj(ctx, CS, layout)
aug_wbin = build_originzc_augmented_obj(ctx, CS, layout)
ctx_cm_dense = merge(ctx, (obj = aug_dense.obj_cm,))
ctx_cm_wbin = merge(ctx, (obj = aug_wbin.obj_cm,))
octx_dense = build_originzc_core_hess_ctx(aug_dense; zc_cross_hessian_backend = :dense_reference)
octx_wbin = build_originzc_core_hess_ctx(aug_wbin; zc_cross_hessian_backend = :winner_bin)
ctx_cm_dense = merge(ctx_cm_dense, (octx = octx_dense,))
ctx_cm_wbin = merge(ctx_cm_wbin, (octx = octx_wbin,))
check("winner_bin octx backend set", octx_wbin.zc_cross_hessian_backend === :winner_bin)

println("  solving dense-reference base state..."); flush(stdout)
@time base_dense = archOZ_base_state(x_free_calib, ν0, ctx_cm_dense)
println("  solving winner_bin base state..."); flush(stdout)
@time base_wbin = archOZ_base_state(x_free_calib, ν0, ctx_cm_wbin)
check("dense inner solve feasible (status=$(base_dense.inner_status))", base_dense.inner_status in (0, -100, -101, -103))
check("winner_bin inner solve feasible (status=$(base_wbin.inner_status))", base_wbin.inner_status in (0, -100, -101, -103))
check("complete inner solve status matches (nStatus $(base_dense.inner_status) vs $(base_wbin.inner_status))",
    base_dense.inner_status == base_wbin.inner_status)
dual_diff = maximum(abs.(vcat(base_dense.ζstar, base_dense.λstar) .- vcat(base_wbin.ζstar, base_wbin.λstar)))
check("complete inner solve dual point matches (max|Δ|=$dual_diff)", dual_diff < 1e-6)

NCORE = octx_dense.NCORE; n_eta_total = octx_dense.n_eta
n = NCORE + n_eta_total

for (label, x) in (("calib", vcat(base_dense.ζstar, base_dense.λstar)),
                    ("near_delta1_perturbed", vcat(base_dense.ζstar, base_dense.λstar) .+ vcat(0.005, 0.01 .* randn(length(base_dense.λstar)))))
    print("  $label dense:      "); @time hd = packed_hess_via_octx(octx_dense, ctx_cm_dense.obj, x, n)
    print("  $label winner_bin: "); @time hw = packed_hess_via_octx(octx_wbin, ctx_cm_wbin.obj, x, n)
    Hd = unpack_packed(hd, n); Hw = unpack_packed(hw, n)
    maxdiff = maximum(abs.(Hd .- Hw))
    relscale = max(1.0, maximum(abs.(Hd)))
    ok = maxdiff < 1e-6 * relscale
    check("$label: complete packed Hessian dense vs winner_bin matches (max|Δ|=$(maxdiff))", ok)
    @printf("    max|ΔH|=%.3e  scale=%.3e\n", maxdiff, relscale)
end

# persistent-workspace allocation check
x0 = vcat(base_wbin.ζstar, base_wbin.λstar)
packed_hess_via_octx(octx_wbin, ctx_cm_wbin.obj, x0, n)  # warmup
scratch_id_before = objectid(octx_wbin.zc_cross_scratch)
bytes = @allocated packed_hess_via_octx(octx_wbin, ctx_cm_wbin.obj, x0, n)
scratch_id_after = objectid(octx_wbin.zc_cross_scratch)
check("zc_cross_scratch object identity stable across calls (no resize)", scratch_id_before == scratch_id_after)
@printf("  warm archA_partitioned_hess_cb_builder (winner_bin) @allocated=%.3f MB\n", bytes / 1e6)

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
