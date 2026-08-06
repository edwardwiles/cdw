# 2026-08-05 root-cause fix #3 verification: CM+ZC's widened-row H_CZ block under the THREADED
# Architecture C dispatch (hessian_cm_structured_v2!, threaded_bins=true) with a two-family
# (n_families==2) context. The pre-existing test_cm_meanzc_archc_twofamily_2026-08-05.jl only ever
# calls the SERIAL hessian_cm_structured! directly (bypassing cctx.use_threaded_bins entirely), so
# it never actually exercised the threaded/v2 dispatch this fix touches -- this script reuses that
# test's exact context/point construction (K_mean=1, K_pair=0, genuinely ZC-widened per its own
# `@assert cctx.ncore_core < cctx.NCORE`) and adds a direct hessian_cm_structured_v2! comparison.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "oracle_fast.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

function unpack_packed(h::AbstractVector, n::Int)
    Mx = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mx[i, j] = h[k]; Mx[j, i] = h[k]
        k += 1
    end
    return Mx
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 3
K_mean = 1

aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = 0, include_truncated_moment = true)
@assert aug.n_families == 2
objA = aug.obj_cm
n = objA.outer_constr_index

nu1_guess = sum(aug.Zraw_all[1]) / length(aug.Zraw_all[1])
x_free_ext = vcat(x_free_calib, fill(nu1_guess, K_mean))
m_ext = CS.FreeParamMap(ctx.l_full + K_mean, vcat(ctx.free_idx, [ctx.l_full + k for k in 1:K_mean]),
                         ctx.fixed_idx, ctx.fixed_vals)
θ_ext = CS.reconstruct_full(x_free_ext, m_ext)

K = zeros(size(ctx.U, 1))
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_ext, objA.U, objA)
objA.H[:, 1] .= K
objA.H[:, 2] .= 1.0

println("="^100)
println("Threaded/v2 dispatch (hessian_cm_structured_v2!, threaded_bins=true) vs Architecture A")
println("and vs the already-gated serial hessian_cm_structured!, at the genuinely ZC-widened,")
println("two-family CM+ZC point (K_mean=1, K_pair=0)")
println("="^100)

Random.seed!(4044)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]

cctx = build_cm_meanzc_bin_ctx(ctx, aug; threaded_bins = true)
@assert cctx.n_families == 2 && cctx.ncore_core < cctx.NCORE
@assert cctx.use_threaded_bins "expected threaded_bins=true to actually set cctx.use_threaded_bins"
lp_cht = cctx.cross_hessian_threaded
println("cctx.use_threaded_bins=", cctx.use_threaded_bins, "  cctx.cross_hessian_threaded=", lp_cht,
        "  cctx.hcz_prep_backend=", cctx.hcz_prep_backend)
cctx.nu_ref[] = collect([nu1_guess])
cf = cf_build(θ_ext, ctx; check_ties = false)
cctx.core_cf_ref[] = cf
tls = build_thread_local_scratch(cctx)

for (pi_, x) in enumerate(xs)
    hA = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    objA(x, h = hA)
    HA = unpack_packed(hA, n)

    hS = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _archC_prep_for_hessian!(objA, x)
    hessian_cm_structured!(hS, objA, cctx)
    HS = unpack_packed(hS, n)
    errS = maximum(abs.(HS .- HA))

    hV2 = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _archC_prep_for_hessian!(objA, x)
    hessian_cm_structured_v2!(hV2, objA, cctx; threaded_bins = true, tls = tls)
    HV2 = unpack_packed(hV2, n)
    errV2 = maximum(abs.(HV2 .- HA))
    errV2vsS = maximum(abs.(HV2 .- HS))

    used_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cctx.core_cf_ref[])
    used_direct_hcz = used_winner_bin && _cm_cross_hessian_wants_direct_hcz(cctx, cctx.core_cf_ref[])
    check("point $pi_: used_winner_bin=$used_winner_bin used_direct_hcz=$used_direct_hcz, " *
          "v2(threaded) vs Architecture A (max|Δ|=$errV2), v2 vs serial (max|Δ|=$errV2vsS)",
          errV2 < 1e-8 && errV2vsS < 1e-8 && used_winner_bin && used_direct_hcz)
end

println("="^80)
if isempty(FAILURES)
    println("ALL CM+ZC THREADED WIDENED-HCZ POW GATES PASS")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
