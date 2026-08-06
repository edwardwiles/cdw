# 2026-08-05 truncated-power task: CM+ZC's own two-family Architecture C gate -- dense-H-forced
# path only (this task's disclosed scope: CM+ZC's widened-row H_CZ cross was not given its own
# Pow extension, so `hessian_cm_structured!` forces `use_winner_bin=false` whenever a two-family
# context is ALSO ZC-widened -- see that function's own top-of-body comment). Same
# "Architecture C vs Architecture A at a fixed point" methodology as the plain-flexible-CM gate.
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

# 2026-08-05: CM+ZC's θ_free is EXTENDED with K_mean nu-target parameters (theta_econ_free ++
# nu_1..nu_K_mean) -- NOT just the plain economic theta_full -- matching
# test_cm_meanzc_truncated_power_d4_2026-08-05.jl's own already-validated setup exactly (a naive
# plain theta_full here throws a BoundsError deep inside cf_build/constCons_matrix, confirmed live
# -- a test-setup bug, not a production defect: reconstruct_full via the UNEXTENDED ctx.m produces
# a too-short theta vector for this widened obj_cm).
nu1_guess = sum(aug.Zraw_all[1]) / length(aug.Zraw_all[1])
x_free_ext = vcat(x_free_calib, fill(nu1_guess, K_mean))
m_ext = CS.FreeParamMap(ctx.l_full + K_mean, vcat(ctx.free_idx, [ctx.l_full + k for k in 1:K_mean]),
                         ctx.fixed_idx, ctx.fixed_vals)
θ_ext = CS.reconstruct_full(x_free_ext, m_ext)

K = zeros(size(ctx.U, 1))
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_ext, objA.U, objA)
objA.H[:, 1] .= K
objA.H[:, 2] .= 1.0

cctx = build_cm_meanzc_bin_ctx(ctx, aug)
@assert cctx.n_families == 2
@assert cctx.ncore_core < cctx.NCORE   # genuinely ZC-widened
cctx.nu_ref[] = collect([nu1_guess])   # required before any CM+ZC Hessian call -- see
# archC_meanzc_base_state's own identical line (cm_meanzc_production.jl); refresh_zc_targets!
# (zc_restriction_operator.jl) reads this to build the current outer point's mean/pair targets.

Random.seed!(4044)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
for (pi_, x) in enumerate(xs)
    hA = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    objA(x, h = hA)
    HA = unpack_packed(hA, n)

    hC = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _archC_prep_for_hessian!(objA, x)
    hessian_cm_structured!(hC, objA, cctx)
    HC = unpack_packed(hC, n)
    errC = maximum(abs.(HC .- HA))
    check("point $pi_: full packed Hessian matches Architecture A (max|Δ|=$errC)", errC < 1e-8)
end

println("="^100)
println("Section 2: winner-bin (Pow-extended) H_EC + H_CZ path -- the genuine no-dense-H CM+ZC")
println("production path (user directive, second follow-up: no uses of dense H anywhere)")
println("="^100)
let
    cctx2 = build_cm_meanzc_bin_ctx(ctx, aug)
    @assert cctx2.n_families == 2 && cctx2.ncore_core < cctx2.NCORE
    cctx2.nu_ref[] = collect([nu1_guess])
    cf = cf_build(θ_ext, ctx; check_ties = false)
    cctx2.core_cf_ref[] = cf

    for (pi_, x) in enumerate(xs)
        hA = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        objA(x, h = hA)
        HA = unpack_packed(hA, n)

        hC = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        _archC_prep_for_hessian!(objA, x)
        hessian_cm_structured!(hC, objA, cctx2)
        HC = unpack_packed(hC, n)
        errC = maximum(abs.(HC .- HA))
        used_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx2, cctx2.core_cf_ref[])
        used_direct_hcz = used_winner_bin && _cm_cross_hessian_wants_direct_hcz(cctx2, cctx2.core_cf_ref[])
        check("winner-bin point $pi_: used_winner_bin=$used_winner_bin used_direct_hcz=$used_direct_hcz, full packed Hessian matches Architecture A (max|Δ|=$errC)",
              errC < 1e-8 && used_winner_bin && used_direct_hcz)
    end
end

println("="^80)
if isempty(FAILURES)
    println("ALL CM+ZC TWO-FAMILY GATES PASS")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
