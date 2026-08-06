# 2026-08-05: user-requested independent ground truth for the two-family (eq.35+eq.36) CM
# gradient/Hessian, via ForwardDiff rather than comparing Architecture C against a hand-derived
# "brute-force" dense reference -- the D4 gates in this same task (test_cm_archc_hcc_twofamily_*,
# test_cm_archc_full_twofamily_*, test_cm_lookup_fg_twofamily_*) all compare Architecture C against
# EITHER Architecture A's own analytic H (objA(x,h=...)) OR a hand-typed `raw_A`/`raw_B` brute-force
# formula written in THIS SAME session with the SAME mental model of the eq.36 indicator direction
# -- so a shared conceptual mistake (e.g. still getting `<=` vs `>` wrong in some way both the fix
# and the "independent" brute-force check agree on) would NOT be caught by any of those. This script
# is deliberately independent of ALL of that: it reads ONLY the raw feature matrix
# `objA.H[:, 3:n+1]` (== G, the actual production moment-feature columns objA.moments! populates,
# including the two-family CM columns) and the known closed-form Psi(a) = e^a-1 (a<=1) /
# (e/2)(a^2+1)-1 (a>1) (cc_algo/Psi.jl), then differentiates
#     f(zeta,lambda) = (1/M) * sum_s Psi(-zeta - dot(G[s,:], lambda)) + zeta
# via ForwardDiff -- a fully mechanical computation that cannot inherit any Architecture-C-specific
# or brute-force-specific bug. Compared against:
#   - gradient: CMLookupState's own functor `st(x, g)` -- the REAL operator-bundle FG path
#     production's inner KNITRO solve calls every iteration (cm_lookup_kernels.jl).
#   - Hessian: `hessian_cm_structured!(h, objA, cctx)` on a winner-bin-eligible cctx (real
#     `core_cf_ref` populated via `cf_build`) -- the REAL no-dense-H production Hessian path
#     (archC_hess_cb_builder wraps exactly this function for KNITRO).
# If G itself already has a further, unfixed sign/indicator bug, both the operator-bundle output AND
# this autodiff ground truth would use the SAME (still-wrong) G and could still agree with each
# other while being jointly wrong about the ECONOMIC MEANING of eq.36 -- but they would no longer
# be free to silently share a HESSIAN/GRADIENT ASSEMBLY bug, which is the specific risk this script
# targets (whether hessian_cm_structured!'s new T12/T22/winner-bin-Pow/H_CZ-Pow reflected-table
# machinery correctly computes d/dlambda and d^2/dlambda^2 of the TRUE objective GIVEN G).
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
using Printf, LinearAlgebra, Random, ForwardDiff

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

# Independent reimplementation of cc_algo/Psi.jl's Psi(a) -- copied by VALUE (the two-piece
# exp/quadratic formula, not by calling Psi!) so this file has zero code-sharing with the
# production Psi!/dPsi!/ddPsi! implementation, only the same well-known closed form.
psi_ad(a::Real) = a <= 1.0 ? exp(a) - 1.0 : (0.5 * exp(1)) * (a^2 + 1.0) - 1.0

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
L = 4

println("="^100)
println("Autodiff ground-truth check: winner-bin (real production) path, two-family CM, D4")
println("="^100)

aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true)
@assert aug.n_families == 2
objA = aug.obj_cm
n = objA.outer_constr_index
M = size(ctx.U, 1)
K = zeros(M)
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
objA.H[:, 1] .= K
objA.H[:, 2] .= 1.0

# G is EXACTLY what production's inner objective contracts lambda against (arg0 = -zeta*1 - G*lambda,
# PsiObjectiveBundleImplicit's own functor, cc_algo/PsiObjectiveBundle.jl) -- read once as a plain
# Float64 matrix, no further processing.
G = Matrix(objA.H[:, 3:n+1])
@assert size(G) == (M, n - 1)

f_autodiff(xv) = sum(psi_ad.(.-xv[1] .- G * xv[2:end])) / M + xv[1]

bins_u = let
    cctx0 = build_cm_bin_ctx(ctx, aug)
    cctx0.Bidx isa Matrix{UInt32} ? cctx0.Bidx : Matrix{UInt32}(cctx0.Bidx)
end
cctx = build_cm_bin_ctx(ctx, aug)
@assert cctx.n_families == 2 && cctx.Pow !== nothing
cf = cf_build(θ_full, ctx; check_ties = false)
cctx.core_cf_ref[] = cf
used_winner_bin_any = false

st = CMLookupState(objA, cctx.NCORE, cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R;
                    method = :suffix, Pow = cctx.Pow)

ncore1 = cctx.NCORE - 1
ncm_cdf = cctx.nO * cctx.L
ncm_total = 2 * ncm_cdf

Random.seed!(90210)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n), 0.2 .* randn(n)]
for (pi_, x) in enumerate(xs)
    # ---- ground truth via ForwardDiff ----
    g_ad = ForwardDiff.gradient(f_autodiff, x)
    H_ad = ForwardDiff.hessian(f_autodiff, x)

    # ---- operator-bundle gradient (CMLookupState, real per-iterate FG path) ----
    g_op = Vector{Float64}(undef, n)
    st(x, g_op)

    # ---- Architecture C Hessian (hessian_cm_structured!, real winner-bin production path) ----
    hC = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _archC_prep_for_hessian!(objA, x)
    hessian_cm_structured!(hC, objA, cctx)
    used_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cctx.core_cf_ref[])
    global used_winner_bin_any |= used_winner_bin
    H_op = let Mx = Matrix{Float64}(undef, n, n); k = 1
        @inbounds for i in 1:n, j in i:n
            Mx[i, j] = hC[k]; Mx[j, i] = hC[k]; k += 1
        end
        Mx
    end

    gerr_all = maximum(abs.(g_ad .- g_op))
    Herr_all = maximum(abs.(H_ad .- H_op))
    check("point $pi_ (winner_bin=$used_winner_bin): full gradient matches ForwardDiff (max|Δg|=$gerr_all)", gerr_all < 1e-6)
    check("point $pi_: full Hessian matches ForwardDiff (max|Δh|=$Herr_all)", Herr_all < 1e-6)

    # Localize to CM sub-blocks specifically (CDF half, POW half, and the core-CM cross block).
    cm_cdf_idx = (2 + ncore1):(1 + ncore1 + ncm_cdf)
    cm_pow_idx = (2 + ncore1 + ncm_cdf):(1 + ncore1 + ncm_total)
    core_idx = 1:(1 + ncore1)   # includes zeta (index 1)

    g_cdf_err = maximum(abs.(g_ad[cm_cdf_idx] .- g_op[cm_cdf_idx]))
    g_pow_err = maximum(abs.(g_ad[cm_pow_idx] .- g_op[cm_pow_idx]))
    check("point $pi_: g CDF-block matches ForwardDiff (max|Δ|=$g_cdf_err)", g_cdf_err < 1e-6)
    check("point $pi_: g POW-block matches ForwardDiff (max|Δ|=$g_pow_err)", g_pow_err < 1e-6)

    H_cc_cdf_err = maximum(abs.(H_ad[cm_cdf_idx, cm_cdf_idx] .- H_op[cm_cdf_idx, cm_cdf_idx]))
    H_cc_pow_err = maximum(abs.(H_ad[cm_pow_idx, cm_pow_idx] .- H_op[cm_pow_idx, cm_pow_idx]))
    H_cc_cross_err = maximum(abs.(H_ad[cm_cdf_idx, cm_pow_idx] .- H_op[cm_cdf_idx, cm_pow_idx]))
    H_ec_cdf_err = maximum(abs.(H_ad[core_idx, cm_cdf_idx] .- H_op[core_idx, cm_cdf_idx]))
    H_ec_pow_err = maximum(abs.(H_ad[core_idx, cm_pow_idx] .- H_op[core_idx, cm_pow_idx]))
    check("point $pi_: H_CC[cdf,cdf] matches ForwardDiff (max|Δ|=$H_cc_cdf_err)", H_cc_cdf_err < 1e-6)
    check("point $pi_: H_CC[pow,pow] matches ForwardDiff (max|Δ|=$H_cc_pow_err)", H_cc_pow_err < 1e-6)
    check("point $pi_: H_CC[cdf,pow] matches ForwardDiff (max|Δ|=$H_cc_cross_err)", H_cc_cross_err < 1e-6)
    check("point $pi_: H_EC[core,cdf] matches ForwardDiff (max|Δ|=$H_ec_cdf_err)", H_ec_cdf_err < 1e-6)
    check("point $pi_: H_EC[core,pow] matches ForwardDiff (max|Δ|=$H_ec_pow_err)", H_ec_pow_err < 1e-6)
end
check("winner-bin path was actually exercised at least once (not silently falling back)", used_winner_bin_any)

println("="^80)
if isempty(FAILURES)
    println("ALL AUTODIFF GROUND-TRUTH GATES PASS")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
