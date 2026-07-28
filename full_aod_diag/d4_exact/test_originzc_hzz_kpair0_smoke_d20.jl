# CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): real D=20/W=80,000 K_pair=0
# (mean-only) smoke for origin-ZC's shared direct H_ZZ (HRR) primitive -- status/no-crash check
# only (not a full residual comparison), per the task's own gate-3 requirement. The main
# correctness gate for origin-ZC's H_ZZ at K_mean=1/K_pair=1 is
# test_originzc_winner_bin_her_wiring_d20.jl (unmodified, re-run this session -- its own
# `zc_cross_hessian_backend=:winner_bin` octx now ALSO exercises the NEW direct H_ZZ routine for
# HRR, since that flag was extended to cover HRR in addition to the pre-existing HER).
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
Random.seed!(2027)

reset_no_dense_g_counters!()

println("=== K_pair=0 mean-only smoke (origin-ZC) ==="); flush(stdout)
K_mean0, K_pair0 = 1, 0
ν00 = nu0vec(K_mean0)
layout0 = SharedByPowerLayout(K_mean0, K_pair0)
aug0 = build_originzc_augmented_obj(ctx, CS, layout0)
ctx_cm0 = merge(ctx, (obj = aug0.obj_cm,))
octx0 = build_originzc_core_hess_ctx(aug0; zc_cross_hessian_backend = :winner_bin)
ctx_cm0 = merge(ctx_cm0, (octx = octx0,))
check("K_pair=0 smoke: hzz_zc_op built", octx0.hzz_zc_op !== nothing)

@time base0 = archOZ_base_state(x_free_calib, ν00, ctx_cm0)
check("K_pair=0 smoke: inner solve feasible (status=$(base0.inner_status))", base0.inner_status in (0, -100, -101, -103))

n0 = octx0.NCORE + octx0.n_eta
x0v = vcat(base0.ζstar, base0.λstar)
@time h0 = packed_hess_via_octx(octx0, ctx_cm0.obj, x0v, n0)
check("K_pair=0 smoke: Hessian call completed with no NaN/Inf", all(isfinite, h0))

println()
println("Winner cross-Hessian counters: ", no_dense_g_report())
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
