# ================================================================================================
# True no-H operator bundle task, Part A generalization: unrestricted operator-bundle (no H) vs
# dense-reference equivalence gate (real KNITRO D=4). Mirrors
# test_operator_no_H_bundle_equivalence_flexcm.jl, adapted to compressed_live.jl's
# inner_loop_internal_compressed/CompressedCBState/_callbackEvalH_inner_compressed!.
#
# Unlike the 4 restricted families, ctx.obj (from d4_exact_setup) IS the production bundle for
# this family directly (no separate build_*_production_context wrapper) -- this test builds a
# SEPARATE OperatorPsiBundle from ctx.obj's own scalar fields (same pattern every restricted
# family's build_*_production_context already uses), leaving ctx.obj itself untouched (it also
# serves as the scalar-field TEMPLATE the other 4 families' builders read from).
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl"]
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
lp(xs...) = (println(xs...); flush(stdout))

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(20260728)

println("="^90)
println("Unrestricted: operator-bundle (no H) vs dense-reference bundle -- equivalence gate")
println("="^90)

obj_d = ctx.obj   # the production dense bundle, unchanged
obj_o = OperatorPsiBundle(δ = obj_d.δ, find_smallest = obj_d.find_smallest,
    γ = obj_d.γ, l = obj_d.l, outer_constr_index = obj_d.outer_constr_index,
    inequality_index = obj_d.inequality_index, complement_index = obj_d.complement_index,
    U = obj_d.U, N = obj_d.N, lower_limit = obj_d.lower_limit,
    use_cached_x = obj_d.use_cached_x, threshold_state = obj_d.threshold_state,
    inner_loop_opt = obj_d.inner_loop_opt)

fn_o = fieldnames(typeof(obj_o))
lp("OperatorPsiBundle fieldnames: ", fn_o)
forbidden = (:H, :H_copy, :G, :K, :ones, :jac_h, Symbol("moments!"))
for f in forbidden
    check("OperatorPsiBundle has no field :$f", !(f in fn_o))
end
let threw = false
    try; obj_o.H; catch e; threw = true; end
    check("obj_o.H throws", threw)
end

UNRESTRICTED_CORE_HESSIAN_BACKEND[] = :exact_winner_pair_parallel

θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
K_d, x_d0, nStatus_d0, _, _, st_d0 = inner_loop_internal_compressed(obj_d, θ_full_calib, ctx)
K_o, x_o0, nStatus_o0, _, _, st_o0 = inner_loop_internal_compressed(obj_o, θ_full_calib, ctx)
check("both backends: calibration inner solve feasible",
      nStatus_d0 in (0,-100,-101,-103) && nStatus_o0 in (0,-100,-101,-103))
lp("  nStatus_dense=", nStatus_d0, "  nStatus_operator=", nStatus_o0)

n = obj_d.outer_constr_index

function full_hessian_unrestricted(obj, st, x::AbstractVector)
    _callbackEvalFG_inner_compressed!(nothing, nothing, (x = x,), (obj = [0.0], objGrad = zeros(n)), st)
    h = Vector{Float64}(undef, n*(n+1)÷2)
    _callbackEvalH_inner_compressed!(nothing, nothing, (x = x,), (hess = h,), st)
    return h
end

function compare_at(x::AbstractVector; label = "")
    cf_d = build_economic_moment_state!(θ_full_calib, ctx; check_ties = false)
    cf_o = build_economic_moment_state!(θ_full_calib, ctx; check_ties = false)
    grav_raw = compressed_gravity_raw(θ_full_calib, ctx)
    st_d = CompressedCBState(obj_d, cf_d, grav_raw, false)
    st_o = CompressedCBState(obj_o, cf_o, grav_raw, false)

    g_d = zeros(n); g_o = zeros(n)
    res_d = (obj = [0.0], objGrad = g_d); res_o = (obj = [0.0], objGrad = g_o)
    _callbackEvalFG_inner_compressed!(nothing, nothing, (x = x,), res_d, st_d)
    _callbackEvalFG_inner_compressed!(nothing, nothing, (x = x,), res_o, st_o)
    e_f = abs(res_d.obj[1] - res_o.obj[1])
    e_g = maximum(abs.(g_d .- g_o))

    h_d = Vector{Float64}(undef, n*(n+1)÷2); h_o = Vector{Float64}(undef, n*(n+1)÷2)
    _callbackEvalH_inner_compressed!(nothing, nothing, (x = x,), (hess = h_d,), st_d)
    _callbackEvalH_inner_compressed!(nothing, nothing, (x = x,), (hess = h_o,), st_o)
    e_H = maximum(abs.(h_d .- h_o))

    @printf("  %-20s |f_d-f_o|=%.3e  max|Δg|=%.3e  max|ΔH|=%.3e\n", label, e_f, e_g, e_H)
    check("$label: objective agrees", e_f < 1e-10)
    check("$label: gradient agrees", e_g < 1e-9)
    check("$label: packed Hessian agrees", e_H < 1e-8)
end

compare_at(zeros(n); label = "x=0")
for i in 1:4
    compare_at(0.05 .* randn(n); label = "random[$i]")
end
compare_at(vcat(x_d0); label = "real solved x* (dense)")

lp("="^90); lp("Full inner solves from x_free_calib")
_, x_d, nStatus_d, _, _, _ = inner_loop_internal_compressed(obj_d, θ_full_calib, ctx)
_, x_o, nStatus_o, _, _, _ = inner_loop_internal_compressed(obj_o, θ_full_calib, ctx)
check("full solve: both backends same accepted status", nStatus_d == nStatus_o)
e_zeta = abs(x_d[1] - x_o[1])
e_lambda = maximum(abs.(x_d[2:end] .- x_o[2:end]))
lp("  Δζ*=$e_zeta   max|Δλ*|=$e_lambda   status_d=$nStatus_d status_o=$nStatus_o")
check("full solve: delta* (ζ*) agrees to 1e-10", e_zeta < 1e-10)
check("full solve: dual vector (λ*) agrees to 1e-8", e_lambda < 1e-8)

println("="^90)
if isempty(FAILURES)
    println("ALL OPERATOR-VS-DENSE UNRESTRICTED EQUIVALENCE GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
