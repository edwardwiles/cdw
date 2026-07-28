# ================================================================================================
# True no-H operator bundle task, Part A.7: unrestricted real D=20/W=100,000 equivalence gate.
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "context_real_d20.jl"]
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

lp("Building real D=20/W=100,000 context...")
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(20260728)
lp("Context built.")

println("="^90)
println("Unrestricted D=20/W=100,000: operator-bundle (no H) vs dense-reference -- equivalence gate")
println("="^90)

obj_d = ctx.obj
obj_o = OperatorPsiBundle(δ = obj_d.δ, find_smallest = obj_d.find_smallest,
    γ = obj_d.γ, l = obj_d.l, outer_constr_index = obj_d.outer_constr_index,
    inequality_index = obj_d.inequality_index, complement_index = obj_d.complement_index,
    U = obj_d.U, N = obj_d.N, lower_limit = obj_d.lower_limit,
    use_cached_x = obj_d.use_cached_x, threshold_state = obj_d.threshold_state,
    inner_loop_opt = obj_d.inner_loop_opt)

fn_o = fieldnames(typeof(obj_o))
forbidden = (:H, :H_copy, :G, :K, :ones, :jac_h, Symbol("moments!"))
for f in forbidden
    check("OperatorPsiBundle has no field :$f", !(f in fn_o))
end

UNRESTRICTED_CORE_HESSIAN_BACKEND[] = :exact_winner_pair_parallel

θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
n = obj_d.outer_constr_index

lp("="^90); lp("Full REAL inner solves (D=20/W=100,000)")
t0 = time()
K_d, x_d, nStatus_d, _, _, st_d0 = inner_loop_internal_compressed(obj_d, θ_full_calib, ctx)
lp("  dense solve done in ", time() - t0, "s, nStatus=", nStatus_d)
t0 = time()
K_o, x_o, nStatus_o, _, _, st_o0 = inner_loop_internal_compressed(obj_o, θ_full_calib, ctx)
lp("  operator solve done in ", time() - t0, "s, nStatus=", nStatus_o)
check("both backends: calibration inner solve feasible", nStatus_d in (0,-100,-101,-103) && nStatus_o in (0,-100,-101,-103))

function compare_at(x::AbstractVector; label = "")
    cf_d = build_economic_moment_state!(θ_full_calib, ctx; check_ties = false)
    cf_o = build_economic_moment_state!(θ_full_calib, ctx; check_ties = false)
    grav_raw = compressed_gravity_raw(θ_full_calib, ctx)
    st_d = CompressedCBState(obj_d, cf_d, grav_raw, false)
    st_o = CompressedCBState(obj_o, cf_o, grav_raw, false)

    g_d = zeros(n); g_o = zeros(n)
    t0 = time()
    res_d = (obj = [0.0], objGrad = g_d); res_o = (obj = [0.0], objGrad = g_o)
    _callbackEvalFG_inner_compressed!(nothing, nothing, (x = x,), res_d, st_d)
    _callbackEvalFG_inner_compressed!(nothing, nothing, (x = x,), res_o, st_o)
    e_f = abs(res_d.obj[1] - res_o.obj[1])
    e_g = maximum(abs.(g_d .- g_o))

    h_d = Vector{Float64}(undef, n*(n+1)÷2); h_o = Vector{Float64}(undef, n*(n+1)÷2)
    _callbackEvalH_inner_compressed!(nothing, nothing, (x = x,), (hess = h_d,), st_d)
    _callbackEvalH_inner_compressed!(nothing, nothing, (x = x,), (hess = h_o,), st_o)
    e_H = maximum(abs.(h_d .- h_o))
    dt = time() - t0

    @printf("  %-20s |f_d-f_o|=%.3e  max|Δg|=%.3e  max|ΔH|=%.3e  (%.1fs)\n", label, e_f, e_g, e_H, dt)
    check("$label: objective agrees", e_f < 1e-8)
    check("$label: gradient agrees", e_g < 1e-7)
    check("$label: packed Hessian agrees", e_H < 1e-6)
end

compare_at(zeros(n); label = "x=0")
for i in 1:2
    compare_at(0.05 .* randn(n); label = "random[$i]")
end
compare_at(x_d; label = "real solved x*")

check("full solve: both backends same accepted status", nStatus_d == nStatus_o)
e_zeta = abs(x_d[1] - x_o[1])
e_lambda = maximum(abs.(x_d[2:end] .- x_o[2:end]))
lp("  Δζ*=$e_zeta   max|Δλ*|=$e_lambda   status_d=$nStatus_d status_o=$nStatus_o")
check("full solve: delta* (ζ*) agrees to 1e-8", e_zeta < 1e-8)
check("full solve: dual vector (λ*) agrees to 1e-6", e_lambda < 1e-6)

println("="^90)
if isempty(FAILURES)
    println("ALL OPERATOR-VS-DENSE UNRESTRICTED D=20/W=100000 EQUIVALENCE GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
