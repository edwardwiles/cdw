# ================================================================================================
# True no-H operator bundle task, Part A generalization: origin-ZC (ZC-only) operator-bundle (no H)
# vs dense-reference equivalence gate (real KNITRO D=4). Mirrors
# test_operator_no_H_bundle_equivalence_flexcm.jl, adapted to
# build_originzc_production_context/archOZ_base_state/OriginZCOperatorState/
# archA_partitioned_hess_cb_builder.
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl",
          "cm_originzc_moments.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl", "cm_originzc_production.jl"]
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
Random.seed!(20260728)
layout = OriginByPowerLayout(ctx.D, 1, 0)   # K_mean=1, K_pair=0
νfull0 = fill(1.0, ctx.D)   # n_eta(layout) == D for K_mean=1,K_pair=0

println("="^90)
println("Origin-ZC: operator-bundle (no H) vs dense-reference bundle -- equivalence gate")
println("="^90)

pcx_d = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, moment_representation = :dense_reference)
pcx_o = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, moment_representation = :operator)
pcx_d.octx.core_hessian_backend = :exact_winner_pair_parallel
pcx_o.octx.core_hessian_backend = :exact_winner_pair_parallel

obj_o = pcx_o.ctx_cm.obj
obj_d = pcx_d.ctx_cm.obj
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

base_d0 = archOZ_base_state(x_free_calib, νfull0, pcx_d.ctx_cm)
base_o0 = archOZ_base_state(x_free_calib, νfull0, pcx_o.ctx_cm)
check("both backends: calibration inner solve feasible",
      base_d0.inner_status in (0,-100,-101,-103) && base_o0.inner_status in (0,-100,-101,-103))
lp("  nStatus_dense=", base_d0.inner_status, "  nStatus_operator=", base_o0.inner_status)

n = pcx_d.octx.NCORE + pcx_d.octx.n_eta

function full_hessian_oz(octx, obj, x::AbstractVector)
    _prep_dual_index_for_archA!(octx, obj, x)
    h = Vector{Float64}(undef, n*(n+1)÷2)
    cb = archA_partitioned_hess_cb_builder(octx)
    fake_req = (x = x,)
    fake_res = (hess = h,)
    cb(nothing, nothing, fake_req, fake_res, obj)
    return unpack_packed(h, n)
end

function compare_at(x::AbstractVector; label = "")
    st_d = OriginZCOperatorState(obj_d, pcx_d.octx.NCORE - 1, pcx_d.octx.fg_zc_op, pcx_d.octx.fg_layout, pcx_d.octx.core_cf_ref)
    st_o = OriginZCOperatorState(obj_o, pcx_o.octx.NCORE - 1, pcx_o.octx.fg_zc_op, pcx_o.octx.fg_layout, pcx_o.octx.core_cf_ref)
    reset_for_solve!(st_d, νfull0)
    reset_for_solve!(st_o, νfull0)

    θ_econ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
    θ_ext_calib = vcat(θ_econ_calib, νfull0)
    obj_d.moments!(@view(obj_d.H[:, 1]), CS.select_G_from_H(obj_d, obj_d.H), θ_ext_calib, obj_d.U, obj_d)
    obj_d.H[:, 2] .= 1.0
    prime_operator!(obj_o, θ_econ_calib, ctx, pcx_o.octx.core_cf_ref; restriction_state = pcx_o.octx)
    pcx_o.octx.nu_ref[] = collect(νfull0)
    pcx_d.octx.nu_ref[] = collect(νfull0)

    g_d = zeros(n); g_o = zeros(n)
    f_d = st_d(x, g_d)
    f_o = st_o(x, g_o)
    e_f = abs(f_d - f_o)
    e_g = maximum(abs.(g_d .- g_o))

    Hd = full_hessian_oz(pcx_d.octx, obj_d, x)
    Ho = full_hessian_oz(pcx_o.octx, obj_o, x)
    e_H = maximum(abs.(Hd .- Ho))

    @printf("  %-20s |f_d-f_o|=%.3e  max|Δg|=%.3e  max|ΔH|=%.3e\n", label, e_f, e_g, e_H)
    check("$label: objective agrees", e_f < 1e-10)
    check("$label: gradient agrees", e_g < 1e-9)
    check("$label: packed Hessian agrees", e_H < 1e-8)
end

compare_at(zeros(n); label = "x=0")
for i in 1:4
    compare_at(0.05 .* randn(n); label = "random[$i]")
end
compare_at(vcat(base_d0.ζstar, base_d0.λstar); label = "real solved x* (dense)")

lp("="^90); lp("Full inner solves from x_free_calib")
base_d = archOZ_base_state(x_free_calib, νfull0, pcx_d.ctx_cm)
base_o = archOZ_base_state(x_free_calib, νfull0, pcx_o.ctx_cm)
check("full solve: both backends same accepted status", base_d.inner_status == base_o.inner_status)
e_zeta = abs(base_d.ζstar - base_o.ζstar)
e_lambda = maximum(abs.(base_d.λstar .- base_o.λstar))
lp("  Δζ*=$e_zeta   max|Δλ*|=$e_lambda   status_d=$(base_d.inner_status) status_o=$(base_o.inner_status)")
check("full solve: delta* (ζ*) agrees to 1e-10", e_zeta < 1e-10)
check("full solve: dual vector (λ*) agrees to 1e-8", e_lambda < 1e-8)

println("="^90)
if isempty(FAILURES)
    println("ALL OPERATOR-VS-DENSE ORIGIN-ZC EQUIVALENCE GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
