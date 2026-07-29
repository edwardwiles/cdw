# ================================================================================================
# True no-H operator bundle task, Part A.7: flexible-CM real D=20/W=100,000 equivalence gate.
# Same structure as the D=4 gate, run on real production data. Reduced random-probe count (2, not
# 4) to keep wall-clock reasonable -- the full real inner solve (the expensive, most important
# check) is still run for both backends at calibration.
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "context_real_d20.jl"]
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

lp("Building real D=20/W=100,000 context...")
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(20260728)
lp("Context built. D=", ctx.D, " D_dest=", hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D, " W=", size(ctx.U,1))

println("="^90)
println("Flexible-CM D=20/W=100,000: operator-bundle (no H) vs dense-reference -- equivalence gate")
println("="^90)

pcx_d = build_cm_production_context(ctx, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = true, moment_representation = :dense_reference)
pcx_o = build_cm_production_context(ctx, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = true, moment_representation = :operator)
pcx_d.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_o.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_d.cctx.cm_cross_hessian_backend = :winner_bin
pcx_o.cctx.cm_cross_hessian_backend = :winner_bin

obj_o = pcx_o.ctx_cm.obj
obj_d = pcx_d.ctx_cm.obj
fn_o = fieldnames(typeof(obj_o))
forbidden = (:H, :H_copy, :G, :K, :ones, :jac_h, Symbol("moments!"))
for f in forbidden
    check("OperatorPsiBundle has no field :$f", !(f in fn_o))
end

n = pcx_d.cctx.NCORE + pcx_d.cctx.ncm
NCORE = pcx_d.cctx.NCORE

function full_hessian_generic(ctx_cm, cctx, x::AbstractVector)
    obj = ctx_cm.obj
    _prep_dual_index_for_archC!(cctx, obj, x)
    h = Vector{Float64}(undef, n*(n+1)÷2)
    hessian_cm_structured!(h, obj, cctx)
    return unpack_packed(h, n)
end

function compare_at(x::AbstractVector; label = "")
    bins_u = pcx_d.cctx.Bidx isa Matrix{UInt32} ? pcx_d.cctx.Bidx : Matrix{UInt32}(pcx_d.cctx.Bidx)
    st_d = CMLookupState(obj_d, pcx_d.cctx.NCORE, pcx_d.cctx.ncm, pcx_d.cctx.L, pcx_d.cctx.origins, pcx_d.cctx.refIndex1,
                          bins_u, pcx_d.cctx.R; method = :suffix, core_cf_ref = pcx_d.cctx.core_cf_ref)
    st_o = CMLookupState(obj_o, pcx_o.cctx.NCORE, pcx_o.cctx.ncm, pcx_o.cctx.L, pcx_o.cctx.origins, pcx_o.cctx.refIndex1,
                          bins_u, pcx_o.cctx.R; method = :suffix, core_cf_ref = pcx_o.cctx.core_cf_ref)
    θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
    obj_d.moments!(@view(obj_d.H[:, 1]), CS.select_G_from_H(obj_d, obj_d.H), θ_full_calib, obj_d.U, obj_d)
    obj_d.H[:, 2] .= 1.0
    prime_operator!(obj_o, θ_full_calib, ctx, pcx_o.cctx.core_cf_ref; restriction_state = pcx_o.cctx)

    g_d = zeros(n); g_o = zeros(n)
    t0 = time()
    f_d = st_d(x, g_d)
    f_o = st_o(x, g_o)
    e_f = abs(f_d - f_o)
    e_g = maximum(abs.(g_d .- g_o))

    Hd = full_hessian_generic(pcx_d.ctx_cm, pcx_d.cctx, x)
    Ho = full_hessian_generic(pcx_o.ctx_cm, pcx_o.cctx, x)
    e_H = maximum(abs.(Hd .- Ho))
    dt = time() - t0

    @printf("  %-20s |f_d-f_o|=%.3e  max|Δg|=%.3e  max|ΔH|=%.3e  (%.1fs)\n", label, e_f, e_g, e_H, dt)
    check("$label: objective agrees", e_f < 1e-8)
    check("$label: gradient agrees", e_g < 1e-7)
    check("$label: packed Hessian agrees", e_H < 1e-6)
end

lp("="^90); lp("Full REAL inner solves from x_free_calib (D=20/W=100,000)")
# Run BEFORE the FG/Hessian callback comparisons below: archC_base_state's own
# inner_loop_internal_cmlookup_production populates cctx.cmlookup_st (needed by
# _prep_dual_index_for_archC!'s dense-G-free dispatch) as a side effect -- matching the D=4 test's
# own call order, not incidental.
t0 = time()
base_d = archC_base_state(x_free_calib, pcx_d.ctx_cm, pcx_d.cctx)
lp("  dense solve done in ", time() - t0, "s, nStatus=", base_d.inner_status)
t0 = time()
base_o = archC_base_state(x_free_calib, pcx_o.ctx_cm, pcx_o.cctx)
lp("  operator solve done in ", time() - t0, "s, nStatus=", base_o.inner_status)

compare_at(zeros(n); label = "x=0")
for i in 1:2
    compare_at(0.05 .* randn(n); label = "random[$i]")
end

check("full solve: both backends same accepted status", base_d.inner_status == base_o.inner_status)
e_zeta = abs(base_d.ζstar - base_o.ζstar)
e_lambda = maximum(abs.(base_d.λstar .- base_o.λstar))
lp("  Δζ*=$e_zeta   max|Δλ*|=$e_lambda   status_d=$(base_d.inner_status) status_o=$(base_o.inner_status)")
check("full solve: delta* (ζ*) agrees to 1e-8", e_zeta < 1e-8)
check("full solve: dual vector (λ*) agrees to 1e-6", e_lambda < 1e-6)

# Structural field/allocation proof at the real solved point too
compare_at(vcat(base_d.ζstar, base_d.λstar); label = "real solved x*")

println("="^90)
if isempty(FAILURES)
    println("ALL OPERATOR-VS-DENSE FLEXIBLE-CM D=20/W=100000 EQUIVALENCE GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
