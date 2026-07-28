# ================================================================================================
# True no-H operator bundle (2026-07-28 continuation), Part A.3: flexible-CM equivalence gate.
#
# Compares, at IDENTICAL draws/outer-parameters/KNITRO options/tolerances/threads, side-by-side
# `build_cm_production_context(...; moment_representation=:dense_reference)` (constructs the
# unchanged `CS.PsiObjectiveBundleImplicit`) against `moment_representation=:operator` (constructs
# the new no-H `OperatorPsiBundle`, primed via `prime_operator!`).
#
# Fixed-point callback comparison (the machine-precision proof, per task instructions): at x=0,
# four deterministic random dual vectors, and one real solved dual vector, compare objective,
# gradient, Psi''(r) (implicitly, via the Hessian), and the packed Hessian.
#
# Then: full inner solves from the same start, comparing accepted status, delta*, dual vector, and
# structural proof that OperatorPsiBundle genuinely has no forbidden fields.
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t 4 \
#          full_aod_diag/d4_exact/test_operator_no_H_bundle_equivalence_flexcm.jl
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl"]
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

println("="^90)
println("PART A.3: flexible-CM operator-bundle (no H) vs dense-reference bundle -- equivalence gate")
println("="^90)

pcx_d = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = false, moment_representation = :dense_reference)
pcx_o = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = false, moment_representation = :operator)
pcx_d.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_o.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_d.cctx.cm_cross_hessian_backend = :winner_bin
pcx_o.cctx.cm_cross_hessian_backend = :winner_bin

# --- Structural proof (task §3): OperatorPsiBundle has none of the forbidden fields ---
obj_o = pcx_o.ctx_cm.obj
obj_d = pcx_d.ctx_cm.obj
fn_o = fieldnames(typeof(obj_o))
lp("OperatorPsiBundle fieldnames: ", fn_o)
forbidden = (:H, :H_copy, :G, :K, :ones, :jac_h, Symbol("moments!"))
for f in forbidden
    check("OperatorPsiBundle has no field :$f", !(f in fn_o))
end
check("dense-reference bundle IS PsiObjectiveBundleImplicit (unchanged type)", obj_d isa CS.PsiObjectiveBundleImplicit)
check("operator bundle IS OperatorPsiBundle (unchanged type)", obj_o isa OperatorPsiBundle)

# Attempts to reach for the forbidden interface must fail immediately, not silently no-op.
let threw = false
    try
        obj_o.H
    catch e
        threw = true
    end
    check("obj_o.H throws", threw)
end
let threw = false
    try
        select_G_from_H(obj_o, nothing)
    catch e
        threw = true
    end
    check("select_G_from_H(obj_o, ...) throws", threw)
end

# --- Fixed-point callback comparison (the machine-precision proof) ---
base_d0 = archC_base_state(x_free_calib, pcx_d.ctx_cm, pcx_d.cctx)
base_o0 = archC_base_state(x_free_calib, pcx_o.ctx_cm, pcx_o.cctx)
check("both backends: calibration inner solve feasible",
      base_d0.inner_status in (0,-100,-101,-103) && base_o0.inner_status in (0,-100,-101,-103))

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
    # Objective/gradient: run the SAME FG functor path (CMLookupState) both bundles actually use in
    # production -- construct it fresh here rather than reuse cctx.cmlookup_st, so this comparison
    # is independent of inner-solve call ordering.
    bins_u = pcx_d.cctx.Bidx isa Matrix{UInt32} ? pcx_d.cctx.Bidx : Matrix{UInt32}(pcx_d.cctx.Bidx)
    st_d = CMLookupState(obj_d, pcx_d.cctx.NCORE, pcx_d.cctx.ncm, pcx_d.cctx.L, pcx_d.cctx.origins, pcx_d.cctx.refIndex1,
                          bins_u, pcx_d.cctx.R; method = :suffix, core_cf_ref = pcx_d.cctx.core_cf_ref)
    st_o = CMLookupState(obj_o, pcx_o.cctx.NCORE, pcx_o.cctx.ncm, pcx_o.cctx.L, pcx_o.cctx.origins, pcx_o.cctx.refIndex1,
                          bins_u, pcx_o.cctx.R; method = :suffix, core_cf_ref = pcx_o.cctx.core_cf_ref)
    # Prime both (fills K/payoff, gravity column, and publishes core_cf_ref) at the SAME θ (calibration).
    θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
    obj_d.moments!(@view(obj_d.H[:, 1]), CS.select_G_from_H(obj_d, obj_d.H), θ_full_calib, obj_d.U, obj_d)
    obj_d.H[:, 2] .= 1.0
    prime_operator!(obj_o, θ_full_calib, ctx, pcx_o.cctx.core_cf_ref; restriction_state = pcx_o.cctx)

    g_d = zeros(n); g_o = zeros(n)
    f_d = st_d(x, g_d)
    f_o = st_o(x, g_o)
    e_f = abs(f_d - f_o)
    e_g = maximum(abs.(g_d .- g_o))

    Hd = full_hessian_generic(pcx_d.ctx_cm, pcx_d.cctx, x)
    Ho = full_hessian_generic(pcx_o.ctx_cm, pcx_o.cctx, x)
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

# --- Full inner-solve comparison ---
lp("="^90); lp("Full inner solves from x_free_calib")
base_d = archC_base_state(x_free_calib, pcx_d.ctx_cm, pcx_d.cctx)
base_o = archC_base_state(x_free_calib, pcx_o.ctx_cm, pcx_o.cctx)
check("full solve: both backends same accepted status", base_d.inner_status == base_o.inner_status)
e_zeta = abs(base_d.ζstar - base_o.ζstar)
e_lambda = maximum(abs.(base_d.λstar .- base_o.λstar))
lp("  Δζ*=$e_zeta   max|Δλ*|=$e_lambda   status_d=$(base_d.inner_status) status_o=$(base_o.inner_status)")
check("full solve: delta* (ζ*) agrees to 1e-10", e_zeta < 1e-10)
check("full solve: dual vector (λ*) agrees to 1e-8", e_lambda < 1e-8)

# --- THREADED-BINS re-check (production default is threaded_bins=true; the above used false).
# Added after finding build_bin_tables_threaded! had a dense-only method signature that would have
# broken exactly this path -- re-verify the actual production default config, not just the serial one.
lp("="^90); lp("Threaded-bins (production default) re-check")
pcx_d_t = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true,
                                       threaded_bins = true, moment_representation = :dense_reference)
pcx_o_t = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true,
                                       threaded_bins = true, moment_representation = :operator)
pcx_d_t.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_o_t.cctx.core_hessian_backend = :exact_winner_pair_parallel
base_dt = archC_base_state(x_free_calib, pcx_d_t.ctx_cm, pcx_d_t.cctx)
base_ot = archC_base_state(x_free_calib, pcx_o_t.ctx_cm, pcx_o_t.cctx)
check("threaded: both backends feasible", base_dt.inner_status in (0,-100,-101,-103) && base_ot.inner_status in (0,-100,-101,-103))
e_zeta_t = abs(base_dt.ζstar - base_ot.ζstar)
e_lambda_t = maximum(abs.(base_dt.λstar .- base_ot.λstar))
lp("  Δζ*=$e_zeta_t   max|Δλ*|=$e_lambda_t   status_d=$(base_dt.inner_status) status_o=$(base_ot.inner_status)")
check("threaded: delta* (ζ*) agrees to 1e-10", e_zeta_t < 1e-10)
check("threaded: dual vector (λ*) agrees to 1e-8", e_lambda_t < 1e-8)

println("="^90)
if isempty(FAILURES)
    println("ALL OPERATOR-VS-DENSE FLEXIBLE-CM EQUIVALENCE GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
