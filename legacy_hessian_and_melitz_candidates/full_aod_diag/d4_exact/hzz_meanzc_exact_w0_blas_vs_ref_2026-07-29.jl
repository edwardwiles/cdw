# Test the EXACT w0 point the bake-off's run_cm_upper_checkpointed actually starts from
# (cmzc_w0's pivot/A-space transformed point, NOT the raw calibration point), replicating
# cb_F!'s own xf_from_w_econ transform (cm_checkpoint.jl:868) by hand. Raw numbers/exceptions only.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_aspace_coordinate.jl", "nested_quantile_grids.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random, Statistics

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

println("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
# w0 only needs theta0_up (the real gravity regression, unaffected by draw seeding) -- bare
# d20_real_setup is fine ONLY for this purpose (see memory gateC-d20-real-setup-vs-design-seeding).
ctx0_for_w0 = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0_for_w0)
theta_cm0 = cm_fixed_theta(ctx0_for_w0)
xy_cm0 = precompute_cm_aspace_xy(ctx0_for_w0)
w0 = cmzc_w0(ctx0_for_w0, pe0, theta_cm0, xy_cm0)

# THE ACTUAL SOLVE CONTEXT: must be the SEEDED d20_real_setup_design (draw_seed=20260719), exactly
# matching cm_checkpoint.jl:838's own construction -- NOT the bare d20_real_setup used above.
ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
    draw_seed = 20260719, destination_sample = :exclude_row)
ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, 100_000)
pe = build_pivot_elimination(ctx)
theta_cm = cm_fixed_theta(ctx)
xy_cm = precompute_cm_aspace_xy(ctx)
println("w0[1:5] = ", w0[1:5])
println("w0 length = ", length(w0))

K_mean, K_pair, L = 1, 1, 50
D2_econ = length(w0) - K_mean
# EXACT xf_from_w_econ replication (cm_checkpoint.jl:868-869, A_coordinate_mode=:powered_aspace, the production default)
w_econ = w0[1:D2_econ]
xf_at_w0 = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe)
nu_at_w0 = exp.(w0[D2_econ+1:end])
println("\nxf_at_w0[1:5] = ", xf_at_w0[1:5])
println("nu_at_w0 = ", nu_at_w0)

aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct,
    probs = nested_grid_sequence([10, 20, 50])[50])
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
NCORE = cctx.NCORE; ncm = cctx.ncm; n = NCORE + ncm
nx = n_restriction(cctx.hzz_zc_op)
println("NCORE=$NCORE ncm=$ncm n=$n nx (n_restriction)=$nx"); flush(stdout)

println("\n" * "="^80)
println("EXACT bake-off w0 point (via cmzc_w0 -> xf_from_w_econ transform)")
println("="^80); flush(stdout)

println("\n" * "-"^80)
println("DECISIVE TEST: :blas_gemm as the VERY FIRST backend ever touched on a FRESH cctx")
println("(never preceded by a :reference call on this same cctx -- matches the live bake-off,")
println("where ZC_GRAM_BACKEND_DEFAULT[] is set BEFORE the cctx is even built)")
println("-"^80); flush(stdout)
aug_fresh = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct,
    probs = nested_grid_sequence([10, 20, 50])[50])
cctx_fresh = build_cm_meanzc_bin_ctx(ctx, aug_fresh; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin,
    zc_gram_backend = :blas_gemm)   # set AT CONSTRUCTION, never touched by :reference first
ctx_cm_fresh = merge(ctx, (obj = aug_fresh.obj_cm,))
print("[FRESH cctx, :blas_gemm as literally the first backend] => ")
try
    bf = archC_meanzc_verified_state(xf_at_w0, nu_at_w0, ctx_cm_fresh, cctx_fresh)
    println("inner solve status = ", bf[1].inner_status)
catch e
    println("THREW: ", sprint(showerror, e))
end
flush(stdout)

println("\n" * "-"^80)
println("THE ACTUAL QUESTION: does the INNER SOLVE ITSELF converge when its OWN Newton-step")
println("Hessian (H_ZZ) uses BLAS throughout, vs :reference throughout?")
println("-"^80); flush(stdout)

for backend in (:reference, :blas_gemm, :centered_syrk, :threaded_packed)
    cctx.zc_gram_backend = backend
    print("[base_state] cctx.zc_gram_backend = $backend  =>  ")
    try
        b = archC_meanzc_base_state(xf_at_w0, nu_at_w0, ctx_cm, cctx)
        println("inner solve status = ", b.inner_status, "  ζstar=", b.ζstar)
    catch e
        println("THREW: ", sprint(showerror, e))
    end
    flush(stdout)
end

println("\n--- NOW THE ACTUAL FUNCTION cb_F! CALLS: archC_meanzc_verified_state (not base_state) ---")
for backend in (:reference, :blas_gemm)
    cctx.zc_gram_backend = backend
    print("[verified_state] cctx.zc_gram_backend = $backend  =>  ")
    try
        base_v, verify_v = archC_meanzc_verified_state(xf_at_w0, nu_at_w0, ctx_cm, cctx)
        println("inner solve status = ", base_v.inner_status, "  ζstar=", base_v.ζstar)
    catch e
        println("THREW: ", sprint(showerror, e))
    end
    flush(stdout)
end

println("\n--- now the ACTUAL cb_F! call path (verified_screened), same point, same backends ---")
pcx_mock = (ctx_cm = ctx_cm, cctx = cctx, screen_counters = nothing)
for backend in (:reference, :blas_gemm)
    cctx.zc_gram_backend = backend
    print("cctx.zc_gram_backend = $backend  =>  ")
    try
        _, b, v = cm_meanzc_production_value_verified_screened(xf_at_w0, nu_at_w0, pcx_mock; counters = nothing)
        println("verified_screened status = ", b.inner_status)
    catch e
        println("THREW: ", sprint(showerror, e))
    end
    flush(stdout)
end

cctx.zc_gram_backend = :reference
local base
try
    global base = archC_meanzc_base_state(xf_at_w0, nu_at_w0, ctx_cm, cctx)
    println("\ninner solve (base state, :reference) status = ", base.inner_status)
catch e
    println("INNER SOLVE (base state) THREW:")
    println(sprint(showerror, e))
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 20 && break
        println("  [$i] ", frame)
    end
    exit(1)
end
x0v = vcat(base.ζstar, base.λstar)
println("x0v[1:5] = ", x0v[1:5])

local h_ref, HZZ_ref
cctx.zc_gram_backend = :reference
println("\n--- H_ZZ WITHOUT BLAS (:reference) ---")
try
    global h_ref
    _archC_prep_for_hessian!(ctx_cm.obj, x0v)
    h_ref = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured!(h_ref, ctx_cm.obj, cctx)
    Hfull_ref = unpack_packed(h_ref, n)
    global HZZ_ref = Hfull_ref[cctx.ncore_core+1:NCORE, cctx.ncore_core+1:NCORE]
    println("SUCCESS. H_ZZ is $(size(HZZ_ref)). All finite: ", all(isfinite, HZZ_ref))
    println("HZZ_ref diagonal[1:5] = ", diag(HZZ_ref)[1:min(5,nx)])
catch e
    println("THREW: ", sprint(showerror, e))
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 20 && break
        println("  [$i] ", frame)
    end
    global HZZ_ref = nothing
end

local h_blas, HZZ_blas
cctx.zc_gram_backend = :blas_gemm
println("\n--- H_ZZ WITH BLAS (:blas_gemm) ---")
try
    global h_blas
    _archC_prep_for_hessian!(ctx_cm.obj, x0v)
    h_blas = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    hessian_cm_structured!(h_blas, ctx_cm.obj, cctx)
    Hfull_blas = unpack_packed(h_blas, n)
    global HZZ_blas = Hfull_blas[cctx.ncore_core+1:NCORE, cctx.ncore_core+1:NCORE]
    println("SUCCESS. H_ZZ is $(size(HZZ_blas)). All finite: ", all(isfinite, HZZ_blas))
    println("HZZ_blas diagonal[1:5] = ", diag(HZZ_blas)[1:min(5,nx)])
catch e
    println("THREW: ", sprint(showerror, e))
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        i > 20 && break
        println("  [$i] ", frame)
    end
    global HZZ_blas = nothing
end
cctx.zc_gram_backend = :reference

if HZZ_ref !== nothing && HZZ_blas !== nothing
    maxdiff = maximum(abs.(HZZ_ref .- HZZ_blas))
    relscale = max(1.0, maximum(abs.(HZZ_ref)))
    println("\nCOMPARISON: max|HZZ_ref - HZZ_blas| = $maxdiff   (scale=$relscale, relative=$(maxdiff/relscale))")
else
    println("\nCOMPARISON: cannot compare -- at least one backend did not produce a value.")
end
println("\nDONE.")
