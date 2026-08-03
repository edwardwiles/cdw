# Part B: isolated H_ZZ backend correctness/shadow-mode tests (2026-07-29).
#
# Uses ONLY the pure-construction path (build_cm_meanzc_bin_ctx/build_cm_meanzc_augmented_obj --
# NO KNITRO solve, confirmed safe -- see docs/PREEXISTING_DIRECT_CALL_FAILURE_2026-07-29.md for why
# a direct archC_meanzc_base_state/verified_state call is NOT used here) plus synthetic random S
# (task-sanctioned: "random and solver-derived S" -- S>=0 is the only structural requirement, Psi''
# is a convex-conjugate second derivative, never negative).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random, Statistics

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end

println("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
Random.seed!(20260719)   # seeds the SYNTHETIC S draws below, not ctx's own draws (already built)

K_mean, K_pair, L = 1, 1, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
op = cctx.hzz_zc_op
W = size(ctx.U, 1)
nx = n_restriction(op)
println("W=$W, D=$(ctx.D), nx (n_restriction)=$nx"); flush(stdout)

ν0 = [Float64(factorial(k)) for k in 1:K_mean]
refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, ν0)
cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op, W)

const BACKENDS = [:reference, :centered_syrk, :blas_syrk, :blas_gemm, :threaded_packed]
M = 1.0

function compute_hzz(backend::Symbol, cctx, op, S, M, W)
    refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, S; fill_S = backend === :reference)
    HZZ = zeros(nx, nx)
    if backend === :reference
        zc_restriction_gram!(HZZ, cctx.hzz_centered, op, M)
    else
        cctx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(cctx.raw_zc_ws, op, W)
        refresh_zc_raw_target_vector!(cctx.raw_zc_ws, cctx.hzz_zc_ws, op)
        zc_gram_dispatch!(HZZ, backend, cctx.hzz_centered, op, cctx.raw_zc_ws, S, M; workers = 20)
    end
    return HZZ
end

# ---- Every-Hessian-point comparison across 3 independent random-S draws ----
println("\n=== Every-Hessian-point comparison (3 random-S draws) ==="); flush(stdout)
for trial in 1:3
    S = rand(W) .* 2.0 .+ 0.01
    HZZ_ref = compute_hzz(:reference, cctx, op, S, M, W)
    check("trial=$trial reference: finite", all(isfinite, HZZ_ref))
    check("trial=$trial reference: symmetric", maximum(abs.(HZZ_ref .- HZZ_ref')) < 1e-12)
    for backend in BACKENDS[2:end]
        H = compute_hzz(backend, cctx, op, S, M, W)
        finite_ok = all(isfinite, H)
        maxdiff = finite_ok ? maximum(abs.(H .- HZZ_ref)) : NaN
        relscale = max(1.0, maximum(abs.(HZZ_ref)))
        ok = finite_ok && maxdiff < 1e-8 * relscale
        check("trial=$trial $backend: finite", finite_ok)
        check("trial=$trial $backend: matches reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
        # determinism: repeated call, identical input
        H2 = compute_hzz(backend, cctx, op, S, M, W)
        check("trial=$trial $backend: deterministic repeat call (max|Δ|=$(maximum(abs.(H .- H2))))", H == H2)
        # min eigenvalue / PSD sanity (H_ZZ should be PSD -- it's (1/M) Z'SZ, S>=0)
        eigmin = try
            minimum(eigvals(Symmetric(H)))
        catch e
            NaN
        end
        check("trial=$trial $backend: min eigenvalue >= -1e-6*scale (eigmin=$(eigmin))", isfinite(eigmin) && eigmin > -1e-6 * relscale)
        # random quadratic form cross-check: v'Hv should match v'Href*v for random v
        v = randn(nx)
        qf_ref = dot(v, HZZ_ref * v)
        qf_cand = dot(v, H * v)
        check("trial=$trial $backend: random quadratic form matches (Δ=$(abs(qf_ref-qf_cand)))", abs(qf_ref - qf_cand) < 1e-6 * max(1.0, abs(qf_ref)))
    end
end

# ---- Shadow-mode tests A-D, at ONE frozen random-S state, for each non-reference backend ----
println("\n=== Shadow-mode tests A-D ==="); flush(stdout)
Random.seed!(999)
S_shadow = rand(W) .* 2.0 .+ 0.01
for backend in BACKENDS[2:end]
    # A: compute candidate, discard it, pass reference
    refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, S_shadow; fill_S = true)
    Zc_before = copy(cctx.hzz_centered.Zc)
    Hdiscard = compute_hzz(backend, cctx, op, S_shadow, M, W)
    Href_afterA = compute_hzz(:reference, cctx, op, S_shadow, M, W)
    Zc_afterA = copy(cctx.hzz_centered.Zc)
    check("$backend shadow-A: Zc unchanged after discarding candidate compute (no side effect)", Zc_before == Zc_afterA)

    # B: compute reference, discard it, pass candidate
    Href_discard = compute_hzz(:reference, cctx, op, S_shadow, M, W)
    Hcand_afterB = compute_hzz(backend, cctx, op, S_shadow, M, W)
    check("$backend shadow-B: candidate matches its own shadow-A value (max|Δ|=$(maximum(abs.(Hcand_afterB .- Hdiscard))))",
          maximum(abs.(Hcand_afterB .- Hdiscard)) < 1e-8 * max(1.0, maximum(abs.(Hdiscard))))

    # C/D: compute both in separate buffers, compare
    Href_CD = compute_hzz(:reference, cctx, op, S_shadow, M, W)
    Hcand_CD = compute_hzz(backend, cctx, op, S_shadow, M, W)
    maxdiff_CD = maximum(abs.(Href_CD .- Hcand_CD))
    check("$backend shadow-C/D: both buffers agree (max|Δ|=$(maxdiff_CD))", maxdiff_CD < 1e-8 * max(1.0, maximum(abs.(Href_CD))))
end

# ---- BLAS alias/output checks ----
println("\n=== BLAS alias/output checks ==="); flush(stdout)
refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, S_shadow; fill_S = true)
cctx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(cctx.raw_zc_ws, op, W)
refresh_zc_raw_target_vector!(cctx.raw_zc_ws, cctx.hzz_zc_ws, op)
for backend in (:centered_syrk, :blas_syrk, :blas_gemm, :threaded_packed)
    HZZ_poison = fill(NaN, nx, nx)
    zc_gram_dispatch!(HZZ_poison, backend, cctx.hzz_centered, op, cctx.raw_zc_ws, S_shadow, M; workers = 20)
    check("$backend: fully overwrites poisoned output (no NaN survives)", all(isfinite, HZZ_poison))
    aliases_Zc = Base.mightalias(cctx.raw_zc_ws.RW, cctx.hzz_centered.Zc)
    check("$backend: raw_ws.RW does not alias hzz_centered.Zc", !aliases_Zc)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
