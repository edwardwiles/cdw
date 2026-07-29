# Part C wiring gate: complete packed Hessian, wired hcz_prep_backend comparison (2026-07-29).
#
# Uses the pure-construction path (no direct archC_meanzc_base_state/verified_state call -- see
# docs/PREEXISTING_DIRECT_CALL_FAILURE_2026-07-29.md) plus synthetic random dual state to drive
# the ACTUAL production Hessian callback (hessian_cm_structured_v2!) end-to-end, comparing the
# COMPLETE packed Hessian between cctx.hcz_prep_backend = :origin_owned (current default) and
# :draw_chunk_thread_local (Part C's new candidate, now wired via hcz_prep_dispatch!).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

println("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
Random.seed!(20260719)

K_mean, K_pair, L = 1, 1, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
obj_cm = aug.obj_cm
W = size(ctx.U, 1)
NCORE = cctx.NCORE; ncm = cctx.ncm
n = NCORE + ncm
println("W=$W, D=$(ctx.D), NCORE=$NCORE, ncm=$ncm"); flush(stdout)

# Fake dual point: prep for hessian needs obj.arg0 set. Use a modest, finite, arbitrary point --
# S=ddPsi!(arg0) just needs to be a valid, finite, positive weight vector (any x gives one; Psi''
# is a convex-conjugate second derivative, never negative by construction) -- this is the
# task-sanctioned "random S" input, generated via the real ddPsi! rather than by hand.
x_fake = zeros(n)
x_fake[1] = 0.5   # zeta-like coordinate; rest at 0 (lambda-like coordinates)

for trial in 1:3
    Random.seed!(1000 + trial)
    x_fake .= 0.1 .* randn(n)
    tls = build_thread_local_scratch(cctx)

    h_origin = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cctx.hcz_prep_backend = :origin_owned
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(h_origin, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)

    h_dc = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cctx.hcz_prep_backend = :draw_chunk_thread_local
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(h_dc, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)

    Ho = unpack_packed(h_origin, n); Hd = unpack_packed(h_dc, n)
    maxdiff = maximum(abs.(Ho .- Hd))
    relscale = max(1.0, maximum(abs.(Ho)))
    ok = maxdiff < 1e-8 * relscale
    check("trial=$trial: complete packed Hessian origin_owned vs draw_chunk_thread_local (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
    check("trial=$trial: both finite", all(isfinite, h_origin) && all(isfinite, h_dc))
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
