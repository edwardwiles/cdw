# 2026-08-05: user-requested isolation experiment -- drop the eq.35 CDF family entirely and run
# the restriction with ONLY the eq.36 truncated-power (z^(sigma-1)) family, to see whether the D20
# real-data non-convergence (nStatus=-400 at every tested W and L down to 3, see
# test_cm_moment_rank_by_L_2026-08-05.jl / MASTER.md sections 21-24) is about the INTERACTION
# between the two families, or whether the POW family alone is already the problem.
#
# Uses the dense Architecture A path deliberately (NOT Architecture C) -- this is a one-off
# diagnostic about whether the underlying restriction/moment-matrix is solvable at all, which is
# architecture-independent (already confirmed via ForwardDiff that Architecture C computes exactly
# the same thing as the dense objective given the same G matrix, test_cm_autodiff_groundtruth_
# 2026-08-05.jl) -- reusing the validated single-family (CDF-only) dense bundle as a template and
# swapping its CM columns for the (already bug-fixed, section 17) POW feature is the fastest way to
# get a dimensionally-guaranteed-correct POW-only bundle with zero new Hessian/gradient code.
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
M = size(ctx.U, 1)
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat)

function run_variant(name::String, L::Int; use_pow::Bool)
    probs_ = collect(range(1 / L, (L - 1) / L, length = L))
    aug_cdf = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = false, contrasts = :anchored, probs = probs_)
    objA = aug_cdf.obj_cm
    n = objA.outer_constr_index
    K = zeros(M)
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full_calib, objA.U, objA)
    objA.H[:, 1] .= K
    objA.H[:, 2] .= 1.0
    ncm = aug_cdf.ncm

    if use_pow
        aug_two = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored, probs = probs_)
        @assert aug_two.ncm_cdf == ncm
        pow_cols = aug_two.CM[:, aug_two.ncm_cdf+1:end]
        @assert size(pow_cols, 2) == ncm
        cm_start = n + 2 - ncm
        cm_end = n + 1
        objA.H[:, cm_start:cm_end] .= pow_cols
    end

    G = Matrix(objA.H[:, 3:n+1])
    sv = svdvals(G)
    cond_G = sv[1] / sv[end]

    t0 = time()
    K_hard, inner_x, nStatus = CS.inner_loop_internal(objA, θ_full_calib)
    dt = time() - t0
    feasible = nStatus in (0, -100, -101, -102, -103)
    lp(@sprintf("  [%s] L=%2d cond(G)=%12.2f  nStatus=%4d (%s)  %.1fs", name, L, cond_G, nStatus,
                feasible ? "PASS" : "FAIL", dt))
end

lp("="^100)
lp("Baseline sanity: single-family CDF-only (known-good reference, should converge cleanly)")
lp("="^100)
for L in (3, 10)
    run_variant("CDF-only", L; use_pow = false)
end

lp("="^100)
lp("Experiment: single-family POW-only (z^(sigma-1) truncated moment, NO CDF family at all)")
lp("="^100)
for L in (3, 10)
    run_variant("POW-only", L; use_pow = true)
end

lp("="^80)
lp("DONE")
