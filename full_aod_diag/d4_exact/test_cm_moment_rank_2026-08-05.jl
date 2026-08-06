# 2026-08-05: user-requested rank check of the DENSE two-family CM moment matrix at the real D20
# calibration point -- what's running in the D20 fixed-state/trace tests is PLAIN FLEXIBLE CM
# (build_cm_production_context has no cm_extension kwarg at all; :cm_lookup/:operator is documented
# "plain flexible CM only") -- NOT CM+ZC, NOT common Frechet, NOT origin-ZC. This checks whether the
# stalled-optimality-error behavior seen in the outlev=3 trace (test_cm_archc_d20_trace_2026-08-05.jl)
# is explained by a rank-deficient/near-collinear moment matrix -- exactly the mechanism already
# root-caused once in this repo for CM+ZC K=2(sigma=3) (see memory
# cmzc-k2-singularity-root-cause-and-fix-2026-08-05.md: "exact collinearity autarky-moment vs
# k=(sigma-1) restriction"), which is suspicious because eq.36 here is ALSO a k=(sigma-1)-weighted
# restriction.
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
using Printf, LinearAlgebra, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat)

const L = 10
probs_ = collect(range(1 / L, (L - 1) / L, length = L))

aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored, probs = probs_)
@assert aug.n_families == 2
objA = aug.obj_cm
n = objA.outer_constr_index
M = size(ctx.U, 1)
K = zeros(M)
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full_calib, objA.U, objA)
objA.H[:, 1] .= K
objA.H[:, 2] .= 1.0

G_full = Matrix(objA.H[:, 3:n+1])          # core + CM columns, exactly what the inner solve contracts lambda against
ncm_cdf = aug.ncm_cdf
CM = aug.CM                                 # CM-only block (post-fix): [:, 1:ncm_cdf]=CDF family, [:, ncm_cdf+1:end]=POW family
ncore = size(G_full, 2) - size(CM, 2)
lp("G_full size: ", size(G_full), "  (ncore=", ncore, ", ncm_cdf=", ncm_cdf, ", ncm_pow=", size(CM,2)-ncm_cdf, ")")

function rank_report(name::String, Mx::AbstractMatrix; tol_rel = 1e-10)
    sv = svdvals(Mx)
    smax = sv[1]
    smin = sv[end]
    r_eff = count(s -> s > tol_rel * smax, sv)
    lp("  [$name] size=", size(Mx), " sigma_max=", smax, " sigma_min=", smin,
       " cond=", smax/smin, " rank(tol=$(tol_rel)*sigma_max)=", r_eff, "/", size(Mx,2))
    # print the smallest few singular values for context
    k = min(8, length(sv))
    lp("    smallest ", k, " singular values: ", sv[end-k+1:end])
    return sv
end

lp("="^100)
lp("Section 1: CM-only block (aug.CM), CDF half alone")
lp("="^100)
rank_report("CDF-only (eq.35)", CM[:, 1:ncm_cdf])

lp("="^100)
lp("Section 2: CM-only block, POW half alone")
lp("="^100)
rank_report("POW-only (eq.36)", CM[:, ncm_cdf+1:end])

lp("="^100)
lp("Section 3: CM-only block, BOTH families together (this is the object whose rank matters for the")
lp("two-family inner solve's lambda_cm block)")
lp("="^100)
sv_both = rank_report("CDF+POW together", CM)

lp("="^100)
lp("Section 4: full G (core + both CM families) -- what the inner KKT system actually factors")
lp("="^100)
sv_full = rank_report("core+CDF+POW", G_full)

lp("="^100)
lp("Section 5: pairwise correlation between each (origin,l) CDF column and its POW counterpart --")
lp("localizes a collinearity to specific (origin,l) pairs rather than 'somewhere in the 380 columns'")
lp("="^100)
nO = length(aug.origins)
worst_corr = 0.0
worst_pair = (0, 0)
for l in 1:L, oi in 1:nO
    c_cdf = @view CM[:, (l-1)*nO + oi]
    c_pow = @view CM[:, ncm_cdf + (l-1)*nO + oi]
    ρ = abs(cor(c_cdf, c_pow))
    if ρ > worst_corr
        global worst_corr = ρ
        global worst_pair = (l, oi)
    end
end
lp("  max |corr(CDF_col, matching POW_col)| over all (l,origin) pairs = ", worst_corr, " at (l,oi)=", worst_pair)

lp("="^80)
lp("DONE")
