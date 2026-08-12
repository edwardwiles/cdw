# ================================================================================================
# What fraction of the inner Hessian we hand KNITRO is STRUCTURALLY zero?
#
# We declare KN_DENSE_ROWMAJOR, i.e. we tell KNITRO every entry of the upper triangle may be
# nonzero. ma97 is a SPARSE multifrontal solver; if the true pattern is far from dense, that
# declaration throws away everything it could exploit.
#
# The restriction rows are indicator moments, so some blocks are structurally zero for a reason no
# amount of data can change: a draw cannot be in bin a AND bin a' of the SAME origin, so two
# marginal rows on one origin never co-occur; likewise two joint cells of the SAME pair.
#
# Counts the pattern EMPIRICALLY (|H| > 0 over the assembled matrix) and reports it against the
# combinatorial prediction, so a mismatch in either direction is visible rather than assumed.
# ================================================================================================
_D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_cplus.jl",
          "pairwise_quantile_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, Random
lp(xs...) = (println(xs...); flush(stdout))

const D  = 20
const L  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
const nl = L - 1                       # bins retained per origin after the identification drop

nMarg = D * nl
nPair = binomial(D, 2) * nl^2
nRows = nMarg + nPair
lp("D=", D, " L=", L, "  marginal rows=", nMarg, "  pair rows=", nPair, "  total=", nRows)

# ---- combinatorial prediction of the STRUCTURAL pattern -----------------------------------------
# marg(o,a) x marg(o,a'):   zero unless a==a'        (one draw sits in ONE bin per origin)
# marg(o,a) x marg(p,a'):   dense
# marg(o,a) x pair(pq,..):  dense in general; zero only where o in {p,q} and the bin disagrees
# pair(op,ab) x pair(op,a'b'):            zero unless (a,b)==(a',b')
# pair(op,ab) x pair(oq,a'b') sharing o:  zero unless a==a'
# pair(op,ab) x pair(qr,a'b') disjoint:   dense
nP        = binomial(D, 2)
predMM    = D * nl + binomial(D, 2) * nl^2                 # within-origin diagonal + cross-origin full
predMP    = nMarg * nPair                                  # upper bound: treat marg x pair as dense
predPPsam = nP * nl^2                                      # diagonal inside each pair block
share1    = D * binomial(D - 1, 2)                         # unordered pair-of-pairs sharing one origin
predPPsh1 = share1 * nl^3                                  # a==a' pinned -> nl * nl * nl
predPPdis = (binomial(nP, 2) - share1) * nl^4
predTotal = predMM + predMP + predPPsam + predPPsh1 + predPPdis
denseTri  = nRows * (nRows + 1) / 2
lp(@sprintf("predicted structural nnz (restriction block, upper tri): %.4g of %.4g  =>  %.2f%% dense",
    float(predTotal), denseTri, 100 * predTotal / denseTri))

# ---- empirical: assemble the real thing and count -----------------------------------------------
lp("\nbuilding real D=20 context (W=100000) ...")
t0 = time()
ctx_raw = d20_real_setup_design(W = 100_000, δ = 0.1, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("context in ", round(time() - t0, digits = 1), "s")

Z = pairwise_quantile_frechet_features(ctx.U, ctx.μHat)
Q = pairwise_quantile_fixed_cutoffs(Z, L; cutoff_source = :frechet_theoretical, mu_frechet = ctx.μHat)
op = PairwiseQuantileOperator(Z, L, Q)
W  = size(Z, 1)

# We only need the PATTERN, not the values: an entry is structurally nonzero iff the two rows
# co-occur on at least one draw. So mark a Bool matrix. Writes are only ever `true`, so threading
# over draws is race-safe by construction (no read-modify-write, no lost update).
bin = op.bin
lp("\nmarking co-occurrence pattern over W=", W, " draws, ", Threads.nthreads(), " threads ...")
t1 = time()
P = zeros(Bool, nRows, nRows)

pair_id = Dict{Tuple{Int,Int},Int}()
let k = 0
    for o in 1:D-1, p in o+1:D
        k += 1; pair_id[(o, p)] = k
    end
end
# flatten to a lookup array so the hot loop never touches a Dict
pid = zeros(Int, D, D)
for ((o, p), k) in pair_id; pid[o, p] = k; end

Threads.@threads :dynamic for w in 1:W
    idx = Int[]
    sizehint!(idx, D + binomial(D, 2))
    @inbounds begin
        for o in 1:D
            a = bin[w, o]
            a <= nl && push!(idx, (o - 1) * nl + a)
        end
        for o in 1:D-1, p in o+1:D
            a = bin[w, o]; b = bin[w, p]
            (a <= nl && b <= nl) && push!(idx, nMarg + (pid[o, p] - 1) * nl^2 + (a - 1) * nl + b)
        end
        for ii in eachindex(idx), jj in eachindex(idx)
            P[idx[ii], idx[jj]] = true
        end
    end
end
lp("marked in ", round(time() - t1, digits = 1), "s")

# count inside a function -- a top-level `for` cannot assign to a global (Julia scope rule)
function count_upper(P::Matrix{Bool})
    n = size(P, 1)
    c = 0
    @inbounds for i in 1:n, j in i:n
        P[i, j] && (c += 1)
    end
    c
end
nnzTri = count_upper(P)
lp(@sprintf("\nEMPIRICAL structural nnz (restriction block, upper tri) = %d of %.0f  =>  %.2f%% dense",
    nnzTri, denseTri, 100 * nnzTri / denseTri))
lp(@sprintf("PREDICTED                                                = %.0f            =>  %.2f%% dense",
    float(predTotal), 100 * predTotal / denseTri))
lp(@sprintf("\nALWAYS-ZERO share of the matrix we declare KN_DENSE_ROWMAJOR: %.2f%%",
    100 * (1 - nnzTri / denseTri)))
lp(@sprintf("prediction error: %.3f%% of the dense triangle", 100 * abs(predTotal - nnzTri) / denseTri))
