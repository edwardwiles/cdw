# Isolated diagnostic: compare the deduped T3/T4 build+read against a brute-force direct
# recomputation (mimicking the OLD non-deduped scatter, but computed on synthetic data so it needs
# no real KNITRO context), to localize the dedup bug fast.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl"]
    include(joinpath(D4X, f))
end
using Random, LinearAlgebra

Random.seed!(42)
D = 4
W = 500
L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5   # test-file convenience only
nc = L - 1
# Synthetic draws + synthetic cutoffs: under version B the bin assignment is a campaign constant
# owned by the operator, so instead of overwriting a mutable bin array after the fact (what version
# A's version of this script did) we choose cutoffs that reproduce the desired bin pattern -- here
# just the quantiles of uniform draws, which is enough to exercise every T3/T4 combo.
U = rand(W, D)
MU_FRECHET = 1.0 / 6.0   # synthetic; the T3/T4 dedup is bin-pattern-only and mu-independent
Zs = frechet_power_feature(U, 1, MU_FRECHET)
Q = pairwise_quantile_fixed_cutoffs(Zs, L; cutoff_source = :empirical_quantile, mu_frechet = MU_FRECHET)
op = PairwiseQuantileOperator(Zs, L, Q)
println("L=", L, "  npair=", op.npair, "  triple_combos=", length(op.triple_combos), "  quad_combos=", length(op.quad_combos))

h = rand(W) .+ 0.1

tls = build_pairwise_quantile_thread_scratch(D, op.npair, L)
tabs = PairwiseQuantileHessianTables(op)
build_pairwise_quantile_hessian_tables!(tabs, op, h, tls)

# ---- brute force T3: direct scatter over the FULL (redundant) triple_combos list, no dedup ----
bin = op.bin
nlast = UInt8(nc)
T3_brute = zeros(nc, nc, nc, length(op.triple_combos))
triple_opq = [(o, op.pairs[pidx][1], op.pairs[pidx][2]) for (o, pidx) in op.triple_combos]
for w in 1:W, k in 1:length(triple_opq)
    (o, p, q) = triple_opq[k]
    a = bin[w, o]; a > nlast && continue
    b = bin[w, p]; b > nlast && continue
    c = bin[w, q]; c > nlast && continue
    T3_brute[a, b, c, k] += h[w]
end

maxerr3 = 0.0
for tidx in 1:length(op.triple_combos), a in 1:nc, b in 1:nc, c in 1:nc
    v_dedup = read_T3(tabs, tidx, a, b, c)
    v_brute = T3_brute[a, b, c, tidx]
    global maxerr3 = max(maxerr3, abs(v_dedup - v_brute))
end
println("T3 dedup vs brute-force: max abs error = ", maxerr3, "  ", maxerr3 < 1e-9 ? "PASS" : "FAIL")

# ---- brute force T4 ----
T4_brute = zeros(nc, nc, nc, nc, length(op.quad_combos))
quad_oooo = [(op.pairs[pidx1][1], op.pairs[pidx1][2], op.pairs[pidx2][1], op.pairs[pidx2][2])
             for (pidx1, pidx2) in op.quad_combos]
for w in 1:W, k in 1:length(quad_oooo)
    (o1, o2, o3, o4) = quad_oooo[k]
    a = bin[w, o1]; a > nlast && continue
    b = bin[w, o2]; b > nlast && continue
    c = bin[w, o3]; c > nlast && continue
    d = bin[w, o4]; d > nlast && continue
    T4_brute[a, b, c, d, k] += h[w]
end

maxerr4 = 0.0
for combo in 1:length(op.quad_combos), a in 1:nc, b in 1:nc, c in 1:nc, d in 1:nc
    v_dedup = read_T4(tabs, combo, a, b, c, d)
    v_brute = T4_brute[a, b, c, d, combo]
    global maxerr4 = max(maxerr4, abs(v_dedup - v_brute))
end
println("T4 dedup vs brute-force: max abs error = ", maxerr4, "  ", maxerr4 < 1e-9 ? "PASS" : "FAIL")
