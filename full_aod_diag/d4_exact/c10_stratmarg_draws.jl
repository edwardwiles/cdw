# ============================================================================
# Continuation 10 (branch c10-stratified-marginal): per-country stratified-
# marginal (Latin Hypercube) draw generator for the W x D "F*" draw matrix,
# following the house style of qmc_draws.jl's pseudorandom_U/halton_U/sobol_U
# (same signature convention: (W, D; seed) -> W x D Exp(1) matrix). ADDITIVE
# ONLY -- does not modify qmc_draws.jl. Depends on `exp_from_uniform01` (relocated to
# prepare_cc/genRands.jl by the unify-random-draw-production-pipeline task, 2026-07-30) already
# being in scope -- include context_real_d20.jl BEFORE this file (it pulls in genRands.jl
# transitively via context.jl), exactly as qmc_draws.jl itself requires of its own callers.
#
# WHY (distinct question from the prior QMC/Halton/Sobol investigation):
# docs/fullA_D20_infeasibility_screening_report.md sec 5 found W=8,000
# uniformly infeasible at D=20 because 5/400 positive-share bilateral
# (origin,destination) pairs get literally ZERO winning draws on that draw
# support -- a TAIL-COVERAGE problem in each country's own marginal shock
# distribution (U is W x D, ORIGIN-indexed only, shared across destinations --
# confirmed in qmc_context_real_d20.jl's header note), not the
# smoothness/precision question the QMC investigation already asked and
# answered (null result: gradient cosine ~0.9999999 across draw types at the
# decision-relevant point). Stratified-marginal (Latin Hypercube) sampling
# forces exactly-even coverage of each country's own [0,1) marginal (hence its
# Exp(1) tail, after the inverse-CDF transform) BY CONSTRUCTION -- plain
# pseudorandom sampling gives no such guarantee, and standard Halton/Sobol
# (as tested in c10-qmc-is) equidistribute jointly across the FULL D-dim cube
# rather than specifically guaranteeing each 1-D marginal's stratification,
# so this is a genuinely different design, not a re-test of the prior one.
#
# DESIGN CHOICE -- Q (number of strata per column):
#   Default Q = W (the finest possible stratification: exactly one jittered
#   draw per stratum, the classic Latin Hypercube Sample). Chosen over a
#   coarser Q (e.g. low hundreds/thousands, with W/Q jittered draws per
#   stratum) because constructing D independent length-W permutations is
#   O(D*W) -- at most 20*80,000 = 1.6M elements, milliseconds -- so there is
#   no computational reason to coarsen, and Q=W gives the TIGHTEST possible
#   per-country marginal coverage guarantee (no stratum can ever be empty)
#   with no extra tuning parameter to justify. `Q` is left as an optional
#   keyword (Q <= W, W need not be an exact multiple of Q -- strata are
#   assigned as evenly as floor/ceil allows) purely for robustness/future use
#   at scales where D*W stops being cheap; it is NOT exercised at Q<W
#   anywhere in this task's test matrix.
#
# CRITICAL correctness point (explicit per the task spec): each of the D
# columns (one per ORIGIN country) is stratified with an INDEPENDENT random
# permutation (`Random.shuffle` on a fresh assignment vector per column). Do
# NOT reuse one permutation across columns -- that would be the "textbook
# wrong" LHS construction and would induce spurious rank correlation between
# countries' shocks that is not part of the actual model (U's columns are
# meant to be independent across origins).
# ============================================================================
using Random

"""
    stratified_marginal_U(W, D; seed, Q=W) -> Matrix{Float64}

Per-column (per-origin-country) Latin Hypercube stratification of the W x D
uniform draw matrix at `Q` equal-probability strata (default `Q=W`, the
finest/classic LHS), each column independently permuted, then passed through
the model's REAL Exp(1) inverse-CDF (`exp_from_uniform01`, defined in
qmc_context_real_d20.jl -- must already be in scope, matching qmc_draws.jl's
own convention).

Construction per column `d`:
1. Partition [0,1) into `Q` equal-width strata.
2. Distribute the `W` draw rows across the `Q` strata as evenly as possible
   (`W ÷ Q` or `W ÷ Q + 1` rows per stratum) -- `base_strata` below.
3. Independently permute (`Random.shuffle`) that row->stratum assignment for
   THIS column only (fresh randomness per column/dimension).
4. Draw one uniformly-jittered point within each row's assigned stratum.

At `Q=W` this reduces to the textbook one-point-per-stratum LHS: row `i`
(post-permutation) lands in stratum `perm[i]`, value `(perm[i]-1+u)/W` for
`u ~ Uniform(0,1)`.
"""
function stratified_marginal_U(W::Int, D::Int; seed::Int, Q::Int = W)
    @assert 1 <= Q <= W "Q must be in [1, W], got Q=$Q, W=$W"
    Random.seed!(seed)
    U01 = Matrix{Float64}(undef, W, D)
    # Balanced 0-based stratum id for each of the W rows (before per-column permutation):
    # row i (1-indexed) -> stratum floor((i-1)*Q/W), which distributes W rows across Q
    # strata as evenly as floor/ceil allows (exactly one-per-stratum when Q==W).
    base_strata = [((i - 1) * Q) ÷ W for i in 1:W]
    width = 1.0 / Q
    for d in 1:D
        assign = Random.shuffle(base_strata)   # INDEPENDENT permutation per column -- see header note
        u = rand(W)
        @inbounds for i in 1:W
            U01[i, d] = (assign[i] + u[i]) * width
        end
    end
    return exp_from_uniform01(U01)
end
