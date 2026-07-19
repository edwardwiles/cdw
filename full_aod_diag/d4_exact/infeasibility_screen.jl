# ============================================================================
# Continuation 9 (branch c9-infeasibility-screen): exact early infeasibility
# screening for hard-winner outer points, applied BEFORE constructing
# moments or invoking the inner CC dual solver (KNITRO).
#
# WHY: for every bilateral pair (o,d) with positive target trade share
# Pmat[o,d] > 0, if origin o never wins destination d on the FIXED simulation
# support (the W Frechet draws baked into ctx.U at context-construction
# time), no reweighting of those draws can ever satisfy the bilateral factual
# moment -- the primal moment problem is infeasible and the minimum
# divergence is exactly +infinity. This is a STRUCTURAL property of
# (theta_full, ctx.U), independent of delta, independent of the inner CC
# dual solve, and (for the pairwise certificate, Section 2 below) computable
# WITHOUT ever touching a single draw at the current outer point. Detecting
# this before building moments/calling KNITRO turns a ~1-5s wasted inner
# solve into microseconds-to-milliseconds.
#
# ADDITIVE ONLY. Does not modify winners.jl, winners_v2.jl,
# compressed_moments.jl, compressed_cc_inner.jl, compressed_live.jl,
# lfix_incremental.jl, oracle.jl, or oracle_fast.jl. Every function here is
# NEW; the only "integration" point is `evaluate_fullA_screened` /
# `evaluate_fullA_screened_compressed`, which are new opt-in entry points
# that FALL THROUGH to the EXISTING `evaluate_fullA_fast` (dense) once the
# screen passes -- callers who never call these new functions see zero
# behavior change.
#
# ----------------------------------------------------------------------------
# MAPPING TABLE: spec math notation -> this codebase's real variable/function
# names (empirically confirmed, not assumed -- see
# docs/fullA_D20_infeasibility_screening_report.md sec 1 for the validation
# script and its 0/32000-mismatch result at two independent D=4 points, one
# calibration and one random A-perturbation):
#
#   hard score S_sod          ==  B_so + a_od  (defined below; NOT a
#                                  pre-existing named quantity anywhere in
#                                  this codebase -- derived here from
#                                  winners.jl::factual_prices' price formula)
#   winner_{s,d} = argmin_o price[s,o,d]  ==  argmax_o S_sod  (S := -log(price)/mu,
#                                  a strictly increasing transform since mu>0,
#                                  so argmin price == argmax S exactly)
#   B_so                       ==  -log(ctx.U[s,o])   (W x D; PURE DATA, no
#                                  theta-dependence at all -- "fixed across
#                                  optimization" in the literal, not just
#                                  loose, sense)
#   a_od                       ==  log(Aod_theta[o,d]) + log(lambda[o,d])
#                                  - log(lambda[1,d]) - (1/mu)*log(wHat[1,1]*tau[1,d])
#                                  where Aod_theta = reshape(theta_full[ctx.Aod_offset+1
#                                  : ctx.Aod_offset+D^2], D, D) is EXACTLY the
#                                  free A_od outer-loop parameter this whole
#                                  investigation optimizes over (log(A_od) is
#                                  the "almost certainly" guess in the task
#                                  brief, confirmed correct up to fixed,
#                                  theta-independent data terms that must be
#                                  folded in somewhere for the additive
#                                  B_so+a_od split to hold -- mu, lambda, wHat,
#                                  tau are all FIXED at every outer point in
#                                  this investigation's outer loop, which
#                                  varies only (gamma'_focal, A_od); a_od is
#                                  recomputed fresh at every outer point same
#                                  as A_od itself is, consistent with the
#                                  spec's framing)
#   B_so fixed across optim.   ==  literally true: B depends only on ctx.U
#   target trade share         ==  Pmat[o,d] = ctx.gamma.P[d+(o-1)*D]  (the
#                                  observed bilateral expenditure-share matrix
#                                  lambda -- for D=20 real data this is the
#                                  pi.csv-derived lambda/pi matrix loaded via
#                                  d20_real_setup's fakeData==3 path; for D=4
#                                  synthetic it is createFakeData's synthetic
#                                  lambda. SAME field either way -- no
#                                  D=20-specific code needed.) Confirmed
#                                  identical to the `lambda` matrix used
#                                  inside compressed_moments.jl's own
#                                  build_compressed_factual (Pmat ==
#                                  reshape(P,(D,D))' bit-for-bit).
#   M_ok = max_s(B_so - B_sk)  ==  precompute_pairwise_M(ctx) -> PairwiseCertificate.M
#   a_kd - a_od > M_ok         ==  pairwise_certificate's per-(o,d) rejection test
#   m_od slack                 ==  pairwise_certificate's PairwiseScreenResult.m_od
#   hard winner construction   ==  screen_hard_winners (destination-major,
#                                  literal copy of compressed_moments.jl::
#                                  build_compressed_factual's winner-finding
#                                  inner loop, attributed reuse, NOT
#                                  reimplemented from scratch), with
#                                  immediate per-destination zero-win-count
#                                  rejection added (the new part) and a
#                                  TIE-SAFE win-count pass (see note in
#                                  screen_hard_winners' docstring: crediting
#                                  only the first-index winner would
#                                  UNDER-count wins at an exact price tie,
#                                  since MinInd!/hFunction! splits mass
#                                  across ALL tied-minimum origins, not just
#                                  the first -- this is the one place this
#                                  screen deliberately diverges from
#                                  compute_winners'/compressed's single-index
#                                  `winner[s,d]` convention, specifically to
#                                  stay an EXACT rejection test, never a false
#                                  positive, at the (probability-zero in real
#                                  data, but deliberately exercised by this
#                                  report's adversarial test points)
#                                  exact-tie boundary)
# ----------------------------------------------------------------------------
# ============================================================================

using SpecialFunctions: gamma as spgamma

# ============================================================================
# SECTION 2 (implemented first -- Section 1's winner construction reuses its
# output for destination ordering): draw-free pairwise impossibility
# certificate.
# ============================================================================

"B[s,o] = -log(U[s,o]) -- the fixed (draw-only) part of the hard score S_sod = B_so + a_od."
hard_score_B(ctx) = -log.(ctx.U)   # W x D

struct PairwiseCertificate
    D::Int
    M::Matrix{Float64}   # D x D; M[o,k] = max_s(B[s,o]-B[s,k]) for o!=k, -Inf on the diagonal (unused)
end

"""
    precompute_pairwise_M(ctx) -> PairwiseCertificate

Precomputes `M_ok = max_s (B_so - B_sk)` for every ordered pair `o != k`,
O(D^2 * W) once (draw-free at every SUBSEQUENT outer point -- B depends only
on ctx.U, never on theta). Reuse this across many outer-point evaluations by
building it ONCE per ctx and passing it into `pairwise_certificate` /
`evaluate_fullA_screened` via the `pairwise=` kwarg.
"""
function precompute_pairwise_M(ctx)
    B = hard_score_B(ctx)
    D = ctx.D; W = size(B, 1)
    M = fill(-Inf, D, D)
    @inbounds for k in 1:D
        for o in 1:D
            o == k && continue
            best = -Inf
            for s in 1:W
                v = B[s, o] - B[s, k]
                v > best && (best = v)
            end
            M[o, k] = best
        end
    end
    return PairwiseCertificate(D, M)
end

"""
    compute_a_od(theta_full, ctx) -> Matrix{Float64} (D x D)

a_od as defined in the mapping table above. Recomputed at every outer point
(A_od is the free parameter), O(D^2).
"""
function compute_a_od(θ_full::AbstractVector, ctx)
    D = ctx.D
    γo = ctx.γ
    μ = θ_full[1]
    lambda = reshape(γo.P, (D, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    a = Matrix{Float64}(undef, D, D)
    invmu = 1.0 / μ
    @inbounds for d in 1:D
        cst_d = invmu * log(γo.wHat[1, 1] * γo.τ[1, d])
        l1d = log(lambda[1, d])
        for o in 1:D
            a[o, d] = log(Aod_θ[o, d]) + log(lambda[o, d]) - l1d - cst_d
        end
    end
    return a
end

"target_shares(ctx) -> Pmat (D x D); Pmat[o,d] = observed bilateral share lambda[o,d] (the target trade share)."
function target_shares(ctx)
    D = ctx.D
    γo = ctx.γ
    return [γo.P[d + (o - 1) * D] for o in 1:D, d in 1:D]
end

struct PairwiseScreenResult
    infeasible::Bool
    m_od::Matrix{Float64}   # D x D; m_od[o,d] = min_{k!=o}[M_ok-(a_kd-a_od)] for Pmat[o,d]>0, else +Inf
    worst_o::Int
    worst_d::Int
    worst_k::Int
    worst_slack::Float64
end

"""
    pairwise_certificate(a, pc::PairwiseCertificate, Pmat; tol=1e-9) -> PairwiseScreenResult

For every (o,d) with `Pmat[o,d] > 0`, computes
    m_od = min_{k != o} [ M_ok - (a_kd - a_od) ]
`m_od < -tol` is an EXACT rejection certificate: origin o can never beat
rival k on ANY draw (since `a_kd - a_od > M_ok` means the (o,d)-vs-(k,d) hard
score gap origin o would need on its BEST draw is not enough to overcome
rival k's fixed advantage `a_kd - a_od`), hence o can never win destination
d, hence the bilateral target moment `lambda[o,d] Pmat[o,d] > 0` cannot be
matched by ANY reweighting of the draws -- the point is EXACTLY infeasible
(minimum divergence = +infinity).

`tol` is a small numerical slack applied ONLY on the rejecting side (i.e. we
require `m_od < -tol`, not `m_od < 0`) so ordinary floating-point noise at an
exact-equality boundary (measure-zero in real data, but reachable by
construction in adversarial tests) can only make the certificate MORE
conservative -- it can never manufacture a false rejection that a
`tol=0`-exact comparison would not also have made; it exists purely to avoid
flagging a point as certified-infeasible when the TRUE mathematical slack is
exactly 0 up to floating-point rounding (a tie at the boundary, which the
winner-scan stage handles exactly via its own tie-safe win-count, see
`screen_hard_winners`). `tol=1e-9` is a conservative choice, three orders of
magnitude above double-precision accumulation error for the O(D) sums
involved (D<=400 terms of O(1)-scale log-quantities) -- see the validation
report's tolerance-sensitivity check.
"""
function pairwise_certificate(a::AbstractMatrix, pc::PairwiseCertificate, Pmat::AbstractMatrix; tol::Float64 = 1e-9)
    D = pc.D
    m_od = fill(Inf, D, D)
    worst = Inf; wo = 0; wd = 0; wk = 0
    @inbounds for d in 1:D
        for o in 1:D
            Pmat[o, d] > 0 || continue
            best = Inf; bk = 0
            for k in 1:D
                k == o && continue
                slack = pc.M[o, k] - (a[k, d] - a[o, d])
                if slack < best
                    best = slack; bk = k
                end
            end
            m_od[o, d] = best
            if best < worst
                worst = best; wo = o; wd = d; wk = bk
            end
        end
    end
    infeasible = worst < -tol
    return PairwiseScreenResult(infeasible, m_od, wo, wd, wk, worst)
end

# ============================================================================
# SECTION 1: destination-by-destination hard-winner construction with
# immediate zero-win-count rejection, deterministic vulnerability ordering.
# ============================================================================

"""
    order_destinations(pres::PairwiseScreenResult, D) -> Vector{Int}

Deterministic vulnerability ordering (Section 1's "analytical slack"
heuristic): ascending by `min_{o: Pmat[o,d]>0} m_od[o,d]` -- the destination
whose most-vulnerable origin has the most-negative (or least-positive)
pairwise slack is scanned FIRST, since it is the destination most likely to
fail the exact winner-count check. Pure ordering heuristic: does not affect
which points are rejected or their computed values at feasible points (see
`test_infeasibility_screen.jl`'s order-independence check), only how fast a
genuinely-infeasible point is found.
"""
function order_destinations(pres::PairwiseScreenResult, D::Int)
    dest_score = [minimum(@view pres.m_od[:, d]) for d in 1:D]
    return sortperm(dest_score)
end

"""
    order_destinations_by_margin(prev_gap::AbstractMatrix, D) -> Vector{Int}

Alternative ordering heuristic (Section 1's "previous winner margins"):
ascending by the smallest per-destination winner/runner-up price gap
observed at a PREVIOUS nearby outer point (`prev_gap`, W x D, e.g. from a
prior `compute_winners`/`compute_winners_fast` call's `gap` output) --
destinations with a thin margin anywhere in `prev_gap` are more likely to
have flipped a winner (and hence be more likely to have lost a positive-share
origin's only win) at a NEARBY new point. Provided for completeness /
benchmarking per the task brief's explicit "candidate orderings include...
previous winner margins" list; `order_destinations` (the analytical-slack
ordering) is the one actually wired into `evaluate_fullA_screened` by
default since it requires no state from a previous call.
"""
function order_destinations_by_margin(prev_gap::AbstractMatrix, D::Int)
    dest_min_gap = [minimum(@view prev_gap[:, d]) for d in 1:D]
    return sortperm(dest_min_gap)
end

struct WinnerScreenResult
    feasible::Bool
    stage::Int                                    # # destinations completed (in `order`) when decided
    failing_o::Int
    failing_d::Int
    order::Vector{Int}
    winner::Union{Nothing,Matrix{Int}}             # W x D, materialized only if feasible or full_scan
    wval::Union{Nothing,Matrix{Float64}}
    win_counts::Matrix{Int}                        # D(origin) x D(destination); 0 for un-scanned dest.
end

"""
    screen_hard_winners(theta_full, ctx, Pmat; order=1:D, full_scan=false) -> WinnerScreenResult

Destination-major hard-winner construction -- the per-cell price formula
(`constCons`, `constConsσ`, `UPow`, `UσPow`) is a DIRECT, attributed reuse of
`compressed_moments.jl::build_compressed_factual`'s own winner-finding inner
loop (UoModel==1 branch), not re-derived. NEW relative to that function:
(1) processes destinations in caller-supplied `order` (default natural
1:D), (2) after EACH destination completes, immediately checks whether any
origin `o` with `Pmat[o,d] > 0` got zero wins at that destination and, if so,
returns `feasible=false` right away WITHOUT touching any later destination in
`order` -- no moment matrix, no CompressedFactual, no KNITRO call for this
point; (3) win-COUNTING (as opposed to the single `winner[s,d]` index used
for `wval`/hashing) is TIE-SAFE: every origin achieving the exact row-minimum
price on a draw is credited a win, not just the first-index one `winner[s,d]`
records. This matters for EXACTNESS: `MinInd!`/`hFunction!` (the dense
production winner rule) splits primal mass across ALL tied-minimum origins,
not just the lowest-indexed one, so an origin whose only route to winning
destination d is via an exact tie would be (incorrectly) flagged zero-win by
a first-index-only count -- a genuine false-positive risk this fix
eliminates. Ties are a probability-zero event for continuous Frechet draws
in real data (never observed in this investigation, per
`docs/fullA_fully_compressed_inner_report.md`) but are deliberately
constructed by this report's adversarial validation tests, so the fix is not
academic.

`full_scan=true` disables early exit (every destination in `order` is always
scanned) and, if any destination fails, the returned `stage` is `D` (all
destinations touched) -- used ONLY for validation: confirming that early
exit and destination order never change WHICH points get rejected or which
(o,d) pair is reported as first-failing under natural order, only how much
work is skipped.
"""
function screen_hard_winners(θ_full::AbstractVector, ctx, Pmat::AbstractMatrix;
        order::AbstractVector{Int} = 1:ctx.D, full_scan::Bool = false)
    γo = ctx.γ
    D = ctx.D; U = ctx.U; W = size(U, 1)
    μ = θ_full[1]; σ = θ_full[2]
    lambda = reshape(γo.P, (D, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    Aod = Aod_θ .* γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ γo.cHat) .^ (-μ)
    constCons = [γo.wHat[o] * AodPow[o, d] * γo.τ[o, d] for o in 1:D, d in 1:D]
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    constConsσ = [wPow[o] * (AodPow[o, d] * γo.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
    UPow = U .^ (-μ)
    UσPow = γo.Uσ .^ (-μ)

    winner = Matrix{Int}(undef, W, D)
    wval = Matrix{Float64}(undef, W, D)
    win_counts = zeros(Int, D, D)

    for (stage, d) in enumerate(order)
        wc = zeros(Int, D)
        @inbounds for s in 1:W
            best = constCons[1, d] / UPow[s, 1]; bo = 1
            for o in 2:D
                p = constCons[o, d] / UPow[s, o]
                if p < best
                    best = p; bo = o
                end
            end
            winner[s, d] = bo
            wval[s, d] = constConsσ[bo, d] / UσPow[s, bo]
            # tie-safe win credit (see docstring): count every o at the exact row-min, not just bo
            for o in 1:D
                p = constCons[o, d] / UPow[s, o]
                p <= best && (wc[o] += 1)
            end
        end
        win_counts[:, d] .= wc
        if !full_scan
            for o in 1:D
                if Pmat[o, d] > 0 && wc[o] == 0
                    return WinnerScreenResult(false, stage, o, d, collect(order), nothing, nothing, win_counts)
                end
            end
        end
    end

    if full_scan
        for d in 1:D, o in 1:D
            if Pmat[o, d] > 0 && win_counts[o, d] == 0
                return WinnerScreenResult(false, D, o, d, collect(order), winner, wval, win_counts)
            end
        end
    end

    return WinnerScreenResult(true, D, 0, 0, collect(order), winner, wval, win_counts)
end

# ============================================================================
# SECTION 3 (optional, additive diagnostic): exact extreme-draw witness
# query -- decides whether a SINGLE (o,d) pair has >=1 winning draw without
# scanning every draw, via a sorted-difference structure per ordered origin
# pair. NOT wired into `evaluate_fullA_screened`'s default path (use_witness
# kwarg, default false) -- see docs/fullA_D20_infeasibility_screening_report.md
# sec on whether this pays for itself at D=20/W=80,000 and W=800,000.
# ============================================================================

struct SortedDiffPair
    idx::Vector{Int32}     # draw indices, sorted ascending by (B[:,o]-B[:,k])
    val::Vector{Float64}   # corresponding sorted values
end

struct ExtremeDrawWitness
    D::Int
    W::Int
    pairs::Matrix{SortedDiffPair}   # D x D; pairs[o,k] valid for o!=k
end

"""
    build_extreme_draw_witness(ctx) -> ExtremeDrawWitness

Precomputes, for every ordered origin pair (o,k), the draws sorted by
`B[s,o]-B[s,k]` ascending, O(D^2 * W log W) once. Memory: `D*(D-1)` sorted
arrays of length W each, `Int32` indices + `Float64` values -- see the
report's memory-vs-D/W table for whether this is worth building at a given
scale.
"""
function build_extreme_draw_witness(ctx)
    B = hard_score_B(ctx)
    D = ctx.D; W = size(B, 1)
    pairs = Matrix{SortedDiffPair}(undef, D, D)
    @inbounds for k in 1:D, o in 1:D
        o == k && continue
        diff = @view(B[:, o]) .- @view(B[:, k])
        p = sortperm(diff)
        pairs[o, k] = SortedDiffPair(Int32.(p), diff[p])
    end
    return ExtremeDrawWitness(D, W, pairs)
end

"""
    query_witness(o, d, a, B, ew) -> (exists, witness_draw, n_candidates_tested, best_candidate_set_size)

Exact existence test: does origin `o` have >=1 draw on which it wins
destination `d` (beats EVERY rival `k != o` simultaneously on the SAME
draw)? Per the task brief's algorithm:
1. thresholds `t_k = a_kd - a_od` for every rival k;
2. binary-search (`searchsortedlast` on the precomputed sorted array) the
   candidate draw SET for each rival (`{s : B[s,o]-B[s,k] > t_k}`, a
   contiguous suffix of the sorted array by construction);
3. pick the rival with the SMALLEST candidate set;
4. test only those candidate draws directly against all REMAINING rivals
   (not via the sorted structure -- a direct `B[s,o]-B[s,k'] > t_k'` check).
Returns as soon as one candidate beats every rival (a genuine witness); if
every candidate in the smallest set fails, `exists=false` is an EXACT
certificate (no other draw could possibly beat the smallest-set rival
either, by construction of that set).
"""
function query_witness(o::Int, d::Int, a::AbstractMatrix, B::AbstractMatrix, ew::ExtremeDrawWitness)
    D = ew.D
    t = Vector{Float64}(undef, D)   # t[k] valid for k != o
    best_k = 0; best_size = typemax(Int); best_lo = 0
    @inbounds for k in 1:D
        k == o && continue
        tk = a[k, d] - a[o, d]
        t[k] = tk
        sd = ew.pairs[o, k]
        lo = searchsortedlast(sd.val, tk) + 1   # first sorted position with val > tk
        sz = ew.W - lo + 1
        if sz < best_size
            best_size = sz; best_k = k; best_lo = lo
        end
    end
    if best_k == 0
        return (true, 1, 0, ew.W)   # D==1 edge case: no rivals, trivially a winner
    end
    sdbest = ew.pairs[o, best_k]
    n_tested = 0
    @inbounds for pos in best_lo:ew.W
        s = sdbest.idx[pos]
        n_tested += 1
        ok = true
        for k in 1:D
            (k == o || k == best_k) && continue
            if !(B[s, o] - B[s, k] > t[k])
                ok = false; break
            end
        end
        if ok
            return (true, s, n_tested, best_size)
        end
    end
    return (false, 0, n_tested, best_size)
end

# ============================================================================
# SECTION 4: integration wrapper -- structured exact-infeasibility status,
# distinct from a numerical inner-solver failure (e.g. KNITRO nStatus=-300).
# Delegates to the EXISTING, unmodified evaluate_fullA_fast (dense) once the
# screen passes; a SEPARATE screen-aware compressed path
# (evaluate_fullA_screened_compressed) additionally reuses the screen's own
# winner/wval arrays for the compressed inner solve, avoiding the redundant
# winner rescan build_compressed_factual would otherwise perform.
# ============================================================================

"""
    infeasible_result(x_free, theta_full, ctx, screen_status, failing_o, failing_d, stage, t_screen, tag, warm)

Builds a result NamedTuple with the SAME FIELD SET as `evaluate_fullA_fast`'s
return value (so any generic downstream code that destructures by field name
keeps working), with `Delta_dual = Delta_primal = +Inf` (the primal moment
problem is infeasible; the minimum divergence is, per the task spec, exactly
+infinity) and a sentinel `inner_status` OUTSIDE the set of real KNITRO
status codes ({0,-100,-101,-103} solved, {-300,-400,...} numerical failure)
so a certified-exact-infeasible point is never confusable with a numerical
inner-solver failure. Three extra fields (`screen_status`, `screen_failing_o`,
`screen_failing_d`, `screen_stage`) carry the certificate detail.
"""
function infeasible_result(x_free, θ_full, ctx, screen_status::Symbol, failing_o::Int, failing_d::Int,
        stage::Int, t_screen::Float64, tag::String, warm::Bool)
    D = ctx.D
    sentinel = screen_status === :pairwise_certified_infeasible ? -9001 :
               screen_status === :witness_certified_infeasible  ? -9002 :
               screen_status === :winner_scan_infeasible        ? -9003 : -9000
    elapsed = (total = t_screen, inner = 0.0, post = 0.0)
    return (x_free = collect(x_free), θ_full = θ_full,
            gamma_focal_prime = θ_full[3+D], logA = fill(NaN, D, D),
            K_hard = NaN, Delta_dual = Inf, Delta_primal = Inf, Delta_minus_delta = Inf,
            gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
            gravity_R_beta = NaN, moment_resid = Float64[], max_abs_moment_resid = NaN,
            zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
            weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
            winner_hash = UInt64(0), inner_status = sentinel, inner_iters = missing,
            primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
            elapsed = elapsed,
            error_reason = "exact_infeasible ($(screen_status)): origin $failing_o has zero possible " *
                            "wins at destination $failing_d (target share Pmat[$failing_o,$failing_d]>0), " *
                            "stage=$stage/$D destinations scanned before rejection",
            screen_status = screen_status, screen_failing_o = failing_o, screen_failing_d = failing_d,
            screen_stage = stage)
end

"""
    evaluate_fullA_screened(x_free, ctx; moment_representation=:dense, cache=nothing,
                             use_cache=true, use_witness=false, mode=:hard, warm=true,
                             tag="", pairwise=nothing, witness=nothing) -> (result, screen_meta)

Runs the exact infeasibility screen (Section 4's order: draw-free pairwise
certificate -> optional extreme-draw witness test -> destination-by-
destination hard-winner construction with immediate zero-count rejection)
BEFORE calling the real inner-CC-dual oracle. On a certified-infeasible
point, returns immediately (see `infeasible_result`) WITHOUT constructing
moments or invoking KNITRO. On a screen-pass, delegates to
`evaluate_fullA_fast` (`:dense`, completely unmodified) or
`evaluate_fullA_screened_compressed` (`:compressed`, reuses the screen's own
winner/wval arrays -- see that function's docstring) for the real
evaluation, and caches the ordinary feasible result too (same exact-point
cache discipline as `oracle.jl`/`oracle_fast.jl`, same `FullAEvalKey`, so a
screened and unscreened caller can share one cache dict).

`pairwise=` / `witness=` let a caller supply a precomputed
`PairwiseCertificate` / `ExtremeDrawWitness` (both draw-free-at-the-outer-
point-level structures, safe and cheap to build ONCE per ctx.U and reuse
across every outer-point evaluation in a run) instead of paying the O(D^2 W)
/ O(D^2 W log W) build cost on every call.
"""
function evaluate_fullA_screened(x_free::AbstractVector{Float64}, ctx;
        moment_representation::Symbol = :dense,
        cache::Union{Nothing,Dict} = nothing, use_cache::Bool = true,
        use_witness::Bool = false, mode::Symbol = :hard, warm::Bool = true, tag::String = "",
        pairwise::Union{Nothing,PairwiseCertificate} = nothing,
        witness::Union{Nothing,ExtremeDrawWitness} = nothing)

    mode == :hard || error("evaluate_fullA_screened: mode=:$mode not implemented (matches oracle.jl)")
    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode)

    if cache !== nothing && use_cache && haskey(cache, key)
        hit = cache[key]
        return merge(hit, (cache_hit = true, tag = tag)),
               (screen_status = get(hit, :screen_status, :cache_hit), elapsed = 0.0)
    end

    t0 = time()
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    Pmat = target_shares(ctx)
    a = compute_a_od(θ_full, ctx)

    pc = pairwise === nothing ? precompute_pairwise_M(ctx) : pairwise
    pres = pairwise_certificate(a, pc, Pmat)
    if pres.infeasible
        t_screen = time() - t0
        result = infeasible_result(x_free, θ_full, ctx, :pairwise_certified_infeasible,
                                    pres.worst_o, pres.worst_d, 0, t_screen, tag, warm)
        cache !== nothing && (cache[key] = result)
        return result, (screen_status = :pairwise_certified_infeasible, worst_o = pres.worst_o,
                         worst_d = pres.worst_d, worst_k = pres.worst_k, worst_slack = pres.worst_slack,
                         elapsed = t_screen)
    end

    if use_witness
        B = hard_score_B(ctx)
        wt = witness === nothing ? build_extreme_draw_witness(ctx) : witness
        for d in 1:ctx.D, o in 1:ctx.D
            Pmat[o, d] > 0 || continue
            exists, s, ntested, csize = query_witness(o, d, a, B, wt)
            if !exists
                t_screen = time() - t0
                result = infeasible_result(x_free, θ_full, ctx, :witness_certified_infeasible, o, d, 0,
                                            t_screen, tag, warm)
                cache !== nothing && (cache[key] = result)
                return result, (screen_status = :witness_certified_infeasible, worst_o = o, worst_d = d,
                                 elapsed = t_screen)
            end
        end
    end

    order = order_destinations(pres, ctx.D)
    wres = screen_hard_winners(θ_full, ctx, Pmat; order = order)
    if !wres.feasible
        t_screen = time() - t0
        result = infeasible_result(x_free, θ_full, ctx, :winner_scan_infeasible, wres.failing_o,
                                    wres.failing_d, wres.stage, t_screen, tag, warm)
        cache !== nothing && (cache[key] = result)
        return result, (screen_status = :winner_scan_infeasible, worst_o = wres.failing_o,
                         worst_d = wres.failing_d, stage = wres.stage, elapsed = t_screen)
    end
    t_screen_passed = time() - t0

    if moment_representation === :dense
        result, prof_meta = evaluate_fullA_fast(x_free, ctx; cache = nothing, use_cache = false,
                                                  mode = mode, warm = warm, tag = tag)
        result = merge(result, (screen_status = :screen_passed,))
        cache !== nothing && (cache[key] = result)
        return result, (screen_status = :screen_passed, screen_elapsed = t_screen_passed, prof_meta...)
    elseif moment_representation === :compressed
        result, prof_meta = evaluate_fullA_screened_compressed(x_free, θ_full, ctx, wres; warm = warm, tag = tag)
        result = merge(result, (screen_status = :screen_passed,))
        cache !== nothing && (cache[key] = result)
        return result, (screen_status = :screen_passed, screen_elapsed = t_screen_passed, prof_meta...)
    else
        error("evaluate_fullA_screened: moment_representation=:$moment_representation not implemented (only :dense, :compressed)")
    end
end

"""
    compressed_factual_from_screen(theta_full, ctx, wres::WinnerScreenResult) -> CompressedFactual

Builds a `CompressedFactual` (compressed_moments.jl) directly from an
ALREADY-COMPUTED `WinnerScreenResult.winner`/`.wval` (i.e. one that has
already passed `screen_hard_winners`), skipping `build_compressed_factual`'s
own O(W*D^2) winner-finding loop entirely -- every OTHER field
(`build_compressed_factual`'s draw-independent bookkeeping: `Pmat`, `denom`,
`gdiv`, `nrm`, `PMM`, `usePMM`, `SW`, `gammafac`, the counterfactual
price-index column) is a direct, attributed copy of that function's own
post-winner-loop code (lines computing those fields do not depend on how the
winner/wval arrays were produced). Tie count is not re-derived here (the
screen's own tie-safe win-count already established feasibility; an actual
exact tie would need `detect_price_ties`-style handling for the COMPRESSED
representation's one-winner-per-draw assumption specifically -- out of this
function's scope, matches the existing `:compressed` mode's own
`check_ties`-then-fallback discipline, not duplicated here since ties are a
separate, already-solved problem upstream).
"""
function compressed_factual_from_screen(θ_full::AbstractVector, ctx, wres::WinnerScreenResult)
    γo = ctx.γ
    D = ctx.D; U = ctx.U; W = size(U, 1)
    μ = θ_full[1]; σ = θ_full[2]
    ind = γo.indicators
    oci = ctx.obj.outer_constr_index

    lambda = reshape(γo.P, (D, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    Aod = Aod_θ .* γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ γo.cHat) .^ (-μ)
    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]
    Pmat = [γo.P[d + (o - 1) * D] for o in 1:D, d in 1:D]

    gammafac = spgamma(μ * (1 - σ) + 1)
    ncol = oci - 1
    gdiv = [j <= D^2 + 1 ? 1.0 / gammafac : 1.0 for j in 1:ncol]
    NM = ind.NormalizeMoments
    without = γo.moments_without_var
    nrm = [(NM == 1 && !(j in without)) ? 1.0 / γo.σ_Moments[j] : 1.0 for j in 1:ncol]
    usePMM = ind.usePMM
    PMMv = usePMM == 1 ? Float64[γo.PMM[j] for j in 1:ncol] : zeros(ncol)
    SW = γo.SamplingWeights[1:W]

    cf_col = D^2 + 1
    cf_raw = zeros(W)
    if cf_col <= ncol
        bi = ctx.bi
        wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
        wPrime_bi = wPrime[bi]
        τPrime_bi = γo.τPrime[bi, bi]
        LPrime_bi = γo.LPrime[bi]
        AodPow_bibi = AodPow[bi, bi]
        γ_prime_bi = θ_full[3 + D]
        constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi * τPrime_bi)^(1 - σ)
        denom_cf = γ_prime_bi^σ * (wPrime_bi * LPrime_bi)
        UσPow_bi = @view(γo.Uσ[:, bi]) .^ (-μ)
        @. cf_raw = constConsσ_bibi / UσPow_bi - denom_cf
    else
        cf_col = 0
    end

    return CompressedFactual(D, W, oci, wres.winner, wres.wval, Pmat, denom, gdiv, nrm,
        PMMv, usePMM, SW, gammafac, cf_raw, cf_col, 0, Tuple{Int,Int}[])
end

"""
    evaluate_fullA_screened_compressed(x_free, theta_full, ctx, wres::WinnerScreenResult;
                                        warm=true, tag="") -> (result, prof_meta)

Screen-aware compressed evaluation for an ALREADY-SCREEN-PASSED point (see
`evaluate_fullA_screened`, `moment_representation=:compressed`). Builds `cf`
directly from `wres` (`compressed_factual_from_screen`, no winner rescan),
then runs the EXISTING `inner_loop_KNITRO_compressed` (compressed_live.jl,
unmodified) and a tail that DELIBERATELY duplicates
`evaluate_fullA_fast_compressed`'s own post-solve bookkeeping (same
provable-non-interference rationale that file's own header states), except
the final `winner_hash` reuses `wres.winner` directly instead of paying a
THIRD winner computation (dense `moments!`'s internal MinInd! is the first,
unavoidable without touching hFunction.jl; `evaluate_fullA_fast_compressed`'s
own `compute_winners_fast` tail call would have been the third) -- this is
the genuine, demonstrable production win of routing a screen-passed point
through the compressed path specifically, reported with numbers in
docs/fullA_D20_infeasibility_screening_report.md.
"""
function evaluate_fullA_screened_compressed(x_free::AbstractVector{Float64}, θ_full::AbstractVector,
        ctx, wres::WinnerScreenResult; warm::Bool = true, tag::String = "")
    obj = ctx.obj
    t_total0 = time()
    if !warm
        obj.x .= NaN
    end

    cf = compressed_factual_from_screen(θ_full, ctx, wres)
    W = size(obj.U, 1)
    SW = ctx.γ.SamplingWeights[1:W]
    obj.H[:, 1] .= θ_full[3 + ctx.D] .* SW
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest
    grav_raw = compressed_gravity_raw(θ_full, ctx)

    st = CompressedCBState(obj, cf, grav_raw, false)
    t_inner0 = time()
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed(obj, st)
    t_inner = time() - t_inner0

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end
    inner_x = x
    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        K_hard = obj.H_save
    else
        obj.x .= NaN
        K_hard = -1e10
    end

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        elapsed = (total = time() - t_total0, inner = t_inner, post = 0.0)
        result = (x_free = collect(x_free), θ_full = θ_full,
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, ctx.D),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, moment_resid = Float64[], max_abs_moment_resid = NaN,
                  zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
                  weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
                  winner_hash = UInt64(0), inner_status = nStatus, inner_iters = inner_iters,
                  primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
                  elapsed = elapsed, error_reason = "inner solve failed: nStatus=$nStatus")
        prof_meta = (n_inner_solves = 1, n_inner_infeasible = 1, n_inner_iters = inner_iters,
                     n_fg_calls = n_fg, n_hess_calls = n_hess)
        return result, prof_meta
    end

    if !st.dense_materialized
        ncolI = st.cf.oci - 1
        materialize_dense_factual!(@view(obj.H[:, 3:2+ncolI]), st.cf)
        fill_gravity_column!(obj, st.grav_raw)
        st.dense_materialized = true
    end

    d = obj.d
    K, G = (copy(@view(obj.H[:, 1])), copy(CS.select_G_from_H(obj, obj.H)))

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = obj(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    ζstar = inner_x[1]; λstar = inner_x[2:end]
    nkkt = min(length(λstar), size(G, 2))
    max_abs_moment_kkt_resid = begin
        acc = 0.0
        @inbounds for j in 1:nkkt
            s = 0.0
            for ω in 1:W
                s += m_weights[ω] * G[ω, j]
            end
            acc = max(acc, abs(s / W))
        end
        acc
    end

    gravity_raw = obj.outer_constr_index <= d ? cbuf[2] : NaN
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D^2], ctx.D, ctx.D)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (ctx.D, ctx.D))'
    Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
    gravity_val = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)
    logA = -log.(AodPow)
    R_sum = sum(ctx.q_tilde .* logA)
    R_mean = R_sum / ctx.D^2
    R_beta = R_sum / sum(ctx.q_tilde .^ 2)

    moment_resid = begin
        mr = zeros(d)
        @inbounds for j in 1:d, ω in 1:W
            mr[j] += G[ω, j]
        end
        mr ./= W
        mr
    end
    max_abs_moment_resid = isempty(moment_resid) ? NaN : maximum(abs.(moment_resid))

    winner_hash = hash(wres.winner)   # REUSED, not recomputed -- the screen's own winner IS the answer

    t_total = time() - t_total0
    elapsed = (total = t_total, inner = t_inner, post = t_total - t_inner)

    result = (x_free = collect(x_free), θ_full = θ_full,
              gamma_focal_prime = θ_full[3+ctx.D], logA = logA,
              K_hard = K_hard, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              Delta_minus_delta = Delta_dual - obj.δ,
              gravity_raw = gravity_raw, gravity_value = gravity_val,
              gravity_R_sum = R_sum, gravity_R_mean = R_mean, gravity_R_beta = R_beta,
              moment_resid = moment_resid, max_abs_moment_resid = max_abs_moment_resid,
              zeta = ζstar, lambda = collect(λstar),
              m_mean = sum(m_weights)/W, m_min = minimum(m_weights), m_max = maximum(m_weights),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              winner_hash = winner_hash, inner_status = nStatus, inner_iters = inner_iters,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              cache_hit = false, warm_started = warm, tag = tag,
              elapsed = elapsed, error_reason = nothing)

    prof_meta = (n_inner_solves = 1, n_inner_infeasible = 0, n_inner_iters = inner_iters,
                 n_fg_calls = n_fg, n_hess_calls = n_hess)
    return result, prof_meta
end
