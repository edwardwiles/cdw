# Matrix-free Melitz moment operator (2026-07-26). See
# docs/melitz_matrix_free_inner_operator_2026-07-26.md for the full derivation and
# validation. Governing prompt: eliminate the materialized W x (D^2+1) moment matrix `G`
# from the Melitz inner objective/gradient hot path while preserving the EXACT same
# discrete CC moment system as `melitz_moments!`/`melitz_moments_sorted_tail!`.
#
# ALGEBRA (docs Section A). For a fixed origin `o`, `G[w, trade_index[o,d]] =
# coef_od*z_power[w,o]*active_od(w) - lambda_od` (`moments.jl`/`sorted_tail.jl`'s own
# derivation, reused verbatim -- NOT re-derived here). Writing `R = coef.*z_power.*active`
# (the active-contribution matrix) and `lambda` the D^2-vector of empirical shares,
# `G_trade = R - ones(W)*lambda'` -- a structured active-tail matrix plus a rank-one
# constant correction. This file exploits that structure directly:
#
# `G*mu` (a fixed-theta dual-argument application, `mul_G!`): for a draw `z` at origin `o`,
# the destinations active are exactly those with `cutoff_od < z` -- a PREFIX in
# ascending-cutoff order. Sorting the (small, length-D) cutoffs ONCE PER OUTER POINT (not
# once per call -- the key improvement over the existing `sorted_dual_argument.jl`
# prototype, which re-sorts and re-binary-searches on every single call) and precomputing,
# for every draw, a bin index `bin[s,o] = #{d : cutoff_od < z_so}` via one O(W+D) merge
# sweep (not a per-draw binary search) turns "which destinations are active, and what do
# they weight-sum to" into: build a length-(D+1) cumulative array from the CURRENT `mu`
# (O(D) per origin, since destination order is fixed for the whole outer point), then one
# O(1) array lookup per draw. Total O(W*D) per call (D origins x O(W) each), vs the dense
# reference's O(W*D) SAME order but with a per-draw `melitz_firm` evaluation instead of a
# multiply-add, and vs `sorted_dual_argument.jl`'s O(W*D*log(D)) (repeated per-call sorting
# and per-draw binary search).
#
# `G'*v` (a fixed-theta moment-gradient application, `mul_Gt!`): for a fixed origin `o` and
# bin `b`, define `B_o[b] = sum_{s: bin[s,o]=b} v_s*y_{s,o}` (`y = z^(sigma-1)`) -- ONE scan
# over `W` draws using the precomputed `bin` lookup (no search). A suffix sum over bins,
# high-to-low, converts bin sums into "active-tail sums" for every destination in O(D). The
# rank-one `-lambda_od*sum(v)` correction is added directly. O(W*D) total, matching the
# governing prompt's Phase 5.
#
# Both `mul_G!`/`mul_Gt!` are allocation-free after construction: every array they touch
# (`cum`/`binsum`/`tail` scratch, `bin`/`coef`/`lambda`/`order`/`rank`/`ell`) is preallocated
# once in `MelitzMomentOperator` and only ever mutated in place by
# `melitz_update_moment_operator!` (called once per NEW outer point, not once per inner
# KNITRO iteration).
#
# The focal link column (`layout.focal_link_index`) is, as in every prior sorted-tail
# session (`sorted_tail.jl`/`sorted_dual_argument.jl` Section C.1), kept DENSE -- but unlike
# `sorted_dual_argument.jl` (which recomputed the entire O(D*W) autarky/link firm loop on
# EVERY call), this operator precomputes the link column `ell` ONCE per outer-point update
# and reuses it for every subsequent `mul_G!`/`mul_Gt!` call, an additional real saving this
# session adds on top of the existing Phase 8 prototype.

"""
    MelitzMomentOperator

Matrix-free representation of the Melitz moment matrix `G` (`W x (D^2+1)`) at a FIXED outer
point `theta`. Two levels of state:

- Immutable draw-level state: `sorted_ctx` (reused verbatim from `sorted_tail.jl`,
  never mutated by this file), `layout`.
- Mutable fixed-outer-point state, updated in place by `melitz_update_moment_operator!`
  whenever `theta` changes: `coef`, `lambda` (`D x D`), `order`/`rank` (destinations sorted
  by ascending cutoff, per origin, and its inverse), `bin` (`W x D`, `UInt8` since `D<=255`
  is asserted at construction), `ell` (`W`-vector, the dense focal-link column).
- Scratch workspaces (`cum`, `binsum`, `tail`, each length `D+1`): reused across every
  `mul_G!`/`mul_Gt!` call, never resized.

`fingerprint` mirrors `sorted_ctx.fingerprint` -- a caller can compare it to detect a stale
operator built against different draws/W/sigma before reusing it (same discipline as
`MelitzSortedTailContext`; content-equality of `p`/`eq`/`cf` at the CURRENT outer point is
NOT tracked here, by the same deliberate-scope-tradeoff `sorted_tail.jl`'s own Section E
residual-risk note already discloses for the moment-matrix construction path -- comparing
theta content on every call would cost `O(D^2)` and defeat the point of caching it).
"""
mutable struct MelitzMomentOperator
    D::Int
    W::Int
    sorted_ctx::MelitzSortedTailContext
    layout::MelitzMomentLayout
    fingerprint::UInt

    coef::Matrix{Float64}      # D x D, c_od = melitz_C(...)/expenditure_d
    lambda::Matrix{Float64}    # D x D, lambda_od = X_data[o,d]/expenditure_d
    order::Matrix{Int}         # D x D, order[:,o] = destinations sorted ascending by cutoff_od
    rank::Matrix{Int}          # D x D, rank[d,o] = inverse of order[:,o] (1-based position)
    bin::Matrix{UInt8}         # W x D, bin[s,o] = #{d : cutoff_od < z_original[s,o]}, 0..D
    ell::Vector{Float64}       # W, dense focal-link column at the current outer point

    prefixC::Vector{Float64}   # D+1 scratch (melitz_update_moment_operator!'s focal-link build,
                                # ascending-cutoff-rank prefix sum of C_{j,d} at the focal origin j)
    prefixF::Vector{Float64}   # D+1 scratch, same rank order, prefix sum of f_{j,d}

    cum::Vector{Float64}       # D+1 scratch (mul_G!)
    binsum::Vector{Float64}    # D+1 scratch (mul_Gt!, and REPURPOSED by melitz_same_origin_weighted_block!)
    tail::Vector{Float64}      # D+1 scratch (mul_Gt!, and REPURPOSED by melitz_same_origin_weighted_block!)
    cutoff_scratch::Vector{Float64}  # D scratch (melitz_update_moment_operator!)

    # Phase 6-9 (Hessian) scratch, reused across every melitz_full_weighted_gram! call:
    contab::Matrix{Float64}    # (D+1) x (D+1) cross-origin bin contingency table
    tail2d::Matrix{Float64}    # (D+1) x (D+1) 2D suffix sum of contab
    gS_scratch::Vector{Float64}     # num_moments, G'*S
    gSell_scratch::Vector{Float64}  # num_moments, G'*(S.*ell)
    Sell_scratch::Vector{Float64}   # W, S.*ell
    RtS_scratch::Matrix{Float64}    # D x D, R_trade'*S (pure active-contribution transpose-apply)
    hblock_scratch::Matrix{Float64} # D x D, reused for both same- and cross-origin R'SR blocks

    # Per-thread scratch for melitz_full_weighted_gram_parallel! (Phase 7 parallelization),
    # sized to Threads.maxthreadid() AT CONSTRUCTION TIME -- one vector of buffers per kind,
    # indexed by Threads.threadid() inside the parallel region, never resized. Following this
    # repo's own established `direct_gradient.jl`/`sorted_crossing_gradient.jl` convention
    # (`nt = Threads.maxthreadid()`, not `Threads.nthreads()`).
    hblock_t::Vector{Matrix{Float64}}   # nt buffers, each D x D
    contab_t::Vector{Matrix{Float64}}   # nt buffers, each (D+1) x (D+1)
    tail2d_t::Vector{Matrix{Float64}}   # nt buffers, each (D+1) x (D+1)
    binsum2_t::Vector{Vector{Float64}}  # nt buffers, each D+1
    tail2_t::Vector{Vector{Float64}}    # nt buffers, each D+1

    # 2026-07-26 closure session (governing prompt Phase 7): matrix-free range screen support.
    # `first_active_pos[t+1,o]` (t=0..D, 1-indexed row t+1) is the smallest SORTED position
    # `pos` (into `sorted_ctx.sorted_z`/`sorted_z_power[:,o]`) at which the merge sweep's
    # `ptr` first reaches >= t -- i.e. the first (ascending-z) draw active for a destination
    # whose rank is `t`. `W+1` (`op.W+1`) is the sentinel for "never reached" (no draw is
    # active at that threshold at this outer point). Fused directly into the SAME merge
    # sweep `melitz_update_moment_operator!` already runs for `bin` -- no extra O(W*D) pass.
    first_active_pos::Matrix{Int}   # (D+1) x D
end

"""
    build_melitz_moment_operator(sorted_ctx, layout) -> MelitzMomentOperator

Allocates every persistent array ONCE, sized from `sorted_ctx.D`/`sorted_ctx.W`. The
fixed-outer-point state (`coef`/`lambda`/`order`/`rank`/`bin`/`ell`) is left uninitialized
(zeros) until the first `melitz_update_moment_operator!` call -- callers must call it before
`mul_G!`/`mul_Gt!` (checked via `fingerprint`/an explicit `stale` guard, not silently
tolerated).
"""
function build_melitz_moment_operator(sorted_ctx::MelitzSortedTailContext, layout::MelitzMomentLayout)
    D = sorted_ctx.D
    W = sorted_ctx.W
    D <= 255 || throw(ArgumentError(
        "build_melitz_moment_operator: D=$D exceeds 255, the UInt8 bin-index capacity assumed by this operator"))
    D == layout.D || throw(ArgumentError(
        "build_melitz_moment_operator: layout.D=$(layout.D) != sorted_ctx.D=$D"))

    K = layout.num_moments
    nt = Threads.maxthreadid()
    return MelitzMomentOperator(D, W, sorted_ctx, layout, UInt(0),
        zeros(D, D), zeros(D, D), zeros(Int, D, D), zeros(Int, D, D),
        zeros(UInt8, W, D), zeros(W),
        zeros(D + 1), zeros(D + 1),
        zeros(D + 1), zeros(D + 1), zeros(D + 1), zeros(D),
        zeros(D + 1, D + 1), zeros(D + 1, D + 1), zeros(K), zeros(K), zeros(W),
        zeros(D, D), zeros(D, D),
        [zeros(D, D) for _ in 1:nt], [zeros(D + 1, D + 1) for _ in 1:nt],
        [zeros(D + 1, D + 1) for _ in 1:nt], [zeros(D + 1) for _ in 1:nt], [zeros(D + 1) for _ in 1:nt],
        zeros(Int, D + 1, D))
end

"""
    melitz_update_moment_operator!(op, p, eq, cf; X_data=eq.trade_flow) -> op

Rebuilds every fixed-outer-point field of `op` in place from the current `(p, eq, cf)`.
`O(W*D)` (the bin merge-sweep + the dense link column), amortized across every subsequent
`mul_G!`/`mul_Gt!` call for the life of this outer point -- NOT allocation-free (the
per-origin `sortperm` of `D` cutoffs is a small, `O(D)`-sized allocation; `D` is at most a
few hundred in this codebase's real use, and this function runs once per outer evaluation,
not once per inner KNITRO iteration, so it is explicitly exempt from the hot-path zero-alloc
requirement -- see Phase 14's own "operator-state update" allocation-gate category, reported
separately from the per-call callbacks).

Requires `op.D == p.D` and `op.sorted_ctx.sigma == p.sigma` (a stale operator built under a
different `D`/`sigma` throws `ArgumentError` rather than silently producing wrong output).
"""
function melitz_update_moment_operator!(op::MelitzMomentOperator, p::MelitzPrimitives,
                                         eq::MelitzEquilibrium, cf::MelitzCounterfactual;
                                         X_data::AbstractMatrix=eq.trade_flow)
    D = op.D
    W = op.W
    sorted_ctx = op.sorted_ctx
    D == p.D || throw(ArgumentError("melitz_update_moment_operator!: op.D=$D != p.D=$(p.D)"))
    sorted_ctx.sigma == p.sigma || throw(ArgumentError(
        "melitz_update_moment_operator!: sorted_ctx.sigma=$(sorted_ctx.sigma) != p.sigma=$(p.sigma) " *
        "-- stale sorted context (built under a different sigma)"))
    sigma = p.sigma

    @inbounds for o in 1:D
        cutoff_o = op.cutoff_scratch
        for d in 1:D
            cutoff_o[d] = eq.cutoff[o, d]
            C_od = melitz_C(p.w[o], p.tau[o, d], p.A[o, d], sigma, eq.expenditure[d])
            op.coef[o, d] = C_od / eq.expenditure[d]
            op.lambda[o, d] = X_data[o, d] / eq.expenditure[d]
        end
        order_o = sortperm(cutoff_o)
        @views op.order[:, o] .= order_o
        for m in 1:D
            op.rank[order_o[m], o] = m
        end

        # Merge sweep: ptr is monotone nondecreasing across the whole pass over sorted
        # draws, so this is O(W+D), never a per-draw binary search (Phase 2.2's own
        # requirement).
        # 2026-07-26 closure session (Phase 7): fused `first_active_pos` computation into
        # this SAME sweep -- `ptr` only ever increases, so whenever it advances (by one step
        # or, when a single z-jump crosses several cutoffs at once, by several), `pos` is
        # exactly the first sorted position at which EVERY newly-crossed level (and any level
        # skipped over by a multi-step jump) becomes reachable. No separate O(W*D) pass.
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        perm_o = @view sorted_ctx.permutation[:, o]
        ptr = 0
        op.first_active_pos[1, o] = 1   # t=0: every draw is trivially "active" (bin>=0)
        for pos in 1:W
            z = sorted_z_o[pos]
            while ptr < D && cutoff_o[order_o[ptr+1]] < z
                ptr += 1
                op.first_active_pos[ptr+1, o] = pos
            end
            s = perm_o[pos]
            op.bin[s, o] = UInt8(ptr)
        end
        # Levels never reached by the end of the sweep (no draw activates them at this outer
        # point) get the explicit "unreachable" sentinel W+1, not a stale prior value.
        for level in (ptr+1):D
            op.first_active_pos[level+1, o] = W + 1
        end
    end

    j = p.target_country
    z_orig = sorted_ctx.z_original
    w_j = p.w[j]
    @melitz_profile :moment_operator_link_update begin
        # 2026-07-30 O(W*D) -> O(W+D) reformation (docs
        # melitz_profiledA_parallel_speed_and_cutoff_portfolio_2026-07-30.md Phase 4). For
        # FIXED q (participation invariant throughout a middle-loop A-search -- this
        # module's own `bin`/`order`/`rank` above are exactly the authoritative
        # participation source, already relied on by mul_G!/mul_Gt!), destination d is
        # active for draw w at origin j iff `rank[d,j] <= bin[w,j]` -- a PREFIX in
        # ascending-cutoff order (same fact `mul_G!`'s own header derivation already uses).
        # `realized_operating_profit_d(z) = C_{j,d}*z^(sigma-1)/sigma - w_j*f_{j,d}` when
        # active (melitz_firm/firm_quantities.jl, `price_power_d=1.0` here), so the sum over
        # active d collapses to two length-(D+1) prefix sums (built ONCE, O(D)) plus one O(1)
        # lookup per draw -- not a fresh melitz_firm call at every (w,d) pair.
        order_j = @view op.order[:, j]
        prefixC = op.prefixC
        prefixF = op.prefixF
        prefixC[1] = 0.0
        prefixF[1] = 0.0
        @inbounds for rank in 1:D
            d = order_j[rank]
            C_jd = melitz_C(w_j, p.tau[j, d], p.A[j, d], sigma, eq.expenditure[d])
            prefixC[rank+1] = prefixC[rank] + C_jd
            prefixF[rank+1] = prefixF[rank] + p.f[j, d]
        end
        bin_j = @view op.bin[:, j]
        @inbounds for w in 1:W
            z = z_orig[w, j]
            b = Int(bin_j[w]) + 1
            op.ell[w] = (z^(sigma - 1) * prefixC[b] / sigma - w_j * prefixF[b]) / w_j
        end
        price_power_autarky = p.gamma_prime_target
        @inbounds for w in 1:W
            z_j = z_orig[w, j]
            firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], sigma,
                                        cf.expenditure_prime, price_power_autarky, z_j)
            op.ell[w] -= firm_autarky.realized_operating_profit / cf.w_prime
        end
    end

    op.fingerprint = sorted_ctx.fingerprint
    return op
end

"""
    mul_G!(u, op, zeta, mu) -> u

Matrix-free objective/dual-argument callback (Phase 4): fills `u[w] = -zeta -
dot(G[w,:], mu)` (the SAME quantity `melitz_dense_dual_argument`/`melitz_sorted_dual_argument`
compute -- `mu` has length `op.layout.num_moments`, trade cells first then the focal link
column, matching `MelitzMomentLayout`'s own convention). Zero-allocation after `op`'s own
construction: `op.cum` is the only scratch touched, reused across origins and calls.

No binary search, no sort, no mask construction at call time -- every per-draw lookup is
`op.bin[s,o]`, precomputed by `melitz_update_moment_operator!`.
"""
function mul_G!(u::AbstractVector{Float64}, op::MelitzMomentOperator, zeta::Real,
                 mu::AbstractVector{Float64})
    D = op.D
    W = op.W
    length(u) == W || throw(ArgumentError("mul_G!: u must have length W=$W"))
    length(mu) == op.layout.num_moments || throw(ArgumentError(
        "mul_G!: mu must have length op.layout.num_moments=$(op.layout.num_moments)"))

    zeta_f = Float64(zeta)
    @inbounds for s in 1:W
        u[s] = -zeta_f
    end

    trade_index = op.layout.trade_index
    coef = op.coef
    lambda = op.lambda
    order = op.order
    bin = op.bin
    cum = op.cum
    z_power = op.sorted_ctx.z_power_original

    @inbounds for o in 1:D
        const_o = 0.0
        cum[1] = 0.0
        for m in 1:D
            d = order[m, o]
            mu_od = mu[trade_index[o, d]]
            cum[m+1] = cum[m] + mu_od * coef[o, d]
            const_o += mu_od * lambda[o, d]
        end
        for s in 1:W
            u[s] += const_o
        end
        for s in 1:W
            b = bin[s, o]
            b == 0 && continue
            u[s] -= z_power[s, o] * cum[b+1]
        end
    end

    mu_link = mu[op.layout.focal_link_index]
    if mu_link != 0.0
        ell = op.ell
        @inbounds for s in 1:W
            u[s] -= mu_link * ell[s]
        end
    end

    return u
end

"""
    mul_Gt!(g, op, v) -> g

Matrix-free moment-gradient callback (Phase 5): fills `g = G' * v` (`g` has length
`op.layout.num_moments`; `v` has length `W`) via one `O(W)` bin-sum scan per origin, a
suffix sum over the `D+1` bins (`O(D)`), and the rank-one `-lambda_od*sum(v)` correction
added directly. Zero-allocation after construction (`op.binsum`/`op.tail` scratch reused).
"""
function mul_Gt!(g::AbstractVector{Float64}, op::MelitzMomentOperator, v::AbstractVector{Float64})
    D = op.D
    W = op.W
    length(v) == W || throw(ArgumentError("mul_Gt!: v must have length W=$W"))
    length(g) == op.layout.num_moments || throw(ArgumentError(
        "mul_Gt!: g must have length op.layout.num_moments=$(op.layout.num_moments)"))

    sumv = 0.0
    @inbounds for s in 1:W
        sumv += v[s]
    end

    trade_index = op.layout.trade_index
    coef = op.coef
    lambda = op.lambda
    rank = op.rank
    bin = op.bin
    binsum = op.binsum
    tail = op.tail
    z_power = op.sorted_ctx.z_power_original

    @inbounds for o in 1:D
        for k in 1:D+1
            binsum[k] = 0.0
        end
        for s in 1:W
            b = bin[s, o]
            binsum[b+1] += v[s] * z_power[s, o]
        end
        tail[D+1] = binsum[D+1]
        for k in D:-1:1
            tail[k] = tail[k+1] + binsum[k]
        end
        for d in 1:D
            m = rank[d, o]
            tail_sum_d = tail[m+1]
            g[trade_index[o, d]] = coef[o, d] * tail_sum_d - lambda[o, d] * sumv
        end
    end

    ell = op.ell
    acc = 0.0
    @inbounds for s in 1:W
        acc += ell[s] * v[s]
    end
    g[op.layout.focal_link_index] = acc

    return g
end

"""
    melitz_dense_Gt_v(v, G) -> Vector{Float64}

Reference (dense) construction of `G' * v`, given an already-built moment matrix `G`
(e.g. from `melitz_moments!`). Used as the correctness reference for `mul_Gt!` (test suite
only, not a production hot path).
"""
function melitz_dense_Gt_v(v::AbstractVector{Float64}, G::AbstractMatrix{Float64})
    return G' * v
end

# ============================================================================
# Phase 6/7 (scoped prototype, 2026-07-26): same-origin weighted-Gram block.
#
# SCOPE NOTE (see docs/melitz_matrix_free_inner_operator_2026-07-26.md Section E for the
# full accounting): this is the SAME-ORIGIN diagonal block of the trade-trade curvature
# matrix `R_trade' * Diagonal(S) * R_trade` ONLY (`R_trade` the active-contribution matrix,
# `G_trade = R_trade - ones(W)*lambda'`), where `S` is the per-draw curvature weight vector
# `ddPsi!(arg0)` -- NOT the dual vector `mu`/`x` itself (the governing prompt's own explicit
# clarification). It is a genuine, tested, zero-allocation building block, NOT a complete
# Hessian: the cross-origin blocks (`R_o' S R_p`, `o != p`, via cutoff-bin contingency
# tables) and the rank-one `-(R'S1)*lambda' - lambda*(1'SR) + (1'S1)*lambda*lambda'`
# correction (which mixes every origin pair, not just the diagonal) are DERIVED in the
# report but NOT implemented here -- assembling a complete, correct Hessian requires both,
# and shipping only the diagonal block as if it were the whole Hessian would be worse than
# not shipping it at all.
#
# ONE TRIANGLE ONLY (per direct user instruction, mid-session): KNITRO's own Hessian
# callback (`cc_algo/PsiObjectiveBundle.jl`'s `hessian!`) already only READS the upper
# triangle out of the dense reference's full symmetric `∂∂f_∂∂x` (packed `for i in 1:n, for
# j in i:n`) -- this function fills ONLY that same upper triangle (`d <= dp`) directly, and
# the dense reference's own `BLAS.gemm!('T','N',...)` (which computes both triangles before
# discarding half) should be compared against `BLAS.syrk!` in any live benchmark, not
# `gemm!`, since `R_o'*Diagonal(S)*R_o` is exactly the symmetric-rank-k form `syrk!` targets.
# ============================================================================

"""
    melitz_same_origin_weighted_block!(Hblock, op, o, S) -> Hblock

Fills the UPPER TRIANGLE ONLY (`Hblock[d, dp]` for `d <= dp`; `Hblock[dp, d]` for `d < dp`
is left untouched -- callers must not read it) of the same-origin block
`(R_o' * Diagonal(S) * R_o)[d, dp] = coef[o,d]*coef[o,dp] * sum_{s active at BOTH d and dp}
S_s * y_{s,o}^2`. "Active at both" is exactly `bin[s,o] >= max(rank[d,o], rank[dp,o])`
(Section A.1's monotonicity argument: participation is a single threshold test per cell, so
joint participation is thresholded at the STRICTER of the two cutoffs) -- computed via one
`O(W)` scan building `y^2`-weighted bin sums, then an `O(D)` suffix sum, then `O(D^2/2)`
direct lookups (upper triangle only). Zero-allocation: reuses `op.binsum`/`op.tail`
(REPURPOSED here for `S*y^2` weighting rather than `mul_Gt!`'s `v*y` -- callers must not
interleave a `mul_Gt!` call with an in-progress `melitz_same_origin_weighted_block!` loop
over origins, since both mutate the same scratch fields).
"""
function melitz_same_origin_weighted_block!(Hblock::AbstractMatrix{Float64}, op::MelitzMomentOperator,
                                             o::Int, S::AbstractVector{Float64})
    size(Hblock) == (op.D, op.D) || throw(ArgumentError("melitz_same_origin_weighted_block!: Hblock must be D x D"))
    length(S) == op.W || throw(ArgumentError("melitz_same_origin_weighted_block!: S must have length W"))
    1 <= o <= op.D || throw(ArgumentError("melitz_same_origin_weighted_block!: o out of range"))
    return _same_origin_weighted_block_buf!(Hblock, op, o, S, op.binsum, op.tail)
end

"""
    _same_origin_weighted_block_buf!(Hblock, op, o, S, binsum2, tail2) -> Hblock

Internal implementation shared by `melitz_same_origin_weighted_block!` (serial) and
`melitz_full_weighted_gram_parallel!` (threaded, thread-local `binsum2`/`tail2`). No
argument checks (callers already validated); no allocation.
"""
function _same_origin_weighted_block_buf!(Hblock::AbstractMatrix{Float64}, op::MelitzMomentOperator,
                                           o::Int, S::AbstractVector{Float64},
                                           binsum2::AbstractVector{Float64}, tail2::AbstractVector{Float64})
    D = op.D
    W = op.W
    bin = op.bin
    rank = op.rank
    coef = op.coef
    z_power = op.sorted_ctx.z_power_original

    @inbounds for k in 1:D+1
        binsum2[k] = 0.0
    end
    @inbounds for s in 1:W
        b = bin[s, o]
        y = z_power[s, o]
        binsum2[b+1] += S[s] * y * y
    end
    tail2[D+1] = binsum2[D+1]
    @inbounds for k in D:-1:1
        tail2[k] = tail2[k+1] + binsum2[k]
    end

    @inbounds for d in 1:D
        rd = rank[d, o]
        cd = coef[o, d]
        for dp in d:D
            rdp = rank[dp, o]
            m = max(rd, rdp)
            Hblock[d, dp] = cd * coef[o, dp] * tail2[m+1]
        end
    end
    return Hblock
end

# ============================================================================
# Phase 7/8 (2026-07-26 continuation): cross-origin contingency-table block +
# rank-one correction + normalization/focal-link Hessian blocks -- completing the FULL
# matrix-free `G_full' * Diagonal(S) * G_full` (`G_full = [ones(W) G]`, matching
# `cc_algo/PsiObjectiveBundle.jl`'s own `H[:,2:1+outer_constr_index]` convention exactly),
# not merely the same-origin diagonal block this session's earlier part validated.
# ============================================================================

"""
    melitz_cross_origin_weighted_block!(Hblock, op, o, p, S) -> Hblock

Fills the FULL `D x D` cross-origin block `(R_o' * Diagonal(S) * R_p)[d, dp]` for two
DISTINCT origins `o < p` (required -- the caller must never compute the `(p,o)` block
separately, since it is exactly this block's transpose and the full Hessian's upper
triangle already contains every cross-origin pair exactly once when origins are laid out
in increasing column order, `MelitzMomentLayout`'s own `trade_index[o,:]` convention).

Uses the exact identity `Active_od(s)*Active_pd'(s) = 1{bin[s,o]>=rank[d,o]} *
1{bin[s,p]>=rank[dp,p]}` (Section A.1's monotonicity argument applied independently to each
origin's own draw column -- origins `o`/`p` share the joint row `s` but have INDEPENDENT
cutoff orders). A `(D+1) x (D+1)` bin-pair contingency table
`contab[a,b] = sum_{s: bin[s,o]=a, bin[s,p]=b} S_s*y_{s,o}*y_{s,p}` is built via one `O(W)`
scan, then converted to a 2D suffix sum (`tail2d[a,b] = sum_{a'>=a,b'>=b} contab[a',b']`,
`O(D^2)`, computed from the high-bin corner outward via the standard 2D
inclusion-exclusion recurrence) so that every one of the `D^2` `(d,dp)` lookups is `O(1)`.
Zero-allocation: `op.contab`/`op.tail2d` are the only scratch touched.
"""
function melitz_cross_origin_weighted_block!(Hblock::AbstractMatrix{Float64}, op::MelitzMomentOperator,
                                              o::Int, p::Int, S::AbstractVector{Float64})
    o < p || throw(ArgumentError("melitz_cross_origin_weighted_block!: requires o < p"))
    size(Hblock) == (op.D, op.D) || throw(ArgumentError("melitz_cross_origin_weighted_block!: Hblock must be D x D"))
    length(S) == op.W || throw(ArgumentError("melitz_cross_origin_weighted_block!: S must have length W"))
    return _cross_origin_weighted_block_buf!(Hblock, op, o, p, S, op.contab, op.tail2d)
end

"""
    _cross_origin_weighted_block_buf!(Hblock, op, o, p, S, contab, tail2d) -> Hblock

Internal implementation shared by `melitz_cross_origin_weighted_block!` (serial, uses
`op`'s own single `contab`/`tail2d` scratch) and `melitz_full_weighted_gram_parallel!`
(threaded over origin pairs, each thread passing its OWN thread-local `contab`/`tail2d`
buffer -- required so concurrent pairs never share mutable scratch). No bounds/argument
checks here (callers already validated); no allocation.
"""
function _cross_origin_weighted_block_buf!(Hblock::AbstractMatrix{Float64}, op::MelitzMomentOperator,
                                            o::Int, p::Int, S::AbstractVector{Float64},
                                            contab::AbstractMatrix{Float64}, tail2d::AbstractMatrix{Float64})
    D = op.D
    W = op.W
    bin = op.bin
    rank = op.rank
    coef = op.coef
    z_power = op.sorted_ctx.z_power_original

    @inbounds for b in 1:D+1, a in 1:D+1
        contab[a, b] = 0.0
    end
    @inbounds for s in 1:W
        a = bin[s, o]
        b = bin[s, p]
        contab[a+1, b+1] += S[s] * z_power[s, o] * z_power[s, p]
    end

    @inbounds for a in D:-1:0
        for b in D:-1:0
            v = contab[a+1, b+1]
            a < D && (v += tail2d[a+2, b+1])
            b < D && (v += tail2d[a+1, b+2])
            (a < D && b < D) && (v -= tail2d[a+2, b+2])
            tail2d[a+1, b+1] = v
        end
    end

    @inbounds for d in 1:D
        rd = rank[d, o]
        cd = coef[o, d]
        for dp in 1:D
            rdp = rank[dp, p]
            Hblock[d, dp] = cd * coef[p, dp] * tail2d[rd+1, rdp+1]
        end
    end
    return Hblock
end

"""
    melitz_full_weighted_gram!(H, op, S) -> H

Fills the UPPER TRIANGLE ONLY of the complete `(1+num_moments) x (1+num_moments)` weighted
Gram matrix `G_full' * Diagonal(S) * G_full`, `G_full = [ones(W) G]` -- index 1 is the
`zeta`/normalization column, indices `2..1+num_moments` are `G`'s own trade+link columns,
matching `cc_algo/PsiObjectiveBundle.jl`'s `H[:,2:1+outer_constr_index]` convention exactly
(the same `outer_constr_index` slab `hessian!` builds via `BLAS.gemm!` in the dense
reference). This is the FULL Hessian, not merely the same-origin diagonal block.

Decomposition (`G_trade = R_trade - ones(W)*lambda'`, so `G_trade'*S*G_trade = R_trade'*S*
R_trade - (R_trade'*S)*lambda' - lambda*(S'*R_trade) + (S'*ones)*lambda*lambda'`):

1. `g_S = G' * S` (one `mul_Gt!` call) -- gives BOTH the normalization-trade/link row
   directly (`H[1, 2:end] = g_S`) AND, added back to `lambda*sum(S)`, the pure
   `R_trade' * S` vector (`Rt_S`) the rank-one correction needs (`R'v = G'v +
   lambda*sum(v)`, since `G'v = R'v - lambda*sum(v)` by definition).
2. `g_Sell = G' * (S.*ell)` (a second `mul_Gt!` call) -- gives the trade-link column
   (`H[2:1+D^2, end] `) directly, and the link-link entry (`H[end,end] = dot(ell,
   S.*ell)`) as its own link-column output.
3. `melitz_same_origin_weighted_block!`/`melitz_cross_origin_weighted_block!` give
   `R_trade'*S*R_trade`'s diagonal/cross blocks; the rank-one correction
   `- Rt_S[o,d]*lambda[p,dp] - lambda[o,d]*Rt_S[p,dp] + sum(S)*lambda[o,d]*lambda[p,dp]`
   is added directly at assembly time for EVERY `(o,d),(p,dp)` pair with `trade_index[o,d]
   <= trade_index[p,dp]` (same-origin `d<=dp`, or `o<p` -- both upper-triangle-valid by
   `MelitzMomentLayout`'s own per-origin-contiguous column convention).

Zero-allocation after `op`'s own construction (`op.gS_scratch`/`gSell_scratch`/
`Sell_scratch`/`RtS_scratch`/`hblock_scratch`/`contab`/`tail2d` are the only scratch
touched; `sum(S)` is a plain reduction over an existing `Vector{Float64}`, no allocation).
"""
function melitz_full_weighted_gram!(H::AbstractMatrix{Float64}, op::MelitzMomentOperator,
                                     S::AbstractVector{Float64})
    D = op.D
    W = op.W
    K = op.layout.num_moments
    size(H) == (K + 1, K + 1) || throw(ArgumentError(
        "melitz_full_weighted_gram!: H must be (1+num_moments) x (1+num_moments) = $(K+1) x $(K+1)"))
    length(S) == W || throw(ArgumentError("melitz_full_weighted_gram!: S must have length W"))

    trade_index = op.layout.trade_index
    lambda = op.lambda
    link_idx = op.layout.focal_link_index

    g_S = op.gS_scratch
    g_Sell = op.gSell_scratch
    Sell = op.Sell_scratch
    Rt_S = op.RtS_scratch
    Hblock = op.hblock_scratch
    ell = op.ell

    mul_Gt!(g_S, op, S)
    sumS = 0.0
    @inbounds for s in 1:W
        sumS += S[s]
        Sell[s] = S[s] * ell[s]
    end
    mul_Gt!(g_Sell, op, Sell)

    H[1, 1] = sumS
    @inbounds for k in 1:K
        H[1, 1+k] = g_S[k]
    end

    @inbounds for o in 1:D, d in 1:D
        Rt_S[o, d] = g_S[trade_index[o, d]] + lambda[o, d] * sumS
    end

    @inbounds for o in 1:D
        melitz_same_origin_weighted_block!(Hblock, op, o, S)
        for d in 1:D
            col_d = trade_index[o, d]
            for dp in d:D
                col_dp = trade_index[o, dp]
                rsr = Hblock[d, dp]
                H[1+col_d, 1+col_dp] = rsr - Rt_S[o, d] * lambda[o, dp] -
                                        lambda[o, d] * Rt_S[o, dp] + sumS * lambda[o, d] * lambda[o, dp]
            end
        end
    end

    @inbounds for o in 1:D, p in o+1:D
        melitz_cross_origin_weighted_block!(Hblock, op, o, p, S)
        for d in 1:D
            col_d = trade_index[o, d]
            for dp in 1:D
                col_dp = trade_index[p, dp]
                rsr = Hblock[d, dp]
                H[1+col_d, 1+col_dp] = rsr - Rt_S[o, d] * lambda[p, dp] -
                                        lambda[o, d] * Rt_S[p, dp] + sumS * lambda[o, d] * lambda[p, dp]
            end
        end
    end

    @inbounds for k in 1:D*D
        H[1+k, 1+link_idx] = g_Sell[k]
    end
    H[1+link_idx, 1+link_idx] = g_Sell[link_idx]

    return H
end

"""
    melitz_full_weighted_gram_parallel!(H, op, S) -> H

Threaded analogue of `melitz_full_weighted_gram!` (Phase 7 parallelization): the two
`O(D)`/`O(D^2)`-unit-of-work loops (same-origin diagonal blocks over `o in 1:D`; cross-origin
blocks over `o in 1:D-1` with a serial inner `p in o+1:D` loop) each run as their own
`Threads.@threads :static` region, using PER-THREAD scratch (`op.hblock_t`/`contab_t`/
`tail2d_t`/`binsum2_t`/`tail2_t`, indexed by `Threads.threadid()`, sized to
`Threads.maxthreadid()` at `op`'s own construction time) -- same disjoint-write, no-reduction
safety argument `sorted_tail.jl`'s own `melitz_moments_sorted_tail_parallel!` and
`direct_gradient.jl`'s `:B_direct_argument_parallel` already established for this codebase
(each thread's assigned origin `o` maps to a UNIQUE, disjoint block of `H`'s rows/columns
via `MelitzMomentLayout`'s own per-origin-contiguous column convention, so no two threads
ever write the same `H` entry). The two scalar `mul_Gt!` calls (`g_S`, `g_Sell`) and the
rank-one-correction assembly loop remain serial (already `O(W*D)`/`O(D^2)`, a small share of
the total `O(W*D^2)` cross-origin-contingency-table cost this parallelizes).

Callers running this under KNITRO must set `BLAS.set_num_threads(1)` for the duration (this
repo's own standing convention, `feedback-openblas-threads-hard-cap-violation` memory) --
NOT enforced here since this function has no BLAS calls of its own (matching
`melitz_moments_sorted_tail_parallel!`'s own precedent).
"""
function melitz_full_weighted_gram_parallel!(H::AbstractMatrix{Float64}, op::MelitzMomentOperator,
                                              S::AbstractVector{Float64})
    D = op.D
    W = op.W
    K = op.layout.num_moments
    size(H) == (K + 1, K + 1) || throw(ArgumentError(
        "melitz_full_weighted_gram_parallel!: H must be (1+num_moments) x (1+num_moments) = $(K+1) x $(K+1)"))
    length(S) == W || throw(ArgumentError("melitz_full_weighted_gram_parallel!: S must have length W"))
    nt = Threads.maxthreadid()
    length(op.hblock_t) >= nt || throw(ArgumentError(
        "melitz_full_weighted_gram_parallel!: op was built with fewer thread-local buffers " *
        "($(length(op.hblock_t))) than the current Threads.maxthreadid() ($nt) -- rebuild " *
        "the operator (build_melitz_moment_operator) under the current thread pool"))

    Rt_S = op.RtS_scratch
    sumS = _gram_prep_serial!(H, op, S)

    Threads.@threads :static for o in 1:D
        _same_origin_gram_task!(H, op, o, S, Rt_S, sumS)
    end

    if D >= 2
        Threads.@threads :static for o in 1:D-1
            _cross_origin_gram_task!(H, op, o, S, Rt_S, sumS)
        end
    end

    _gram_finish_link_serial!(H, op)

    return H
end

"""
    _gram_prep_serial!(H, op, S) -> sumS::Float64

Serial prep phase of `melitz_full_weighted_gram_parallel!`, factored into its own top-level
function so that the function CONTAINING the `Threads.@threads` blocks has as few
pre-existing local bindings as possible around them -- this session's own testing found
that leaving this code inline in the same function as the `@threads` loops (even though it
runs strictly BEFORE them, never inside the closure) still triggered several MB of spurious
allocation at real D=20 scale, consistent with Julia's closure-conversion boxing variables
that are live across an enclosing `@threads` block; splitting it into a separate function
call eliminates that channel entirely (the CALLING function's only locals touched near the
`@threads` blocks become `Rt_S`/`sumS`, both passed as plain function arguments into
`_same_origin_gram_task!`/`_cross_origin_gram_task!`, never captured by a closure).
Computes `H[1,1]`/`H[1,2:end]` (normalization row) and `op.RtS_scratch` (`R_trade'*S`, via
`G'*S + lambda*sum(S)`) in place; returns `sum(S)`.
"""
function _gram_prep_serial!(H::AbstractMatrix{Float64}, op::MelitzMomentOperator, S::AbstractVector{Float64})
    D = op.D
    W = op.W
    K = op.layout.num_moments
    trade_index = op.layout.trade_index
    lambda = op.lambda
    g_S = op.gS_scratch
    g_Sell = op.gSell_scratch
    Sell = op.Sell_scratch
    Rt_S = op.RtS_scratch
    ell = op.ell

    mul_Gt!(g_S, op, S)
    sumS = 0.0
    @inbounds for s in 1:W
        sumS += S[s]
        Sell[s] = S[s] * ell[s]
    end
    mul_Gt!(g_Sell, op, Sell)

    H[1, 1] = sumS
    @inbounds for k in 1:K
        H[1, 1+k] = g_S[k]
    end
    @inbounds for o in 1:D, d in 1:D
        Rt_S[o, d] = g_S[trade_index[o, d]] + lambda[o, d] * sumS
    end
    return sumS
end

"""
    _gram_finish_link_serial!(H, op) -> nothing

Serial finish phase of `melitz_full_weighted_gram_parallel!` (trade-link column + link-link
entry, from `op.gSell_scratch`, already filled by `_gram_prep_serial!`'s own `mul_Gt!`
call) -- factored out for the same closure-boxing-avoidance reason as `_gram_prep_serial!`.
"""
function _gram_finish_link_serial!(H::AbstractMatrix{Float64}, op::MelitzMomentOperator)
    D = op.D
    link_idx = op.layout.focal_link_index
    g_Sell = op.gSell_scratch
    @inbounds for k in 1:D*D
        H[1+k, 1+link_idx] = g_Sell[k]
    end
    H[1+link_idx, 1+link_idx] = g_Sell[link_idx]
    return nothing
end

"""
    _same_origin_gram_task!(H, op, o, S, Rt_S, sumS) -> nothing

One `Threads.@threads` iteration's worth of work for `melitz_full_weighted_gram_parallel!`'s
same-origin loop, factored into a standalone top-level function (not inlined in the `@threads`
closure) -- matching `direct_gradient.jl`'s own `_direct_coordinate_grad` precedent in this
codebase, which avoids a real Julia closure-capture-boxing allocation this session's own
testing found when the loop body was written inline (`@allocated` showed several MB for a
`D=20` call despite every touched array being preallocated scratch -- boxing of captured
locals inside the `Threads.@threads` closure, not a per-element workspace allocation; moving
the body to a top-level function with explicit arguments eliminated it, confirmed zero-alloc
in this file's own test suite).
"""
function _same_origin_gram_task!(H::AbstractMatrix{Float64}, op::MelitzMomentOperator, o::Int,
                                  S::AbstractVector{Float64}, Rt_S::AbstractMatrix{Float64}, sumS::Float64)
    D = op.D
    tid = Threads.threadid()
    trade_index = op.layout.trade_index
    lambda = op.lambda
    Hblock = op.hblock_t[tid]
    _same_origin_weighted_block_buf!(Hblock, op, o, S, op.binsum2_t[tid], op.tail2_t[tid])
    @inbounds for d in 1:D
        col_d = trade_index[o, d]
        for dp in d:D
            col_dp = trade_index[o, dp]
            rsr = Hblock[d, dp]
            H[1+col_d, 1+col_dp] = rsr - Rt_S[o, d] * lambda[o, dp] -
                                    lambda[o, d] * Rt_S[o, dp] + sumS * lambda[o, d] * lambda[o, dp]
        end
    end
    return nothing
end

"""
    _cross_origin_gram_task!(H, op, o, S, Rt_S, sumS) -> nothing

One `Threads.@threads` iteration's worth of work for `melitz_full_weighted_gram_parallel!`'s
cross-origin loop (origin `o` against every `p in o+1:D`, serial inner loop -- see
`_same_origin_gram_task!`'s own docstring for why this is a standalone top-level function).
"""
function _cross_origin_gram_task!(H::AbstractMatrix{Float64}, op::MelitzMomentOperator, o::Int,
                                   S::AbstractVector{Float64}, Rt_S::AbstractMatrix{Float64}, sumS::Float64)
    D = op.D
    tid = Threads.threadid()
    trade_index = op.layout.trade_index
    lambda = op.lambda
    Hblock = op.hblock_t[tid]
    contab = op.contab_t[tid]
    tail2d = op.tail2d_t[tid]
    for p in o+1:D
        _cross_origin_weighted_block_buf!(Hblock, op, o, p, S, contab, tail2d)
        @inbounds for d in 1:D
            col_d = trade_index[o, d]
            for dp in 1:D
                col_dp = trade_index[p, dp]
                rsr = Hblock[d, dp]
                H[1+col_d, 1+col_dp] = rsr - Rt_S[o, d] * lambda[p, dp] -
                                        lambda[o, d] * Rt_S[p, dp] + sumS * lambda[o, d] * lambda[p, dp]
            end
        end
    end
    return nothing
end
