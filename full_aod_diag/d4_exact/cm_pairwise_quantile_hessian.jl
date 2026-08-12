# ================================================================================================
# Hessian for the CM + pairwise-quantile family (family #7, 2026-08-12).
#
# THE ONE FACT THAT ORGANIZES EVERYTHING. The inner dual index is LINEAR in the inner variables:
# `r_w = -zeta - E_w'lambda_E - G^R_w'lambda_R - G^CM_w'lambda_CM`. So with `h_w = Psi''(r_w)`,
#
#       H  =  (1/W) * M' diag(h) M ,        M = [ 1 | E | G^R | G^CM ]     (W x n)
#
# exactly -- no second-order term, because `r` has no curvature in `x`. Every block is therefore a
# weighted cross-moment of feature columns, and since every restriction column is a bin indicator (or
# a cumulative bin indicator), every block is a contraction of low-order CONTINGENCY TABLES. That is
# why no `W x n` object is ever built, and it is the same reason CM and PQ each already have
# table-based Hessians.
#
# WHAT IS REUSED, AND WHY IT IS EXACT RATHER THAN CONVENIENT.
# This family's restriction rows are a strict SUBSET of the standalone pairwise-quantile family's
# rows, evaluated at a `mu` that happens to be COMMON across origins:
#
#     this family's level row a   ==  standalone PQ's marginal row (ref, a)   with mu_{ref,a} = mu_a
#     this family's pair row      ==  standalone PQ's pair row                with mu_{o,a}*mu_{p,b}
#                                                                             = mu_a * mu_b
#
# So if a `PairwiseQuantileMassState` is filled with the shared `mu` REPLICATED into every origin's
# row (`cmpq_replicate_shared_mu!`), then the standalone family's own already-gated Hessian machinery
# -- `build_pairwise_quantile_hessian_tables!`, `fill_pairwise_quantile_hessian_raw!`,
# `center_and_scale_pairwise_quantile_hessian!`, `pairwise_quantile_cross_hessian_block!` -- computes
# THIS family's H_RR and H_E,R as a superset, with identical values, and the dropped per-origin
# marginal rows are simply not read. That is a mathematical identity (checked in the D=4 oracle
# against a dense reference, not asserted), so H_RR and H_E,R need NO new code: only the row map
# `cmpq_to_pq_row` that selects the sub-block. The cost of computing the unread rows is ~5% at
# D=20/L=5 (3120 PQ rows vs this family's 3044) -- deliberately paid, to reuse code that already
# carries 29 dense-oracle checks, a threaded scatter, the T3/T4 canonical dedup, and the `_lo_write!`
# triangle convention.
#
# CM's own two blocks (H_CM,CM and H_E,CM) are likewise supplied by CM's own validated functions
# (`build_bin_tables!`/`prefix_sum_tables!`/`fill_cm_HCC!`, `winner_pair_cross_hessian_cm_block!`).
#
# WHAT IS GENUINELY NEW: the CM x PQ cross block H_R,CM, below. Nothing else.
#
# THE NEW BLOCK'S STRUCTURE. A CM column is a cumulative single-origin indicator contrast,
# `CM_{l,o}(w) = 1{U_o <= z_l} - 1{U_ref <= z_l}` (eq.35), and a restriction row is a PQ bin
# indicator on one origin (level) or two (pair). Their product is therefore a joint indicator over
# at most THREE origins, at MIXED resolution: CM's fine G-cell grid on one origin, PQ's L-bin grid on
# the other one or two. Concretely, with `c_I` the row's own target,
#
#   H_R,CM[I, (l,o)] = (1/W)[ sum_w h_w ind_I(w)(1{U_o<=z_l} - 1{U_ref<=z_l}) - c_I * rcm_{l,o} ]
#   rcm_{l,o}        = sum_w h_w (1{U_o<=z_l} - 1{U_ref<=z_l})
#
# and the two sums come from two new tables, both accumulated at CM-cell resolution and then
# PREFIX-SUMMED once over that axis so every read is O(1):
#
#   Y[k,a,x]        = sum_w h_w 1{bin^CM_x(w)=k} 1{b_ref(w)=a}                 (level rows)
#   X[k,a,b,(x,p)]  = sum_w h_w 1{bin^CM_x(w)=k} 1{b_o(w)=a, b_p(w)=b}         (pair rows)
#
# `X` deliberately ranges over ALL `x in 1..D`, including `x` inside the pair `(o,p)`. Restricting it
# to disjoint `x` and special-casing the overlapping cases against CM's own 2-way table would save
# ~10% of the build and cost a three-branch read path -- exactly where an index error hides. The
# uniform table has ONE read path for every (row, CM column) combination.
#
# eq.36 (`n_families == 2`) reuses the SAME tables, `Pow`-weighted (`Xpow`/`Ypow`), with the
# reflection `sum_{k>l} = total - prefix_l` applied on the read side -- eq.36's indicator is
# `1{U>c}`, not `1{U<=c}` (see `common_marginals_moments.jl`'s own derivation and the bug it fixed).
# No new formula, only a differently-weighted table and a reflected read.
#
# Requires: pairwise_quantile_bin_context.jl, pairwise_quantile_operator.jl,
# pairwise_quantile_hessian.jl, cm_pairwise_quantile_config.jl, cm_pairwise_quantile_moments.jl.
# ================================================================================================

using LinearAlgebra: mul!

"""
    cmpq_to_pq_row(I::Integer, D::Integer, L::Integer, ref::Integer) -> Int

Maps a row index of THIS family's restriction block onto the standalone pairwise-quantile family's
row index for the same moment: level row `a` -> `marginal_row(ref,a,L)`, pair row -> `pair_row(...)`.

STRICTLY INCREASING in `I` (level rows land in `(ref-1)*(L-1)+1 : ref*(L-1)`, which is at or below
`D*(L-1)`, and every pair row lands above it), which is what makes it safe to read the standalone
family's LOWER-triangle-only Hessian at `[sigma(I), sigma(J)]` for `I >= J` without ever needing the
unwritten upper half.
"""
function cmpq_to_pq_row(I::Integer, D::Integer, L::Integer, ref::Integer)
    nc = Int(L) - 1
    i = Int(I)
    if i <= nc
        return marginal_row(Int(ref), i, Int(L))
    end
    return n_marginal_rows(Int(D), Int(L)) + (i - nc)
end

"""
    cmpq_pq_row_map(D, L, ref) -> Vector{Int}

`sig[I] = cmpq_to_pq_row(I, D, L, ref)` materialized once (campaign-lifetime) so the packed write
never recomputes the branch per entry. STRICTLY INCREASING, by `cmpq_to_pq_row`'s own docstring --
and note the tail is AFFINE (`sig[I] = n_marginal_rows(D,L) + I - (L-1)` for every pair row), so a
column walk of `HRR_pq[sig[J], sig[I]]` over `J` is contiguous except for the first `L-1` entries.
"""
function cmpq_pq_row_map(D::Integer, L::Integer, ref::Integer)
    nrow = n_cmpq_restr_rows(Int(D), Int(L))
    sig = Vector{Int}(undef, nrow)
    @inbounds for I in 1:nrow
        sig[I] = cmpq_to_pq_row(I, Int(D), Int(L), Int(ref))
    end
    return sig
end

"""
    cmpq_pk_upper(i, j, n) -> Int

Row-major upper-triangular packed index of `(i,j)`, `i <= j`, within an `n x n` matrix -- the SAME
convention `winner_pair_hessian!` (`core_exact_hessian.jl`) fills its own packed `hee_packed` with,
via a single running counter across both its zeta-row and lambda-lambda loops, and the same closed
form `pairwise_quantile_production.jl::_pk_upper` carries. Duplicated here (rather than reached for
across a file this one does not depend on) so the standalone dense oracle, which loads no production
stack at all, can still gate the packing; the oracle checks it against a literal running counter and
the real-context Hessian gate checks it against `_pk_upper` itself, so the two copies cannot drift
silently.
"""
@inline cmpq_pk_upper(i::Int, j::Int, n::Int) = (i - 1) * (n + 1) - div((i - 1) * i, 2) + (j - i + 1)

"""
    pack_cmpq_hessian!(hess_out, hee_packed, HEQ_pq, HEC, HRR_pq, HRC, HCC, sig,
                       NCORE, n_restr, ncm) -> hess_out

Writes KNITRO's packed ROW-MAJOR UPPER triangle for
`x = [zeta; lambda_E(ncore1); lambda_L(L-1); lambda_P((L-1)^2*npair); lambda_CM(ncm)]`,
i.e. `n = NCORE + n_restr + ncm` with `NCORE = 1 + ncore1`, selecting the correct source block per
`(i,j)` instead of assembling a dense `n x n` first.

Sources, each already in its final (signed, scaled, contrast-applied) form:

| rows \\ cols | `1:NCORE`        | `NCORE+1:nR`          | `nR+1:n`        |
|---|---|---|---|
| `1:NCORE`    | `hee_packed`     | `HEQ_pq[i, sig[.]]`   | `HEC[i,.]`      |
| `NCORE+1:nR` |                  | `HRR_pq[sig,sig]`     | `HRC[.,.]`      |
| `nR+1:n`     |                  |                       | `HCC[.,.]`      |

`HEQ_pq`/`HRR_pq` are the STANDALONE pairwise-quantile family's blocks at a replicated shared `mu`
(see this file's header): `sig` selects this family's sub-block, and because `sig` is strictly
increasing, `HRR_pq[sig[J], sig[I]]` with `J >= I` always lands in the LOWER triangle -- the only
half `fill_pairwise_quantile_hessian_raw!`/`center_and_scale_pairwise_quantile_hessian!` populate.
`HCC` must have BOTH triangles populated (which is what `fill_cm_HCC!` does); it is read by column,
`HCC[jj, ii]`, for the same cache reason.

THREE MEASURED PERFORMANCE PROPERTIES, carried over verbatim from
`pairwisequantile_hess_cb_builder`'s own packed write, where they were measured at D=20/L=10/W=100k
(86.16 s -> the callback stopped being the bottleneck):

 1. COLUMN WALKS. `HRR_pq`/`HCC` are read transposed (`[j, i]`, not `[i, j]`) so the inner loop
    walks a contiguous column of a column-major matrix rather than striding a row. Both blocks are
    symmetric, so the value is identical; for `HRR_pq` it is also the ONLY populated triangle.
 2. CLOSED-FORM ROW OFFSET. `k0(i) = (i-1)*n - (i-1)*(i-2)/2` is the packed index just before row
    `i`, so rows are independent and the loop threads with no reduction and no ordering concern --
    every packed slot belongs to exactly one row. `:dynamic` because upper-triangle rows have
    length `n-i+1` and static chunks are badly imbalanced (~31x at 16 threads).
 3. HOISTED, TYPE-ASSERTED OUTPUT. `hess_out` is taken as a concrete `Vector{Float64}` argument
    rather than re-read from KNITRO's untyped `evalResult` per entry -- that dynamic `getproperty`
    +`setindex!` per entry, not memory, was the entire cost of the standalone family's own packed
    write (~660 ns/entry). The caller does the assertion once.

`HRC`/`HEC` are the only blocks read by ROW here (stride `n_restr`/`NCORE`). Left that way
deliberately for now: at D=20/L=5 that is 5.7M+0.7M of 14.1M packed entries, against a table BUILD
of 380M increments, so a transpose-on-fill would be optimizing the wrong term. Revisit only with a
per-callback block profile in hand.
"""
function pack_cmpq_hessian!(hess_out::Vector{Float64}, hee_packed::Vector{Float64},
        HEQ_pq::AbstractMatrix{Float64}, HEC::AbstractMatrix{Float64},
        HRR_pq::AbstractMatrix{Float64}, HRC::AbstractMatrix{Float64},
        HCC::AbstractMatrix{Float64}, sig::Vector{Int},
        NCORE::Int, n_restr::Int, ncm::Int)
    nR = NCORE + n_restr
    n = nR + ncm
    length(hess_out) == div(n * (n + 1), 2) ||
        error("pack_cmpq_hessian!: hess buffer is $(length(hess_out)) long, expected " *
              "n*(n+1)/2 = $(div(n * (n + 1), 2)) for n=$n")
    length(hee_packed) == div(NCORE * (NCORE + 1), 2) ||
        error("pack_cmpq_hessian!: hee_packed is $(length(hee_packed)) long, expected " *
              "$(div(NCORE * (NCORE + 1), 2)) for NCORE=$NCORE")
    length(sig) == n_restr ||
        error("pack_cmpq_hessian!: length(sig)=$(length(sig)) != n_restr=$n_restr")
    size(HEC) == (NCORE, ncm) || error("pack_cmpq_hessian!: size(HEC)=$(size(HEC)) != ($NCORE,$ncm)")
    size(HRC) == (n_restr, ncm) || error("pack_cmpq_hessian!: size(HRC)=$(size(HRC)) != ($n_restr,$ncm)")
    size(HCC) == (ncm, ncm) || error("pack_cmpq_hessian!: size(HCC)=$(size(HCC)) != ($ncm,$ncm)")
    size(HEQ_pq, 1) == NCORE ||
        error("pack_cmpq_hessian!: size(HEQ_pq,1)=$(size(HEQ_pq,1)) != NCORE=$NCORE")

    Threads.@threads :dynamic for i in 1:n
        k = (i - 1) * n - div((i - 1) * (i - 2), 2)
        @inbounds if i <= NCORE
            for j in i:NCORE
                k += 1
                hess_out[k] = hee_packed[cmpq_pk_upper(i, j, NCORE)]
            end
            for j in NCORE+1:nR
                k += 1
                hess_out[k] = HEQ_pq[i, sig[j-NCORE]]
            end
            for j in nR+1:n
                k += 1
                hess_out[k] = HEC[i, j-nR]
            end
        elseif i <= nR
            ii = i - NCORE
            si = sig[ii]
            for j in i:nR
                hess_out[k+j-i+1] = HRR_pq[sig[j-NCORE], si]   # column walk, lower triangle
            end
            k += nR - i + 1
            for j in nR+1:n
                k += 1
                hess_out[k] = HRC[ii, j-nR]
            end
        else
            ii = i - nR
            @simd for j in i:n
                hess_out[k+j-i+1] = HCC[j-nR, ii]              # column walk, both triangles filled
            end
        end
    end
    return hess_out
end

"""
    cmpq_replicate_shared_mu!(pq_state::PairwiseQuantileMassState, state::CMPQMassState) -> pq_state

Fills every origin's row of a standalone-family mass state with THIS family's single shared `mu`, so
that the standalone family's target vector (`mu_{o,a}`, `mu_{o,a}*mu_{p,b}`) reduces exactly to this
family's (`mu_a`, `mu_a*mu_b`).

This is the adapter that makes the reuse in this file's header an identity rather than an
approximation. It is NOT a way of turning this family into the standalone one: the standalone family's
extra rows are computed and then never read, and this family's OUTER coordinates remain the `L-1`
shared masses (the state written here is inner-solve scratch, refreshed per outer point).
"""
function cmpq_replicate_shared_mu!(pq_state::PairwiseQuantileMassState, state::CMPQMassState)
    D, nc = size(pq_state.mu)
    length(state.mu) == nc ||
        error("cmpq_replicate_shared_mu!: shared mu has length $(length(state.mu)), expected $nc")
    @inbounds for a in 1:nc, o in 1:D
        pq_state.mu[o, a] = state.mu[a]
    end
    @inbounds for o in 1:D
        acc = 0.0
        for a in 1:nc
            acc += state.mu[a]
            pq_state.Pcum[o, a] = acc
        end
        pq_state.mu_last[o] = state.mu_last
    end
    return pq_state
end

"Linear index into the `X` table's combo axis for (CM origin `x`, pair index `pidx`): `(pidx-1)*D + x`.
`x` fastest, so one pair's whole `D`-slice is contiguous -- the order the build loop writes it in."
@inline cmpq_xp_index(x::Integer, pidx::Integer, D::Integer) = (Int(pidx) - 1) * Int(D) + Int(x)

"""
    CMPQCrossHessTables(D, npair, L, Gbins; n_families)

The new mixed-resolution tables for the CM x PQ cross block, campaign-lifetime SHAPE (rebuilt in
value every Hessian callback, never reallocated). `Gbins` is CM's number of BINS (`cm_grid_size`,
i.e. `Lcm + 1`).

  `X`    `Gbins x (L-1) x (L-1) x (D*npair)`  pair rows      (prefix-summed over the CM axis)
  `Y`    `Gbins x (L-1) x D`                  level rows     (prefix-summed over the CM axis)
  `Hcm`  `D x Gbins`                          CM's own h-weighted bin histogram, prefix-summed --
                                              supplies `rcm_{l,o} = Hcm[o,l] - Hcm[ref,l]`
plus `Xpow`/`Ypow`/`Hcmpow` twins when `n_families == 2` (eq.36's `Pow`-weighted versions).

SIZE, so nobody is surprised at D=20: `X` is `50 * 4 * 4 * 3800 = 3.04M` doubles (24 MB) at
`L=5, G=50`, and `15.4M` (123 MB) at `L=10`. Doubled again for `n_families=2`. That is the same order
as the standalone family's own `T4` and is not the binding constraint; the BUILD is (see the builder).
"""
mutable struct CMPQCrossHessTables
    X::Array{Float64,4}
    Y::Array{Float64,3}
    Hcm::Matrix{Float64}
    Xpow::Union{Nothing,Array{Float64,4}}
    Ypow::Union{Nothing,Array{Float64,3}}
    Hcmpow::Union{Nothing,Matrix{Float64}}
    Gbins::Int
    # Per-thread `nrow x nO` scratch for `fill_cmpq_cm_cross_block!`'s per-threshold-block raw slab.
    # ONE PER THREAD, owned here (campaign-lifetime) rather than allocated inside the fill: that
    # allocation was 463 KB on EVERY Hessian callback at D=20/L=5, against this codebase's standing
    # "no per-callback vectors, matrices, closures" discipline.
    #
    # SIZED BY `Threads.maxthreadid()`, NOT `Threads.nthreads()`, and that distinction is a real bug
    # this file hit on its first run: `nthreads()` counts only the DEFAULT pool, while `threadid()`
    # can land anywhere in `1:maxthreadid()` -- under `julia --threads=8` a `:dynamic` task ran on
    # thread 9 (the interactive-pool thread) and indexed one past the end. `:static` would have
    # pinned ids into `1:nthreads()` and hidden it; `:dynamic` did not. Any threadid-indexed scratch
    # in this codebase sized by `nthreads()` and consumed under `:dynamic` has the same exposure.
    blk::Vector{Matrix{Float64}}
end

function CMPQCrossHessTables(D::Int, npair::Int, L::Int, Gbins::Int; n_families::Int, nO::Int,
                             nthreads_use::Int = max(Threads.maxthreadid(), Threads.nthreads()))
    nc = L - 1
    n_families in (1, 2) || error("CMPQCrossHessTables: n_families must be 1 or 2, got $n_families")
    nO >= 1 || error("CMPQCrossHessTables: nO must be >= 1, got $nO")
    fam2 = n_families == 2
    nrow = n_cmpq_restr_rows(D, L)
    nt = max(1, nthreads_use)
    return CMPQCrossHessTables(zeros(Gbins, nc, nc, D * npair), zeros(Gbins, nc, D), zeros(D, Gbins),
        fam2 ? zeros(Gbins, nc, nc, D * npair) : nothing,
        fam2 ? zeros(Gbins, nc, D) : nothing,
        fam2 ? zeros(D, Gbins) : nothing, Gbins,
        [Matrix{Float64}(undef, nrow, nO) for _ in 1:nt])
end

"""
    build_cmpq_cross_hess_tables!(tabs, op, Bidx, h, ref; Pow=nothing) -> tabs

ONE per-Hessian-callback refresh of the mixed-resolution tables, weighted by `h = Psi''(r_w)`, then
prefix-summed over the CM axis in place (so every read in
`fill_cmpq_cm_cross_block!` is O(1), and eq.36's reflection is `total - prefix`).

COST AND THREADING. The `X` accumulation is `O(W * npair * D)` -- 380M increments at
D=20/W=100,000 -- which is the same order as the standalone family's own `T4` scatter (484M) and is
the dominant new cost this family adds to the Hessian callback. Threaded over `pidx`, which is the
right axis for two independent reasons: each `pidx` owns a DISJOINT slice `X[:,:,:,(pidx-1)*D+1 :
pidx*D]` so there is no reduction and no per-thread copy at all, and a thread then works inside one
`Gbins x nc x nc x D` slab (`50*4*4*20 = 16,000` doubles = 128 KB at L=5) rather than striding the
whole table. `:dynamic` because the per-`pidx` cost is uniform but `npair` does not divide evenly
across threads. Each cell is one sequential sum over `w`, so the result is deterministic and
independent of thread count.

`Y` and `Hcm` are `O(W*D)` and built in one serial pass -- 2M increments at D=20/W=100k, i.e. 0.5% of
`X`; threading them would add a reduction for nothing.
"""
function build_cmpq_cross_hess_tables!(tabs::CMPQCrossHessTables, op::PairwiseQuantileOperator,
        Bidx::AbstractMatrix{<:Integer}, h::AbstractVector{Float64}, ref::Int;
        Pow::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    Gb = tabs.Gbins
    length(h) == W || error("build_cmpq_cross_hess_tables!: length(h)=$(length(h)) != W=$W")
    size(Bidx) == (W, D) || error("build_cmpq_cross_hess_tables!: size(Bidx)=$(size(Bidx)) != ($W,$D)")
    (Pow === nothing) == (tabs.Xpow === nothing) ||
        error("build_cmpq_cross_hess_tables!: Pow was $(Pow === nothing ? "not " : "")supplied but the " *
              "tables were built for n_families=$(tabs.Xpow === nothing ? 1 : 2)")

    X = tabs.X; Y = tabs.Y; Hcm = tabs.Hcm
    Xpow = tabs.Xpow; Ypow = tabs.Ypow; Hcmpow = tabs.Hcmpow
    fill!(X, 0.0); fill!(Y, 0.0); fill!(Hcm, 0.0)
    fam2 = Xpow !== nothing
    fam2 && (fill!(Xpow, 0.0); fill!(Ypow, 0.0); fill!(Hcmpow, 0.0))
    bin = op.bin; pairs = op.pairs
    nlast = UInt8(nc)

    # ---- X (and Xpow): threaded over pidx, disjoint slices, no reduction (see docstring) --------
    Threads.@threads :dynamic for pidx in 1:npair
        (o, p) = pairs[pidx]
        base = (pidx - 1) * D
        @inbounds for w in 1:W
            a = bin[w, o]
            a > nlast && continue
            b = bin[w, p]
            b > nlast && continue
            hw = h[w]
            for x in 1:D
                k = Int(Bidx[w, x])
                X[k, a, b, base+x] += hw
                fam2 && (Xpow[k, a, b, base+x] += hw * Pow[w, x])
            end
        end
    end

    # ---- Y, Hcm (and their Pow twins): one serial O(W*D) pass ------------------------------------
    @inbounds for w in 1:W
        hw = h[w]
        aref = bin[w, ref]
        active = aref <= nlast
        for x in 1:D
            k = Int(Bidx[w, x])
            Hcm[x, k] += hw
            fam2 && (Hcmpow[x, k] += hw * Pow[w, x])
            if active
                Y[k, aref, x] += hw
                fam2 && (Ypow[k, aref, x] += hw * Pow[w, x])
            end
        end
    end

    # ---- prefix-sum over the CM axis, in place ---------------------------------------------------
    _cmpq_prefix_X!(X); _cmpq_prefix_Y!(Y); _cmpq_prefix_Hcm!(Hcm)
    if fam2
        _cmpq_prefix_X!(Xpow); _cmpq_prefix_Y!(Ypow); _cmpq_prefix_Hcm!(Hcmpow)
    end
    return tabs
end

function _cmpq_prefix_X!(X::Array{Float64,4})
    Gb, nca, ncb, ncombo = size(X)
    Threads.@threads :dynamic for c in 1:ncombo
        @inbounds for b in 1:ncb, a in 1:nca
            acc = 0.0
            for k in 1:Gb
                acc += X[k, a, b, c]
                X[k, a, b, c] = acc
            end
        end
    end
    return X
end
function _cmpq_prefix_Y!(Y::Array{Float64,3})
    Gb, nca, D = size(Y)
    @inbounds for x in 1:D, a in 1:nca
        acc = 0.0
        for k in 1:Gb
            acc += Y[k, a, x]
            Y[k, a, x] = acc
        end
    end
    return Y
end
function _cmpq_prefix_Hcm!(H::Matrix{Float64})
    D, Gb = size(H)
    @inbounds for x in 1:D
        acc = 0.0
        for k in 1:Gb
            acc += H[x, k]
            H[x, k] = acc
        end
    end
    return H
end

"""
    fill_cmpq_cm_cross_block!(HRC, op, state, tabs, origins, ref, Lcm, R, W) -> HRC

Fills the `n_cmpq_restr_rows(D,L) x ncm` cross block `H_R,CM` (rows = this family's level/pair rows,
columns = CM's stored grid duals, eq.35 block first and eq.36 second when present).

Column layout is CM's own THRESHOLD-MAJOR one, `col(l,oi) = (l-1)*nO + oi` (`oi` indexing `origins`)
-- the layout `reshape(v, nO, L)` reproduces and every CM kernel in this codebase assumes. The
`:orthonormal` contrast is applied per threshold block on the RIGHT (`block * R`), matching
`_fill_cm_HEE!`'s own H_EC convention (`Hraw_EC * cctx.R`); `R === nothing` (`:anchored`) writes the
raw block directly.

eq.36's columns use the REFLECTED read `total - prefix` and the `Pow`-weighted tables -- see this
file's header.
"""
function fill_cmpq_cm_cross_block!(HRC::AbstractMatrix{Float64}, op::PairwiseQuantileOperator,
        state::CMPQMassState, tabs::CMPQCrossHessTables, origins::Vector{Int}, ref::Int, Lcm::Int,
        R::Union{Nothing,AbstractMatrix{Float64}}, W::Int)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nO = length(origins)
    nrow = n_cmpq_restr_rows(D, L)
    fam2 = tabs.Xpow !== nothing
    ncm = (fam2 ? 2 : 1) * nO * Lcm
    size(HRC) == (nrow, ncm) ||
        error("fill_cmpq_cm_cross_block!: size(HRC)=$(size(HRC)) != ($nrow,$ncm)")
    length(state.mu) == nc ||
        error("fill_cmpq_cm_cross_block!: length(state.mu)=$(length(state.mu)) != $nc")
    Lcm + 1 == tabs.Gbins ||
        error("fill_cmpq_cm_cross_block!: Lcm+1=$(Lcm+1) != tabs.Gbins=$(tabs.Gbins) -- CM's level " *
              "count and the table's bin count disagree")

    mu = state.mu
    X = tabs.X; Y = tabs.Y; Hcm = tabs.Hcm
    Xpow = tabs.Xpow; Ypow = tabs.Ypow; Hcmpow = tabs.Hcmpow
    Gb = tabs.Gbins
    invW = 1.0 / W
    nfam = fam2 ? 2 : 1
    size(tabs.blk[1]) == (nrow, nO) ||
        error("fill_cmpq_cm_cross_block!: tabs.blk is $(size(tabs.blk[1])), expected ($nrow,$nO) -- " *
              "the tables were built for a different (D,L,nO)")
    length(tabs.blk) >= Threads.maxthreadid() ||
        error("fill_cmpq_cm_cross_block!: tabs.blk has $(length(tabs.blk)) per-thread slabs but " *
              "Threads.maxthreadid()=$(Threads.maxthreadid()) -- the tables were built under a " *
              "smaller thread pool. Rebuild them, or the :dynamic loop below will index past the end.")

    # THREADED over the flattened (family, threshold-block) index. Safe by construction and with no
    # reduction: block `(fam,l)` writes ONLY columns `colbase + (l-1)*nO+1 : colbase + l*nO` of
    # `HRC`, and those ranges are disjoint across `(fam,l)`. Each task takes its own `nrow x nO`
    # slab from `tabs.blk`, so the raw fill does not race either. `:dynamic` because the per-block
    # cost is uniform but `nfam*Lcm` does not divide evenly across threads.
    #
    # Deterministic: every HRC entry is written exactly once, by one task, from reads of tables that
    # are already complete -- so the result does not depend on thread count. The D=4 dense oracle
    # checks this block against `(1/W) G_R' diag(h) G_CM` on every run, which is what makes that a
    # gated property rather than an argument.
    Threads.@threads :dynamic for lin in 1:(nfam * Lcm)
        fam = div(lin - 1, Lcm) + 1
        l = mod(lin - 1, Lcm) + 1
        colbase = (fam - 1) * nO * Lcm
        blk = tabs.blk[Threads.threadid()]
        begin
            @inbounds for oi in 1:nO
                o = origins[oi]
                # rcm = sum_w h_w * CM_col(l,o); reflected for eq.36.
                rcm = if fam == 1
                    Hcm[o, l] - Hcm[ref, l]
                else
                    (Hcmpow[o, Gb] - Hcmpow[o, l]) - (Hcmpow[ref, Gb] - Hcmpow[ref, l])
                end
                # level rows
                for a in 1:nc
                    raw = if fam == 1
                        Y[l, a, o] - Y[l, a, ref]
                    else
                        (Ypow[Gb, a, o] - Ypow[l, a, o]) - (Ypow[Gb, a, ref] - Ypow[l, a, ref])
                    end
                    blk[cmpq_level_row(a), oi] = (raw - mu[a] * rcm) * invW
                end
                # pair rows
                for pidx in 1:npair
                    co = cmpq_xp_index(o, pidx, D)
                    cr = cmpq_xp_index(ref, pidx, D)
                    for b in 1:nc
                        mub = mu[b]
                        for a in 1:nc
                            raw = if fam == 1
                                X[l, a, b, co] - X[l, a, b, cr]
                            else
                                (Xpow[Gb, a, b, co] - Xpow[l, a, b, co]) -
                                (Xpow[Gb, a, b, cr] - Xpow[l, a, b, cr])
                            end
                            blk[cmpq_pair_row(D, pidx, a, b, L), oi] = (raw - mu[a] * mub * rcm) * invW
                        end
                    end
                end
            end
            cols = colbase + (l - 1) * nO + 1 : colbase + l * nO
            if R === nothing
                @views HRC[:, cols] .= blk
            else
                @views mul!(HRC[:, cols], blk, R)
            end
        end
    end
    return HRC
end

"""
    extract_cmpq_HRR!(HRR, HRR_pq, D, L, ref) -> HRR

Copies the sub-block of the standalone family's centered restriction Hessian corresponding to THIS
family's rows, LOWER TRIANGLE ONLY (`I >= J`), matching `_lo_write!`'s convention so the packed write
reads the same triangle the standalone family's own packed write does.

`HRR_pq` must be the output of `fill_pairwise_quantile_hessian_raw!` followed by
`center_and_scale_pairwise_quantile_hessian!` at a mass state built by `cmpq_replicate_shared_mu!` --
see this file's header for why that makes the copied values exactly this family's own.
"""
function extract_cmpq_HRR!(HRR::AbstractMatrix{Float64}, HRR_pq::AbstractMatrix{Float64},
                            D::Int, L::Int, ref::Int)
    nrow = n_cmpq_restr_rows(D, L)
    size(HRR) == (nrow, nrow) || error("extract_cmpq_HRR!: size(HRR)=$(size(HRR)) != ($nrow,$nrow)")
    npq = n_total_rows(D, L)
    size(HRR_pq) == (npq, npq) || error("extract_cmpq_HRR!: size(HRR_pq)=$(size(HRR_pq)) != ($npq,$npq)")
    Threads.@threads :dynamic for J in 1:nrow
        sJ = cmpq_to_pq_row(J, D, L, ref)
        @inbounds for I in J:nrow
            sI = cmpq_to_pq_row(I, D, L, ref)
            # sigma is increasing, so sI >= sJ and the standalone family's populated LOWER triangle
            # is exactly where this value lives.
            HRR[I, J] = HRR_pq[sI, sJ]
        end
    end
    return HRR
end

"""
    extract_cmpq_HER!(HER, HEQ_pq, D, L, ref) -> HER

Column-selects the standalone family's economic-cross block (`NCORE x n_total_rows(D,L)`) down to
this family's rows (`NCORE x n_cmpq_restr_rows(D,L)`). Same identity, same reason as
`extract_cmpq_HRR!`.
"""
function extract_cmpq_HER!(HER::AbstractMatrix{Float64}, HEQ_pq::AbstractMatrix{Float64},
                            D::Int, L::Int, ref::Int)
    nrow = n_cmpq_restr_rows(D, L)
    NCORE = size(HEQ_pq, 1)
    size(HER) == (NCORE, nrow) || error("extract_cmpq_HER!: size(HER)=$(size(HER)) != ($NCORE,$nrow)")
    @inbounds for J in 1:nrow
        sJ = cmpq_to_pq_row(J, D, L, ref)
        for i in 1:NCORE
            HER[i, J] = HEQ_pq[i, sJ]
        end
    end
    return HER
end
