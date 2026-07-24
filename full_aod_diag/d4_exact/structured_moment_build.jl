# ============================================================================
# Continuation 10, Part 2: structured (rank-one + winner-scatter) dense moment
# chunk construction.
#
# IDENTITY CHECK FIRST (verified against the actual code, not just derived):
# the task brief's schematic  G_{.,d,s} = v_{s,d} * (e_{w_sd} - lambda_hat_{.,d})
# does NOT hold as literally written -- compressed_moments.jl's own header
# (written in a PRIOR phase of this investigation, lines 36-38) already found
# and documented this exact correction:
#
#   "the centering term is DRAW-INDEPENDENT (-P_{od}*denom_d), NOT scaled by
#    the per-draw winner value v_{sd})."
#
# I.e. the correct per-(o,d) cell (before post-processing normalization) is
#
#   r_{s,(o,d)} = v_{s,d} * 1{o = w_{s,d}}  -  P_{o,d} * denom_d
#
# where the SECOND term has no v_{s,d} factor at all. Re-verified independently
# in this task (c10_structured_moment_verify.jl) by direct comparison against
# obj.moments!'s dense G, not merely re-reading the prior header -- confirmed:
# bit-identical (see report).
#
# CONSEQUENCE FOR THE "RANK-ONE" CONSTRUCTION: since the post-processing
# (SW_s row-scale, nrm_j/gdiv_j column-scale, usePMM offset) is applied AFTER
# forming r, the centering term does not simply factor into two rank-one
# pieces the way the brief's schematic implied. Re-deriving with the FULL
# post-processing folded in (see docs/fullA_D20_structured_moment_report.md
# for the algebra):
#
#   G[s,j] = SW[s]*a[j]*1{o(j)=w_{s,d(j)}}*v[s,d(j)]  -  SW[s]*FixedCol[j]
#
#   a[j]        = nrm[j]*gdiv[j]
#   FixedCol[j] = a[j]*Pmat[o(j),d(j)]*denom[d(j)] + usePMM*nrm[j]*PMM[j]     (O(D^2), θ-dependent but DRAW-independent)
#
# The SECOND term, -SW[s]*FixedCol[j], IS genuinely a rank-one outer product
# of SW (length W, or chunk) and FixedCol (length D^2) -- exactly the "rank-one
# BLAS-friendly" component the brief anticipated, just with SW (not v) as the
# per-draw vector, and FixedCol (not lambda_hat alone) as the fixed vector.
# Built via BLAS.ger! (or an equivalent broadcast -- both benchmarked below).
#
# The FIRST term is the winner-scatter: for each (s,d), add
# SW[s]*a[j]*v[s,d] into column j = d+(w_{s,d}-1)*D of row s. Irregular/
# indexed, NOT forced into a BLAS call -- a tight scalar loop, per the task's
# explicit guidance (benchmarked against the alternative of building it as
# a sparse-then-densify step, which was not competitive and is not included
# as a separate variant here, see report).
#
# Winner-finding (O(W*D^2), irreducible) and tie detection are NOT
# re-derived here -- this file builds directly ON TOP of
# `build_compressed_factual` (compressed_moments.jl), reusing its `winner`/
# `wval`/tie-checking UNCHANGED, so tie conventions are identical BY
# CONSTRUCTION (not merely "should agree").
# ============================================================================

using LinearAlgebra: BLAS

"""
    structured_coeffs(cf::CompressedFactual) -> (a, FixedCol)

Precompute the O(D^2) θ-dependent, draw-independent coefficients used by
`structured_fill_chunk!` below: `a[j] = nrm[j]*gdiv[j]` (length ncol, all
inner-dual columns) and `FixedCol[j] = a[j]*Pmat[o,d]*denom[d] +
usePMM*nrm[j]*PMM[j]` (length D^2, bilateral columns only -- the
counterfactual price-index column, if present, is handled separately in
`structured_fill_chunk!` since its "fixed" part is not a function of Pmat/denom).
"""
function structured_coeffs(cf)
    D = cf.D; Ddest = cf.D_dest; ncol = cf.oci - 1
    a = Vector{Float64}(undef, ncol)
    for j in 1:ncol
        a[j] = cf.nrm[j] * cf.gdiv[j]
    end
    FixedCol = Vector{Float64}(undef, D * Ddest)
    for slot in 1:Ddest, o in 1:D
        j = slot + (o - 1) * Ddest
        b = cf.usePMM == 1 ? cf.nrm[j] * cf.PMM[j] : 0.0
        FixedCol[j] = a[j] * cf.Pmat[o, slot] * cf.denom[slot] + b
    end
    return a, FixedCol
end

"""
    structured_fill_chunk!(Gc, cf, a, FixedCol, rows; use_ger=true)

Fill `Gc` (a `length(rows) x (oci-1)` buffer -- may be a chunk view or the
full W matrix) with the EXACT bilateral + counterfactual moment block for
draws `rows`, via:
  1. rank-one fixed-term: `Gc[:,1:D^2] .= -SW[rows] * FixedCol'` (BLAS.ger! if
     `use_ger`, else a broadcast -- both produce the identical result, timed
     separately in the benchmark).
  2. scatter-add of the per-draw winner term (irregular, scalar loop).
  3. the counterfactual price-index column (if present), a direct O(n) fill
     (no winner search needed, computed once already in `cf.cf_raw`).
Bit-for-bit reproduces `materialize_dense_factual!`'s formula (same operations,
reordered/regrouped for BLAS-friendliness), NOT an approximation.
"""
function structured_fill_chunk!(Gc::AbstractMatrix, cf, a::Vector{Float64}, FixedCol::Vector{Float64},
                                 rows::AbstractVector{Int}; use_ger::Bool = true)
    D = cf.D; Ddest = cf.D_dest; n = length(rows)
    size(Gc) == (n, cf.oci - 1) || error("structured_fill_chunk!: size(Gc)=$(size(Gc)) != (n,oci-1)=($n,$(cf.oci-1))")
    SWc = @view cf.SW[rows]

    Gbil = @view Gc[:, 1:D*Ddest]
    if use_ger
        fill!(Gbil, 0.0)
        BLAS.ger!(-1.0, SWc, FixedCol, Gbil)
    else
        @views Gbil .= .-SWc .* FixedCol'
    end

    @inbounds for slot in 1:Ddest
        for li in 1:n
            w = rows[li]
            wo = cf.winner[w, slot]
            j = slot + (wo - 1) * Ddest
            Gc[li, j] += SWc[li] * a[j] * cf.wval[w, slot]
        end
    end

    if cf.cf_col > 0
        j = cf.cf_col
        b_cf = cf.usePMM == 1 ? cf.nrm[j] * cf.PMM[j] : 0.0
        @inbounds for li in 1:n
            s = rows[li]
            Gc[li, j] = cf.SW[s] * (a[j] * cf.cf_raw[s] - b_cf)
        end
    end
    return Gc
end

"""
    structured_dense_factual(cf; use_ger=true) -> Matrix{Float64}

Convenience full-W wrapper around `structured_fill_chunk!` (whole matrix,
`rows = 1:W`), matching `materialize_dense_factual`'s signature/return shape
for direct comparison.
"""
function structured_dense_factual(cf; use_ger::Bool = true)
    W = cf.W; ncol = cf.oci - 1
    G = Matrix{Float64}(undef, W, ncol)
    a, FixedCol = structured_coeffs(cf)
    structured_fill_chunk!(G, cf, a, FixedCol, 1:W; use_ger = use_ger)
    return G
end

"""
    materialize_dense_factual_structured!(Gview, cf::CompressedFactual; use_ger=false) -> Gview

Continuation 10, Section 9 (finalize-architecture, Part A #1): drop-in
replacement for `compressed_moments.jl::materialize_dense_factual!` with the
IDENTICAL signature `(Gview::AbstractMatrix, cf::CompressedFactual)`, so every
call site of the old function can swap to this one with no other change.
Internally calls `structured_coeffs`/`structured_fill_chunk!` (this file) over
`rows = 1:cf.W` (the full draw range), reproducing the exact same formula via
the rank-one-fixed-term + winner-scatter decomposition instead of the old
single nested loop -- see `docs/fullA_D20_structured_moment_report.md` for the
~4-23x isolated / ~1.32x full-cold-inner-solve speedup and the bit-for-bit
equivalence check (`c10_structured_moment_verify.jl`). `use_ger=false`
(broadcast) by default -- that report found broadcast ~30% faster than
`BLAS.ger!` for this D^2-wide rank-one fill; `use_ger=true` is available and
produces an identical result if ever preferred.
"""
function materialize_dense_factual_structured!(Gview::AbstractMatrix, cf; use_ger::Bool = false)
    W = cf.W; ncol = cf.oci - 1
    size(Gview) == (W, ncol) || error("materialize_dense_factual_structured!: size(Gview)=$(size(Gview)) != (W,oci-1)=($W,$ncol)")
    a, FixedCol = structured_coeffs(cf)
    structured_fill_chunk!(Gview, cf, a, FixedCol, 1:W; use_ger = use_ger)
    return Gview
end

"""
    fill_K_directgp!(Kview, θ_full, ctx)

Fill the objective column K exactly as `EK_moments_gammanorm_directgp!` does
for `counterExplicit==0, counterType==1` (autarky, direct-gp objective):
`K[s] = θ_full[3+D] * SamplingWeights[s]` -- a trivial O(W) fill, no winner
search needed, reproduced here (rather than re-called from the dense builder)
so `structured_dense_factual`/`materialize_dense_factual!`-based inner solves
can populate obj.H[:,1] without a second full dense `moments!` call.
"""
function fill_K_directgp!(Kview::AbstractVector, θ_full::AbstractVector, ctx)
    γo = ctx.γ
    counterVal = θ_full[3 + ctx.D]
    W = length(Kview)
    SW = γo.SamplingWeights
    @inbounds for s in 1:W
        Kview[s] = counterVal * SW[s]
    end
    return Kview
end
